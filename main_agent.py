from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.sqlite import SqliteSaver
from IPython.display import Image, display
from langchain_openai import ChatOpenAI
from langchain_core.tools import StructuredTool
from weaviate.classes.query import Filter
from dotenv import load_dotenv
import gradio as gr
import requests
import sqlite3
import os
import weaviate
import re

load_dotenv(override=True)

# ==================== 🔧 تابع جستجوی هوشمند ====================

def intelligent_search(query: str, limit: int = 3) -> str:
    """
    جستجوی چندلایه: ابتدا Exact Match، سپس Metadata، آخر Semantic
    
    ✅ بدون تغییر - همان کد قبلی
    """
    print(f"\n{'=' * 60}")
    print(f"🧠 [INTELLIGENT SEARCH] تحلیل کوئری: '{query}'")
    print(f"{'=' * 60}\n")

    client = weaviate.connect_to_local(host="localhost", port=8080)
    questions = client.collections.get("Question")

    section_patterns = {
        r"بیاموز و بگو": "learn_and_say",
        r"واژه\s*سازی": "word_formation",
        r"بخوان و حفظ کن": "poem",
        r"درست و نادرست|درست\s*نادرست": "exercise_true_false",
        r"بازی": "game_activity",
        r"گوش کن و بگو": "listen_and_speak",
        r"فکر کن و بگو": "think_and_say",
        r"پیدا کن و بگو": "find_and_say",
        r"ایستگاه اندیشه": "thinking_station",
        r"بخوان و بیندیش": "read_and_think",
        r"شعر": "poem",
        r"متن اصلی|داستان|متن": "main_story",
    }

    lesson_match = re.search(r"درس\s+(\S+)", query)
    lesson_number = None
    if lesson_match:
        lesson_text = lesson_match.group(1)
        persian_to_num = {
            "اول": "01", "یک": "01", "۱": "01",
            "دوم": "02", "دو": "02", "۲": "02",
            "سوم": "03", "سه": "03", "۳": "03",
            "چهارم": "04", "چهار": "04", "۴": "04",
            "پنجم": "05", "پنج": "05", "۵": "05",
            "ششم": "06", "شش": "06", "۶": "06",
            "هفتم": "07", "هفت": "07", "۷": "07",
            "هشتم": "08", "هشت": "08", "۸": "08",
            "نهم": "09", "نه": "09", "۹": "09",
            "دهم": "10", "ده": "10", "۱۰": "10",
            "یازدهم": "11", "یازده": "11", "۱۱": "11",
            "دوازدهم": "12", "دوازده": "12", "۱۲": "12",
            "سیزدهم": "13", "سیزده": "13", "۱۳": "13",
            "چهاردهم": "14", "چهارده": "14", "۱۴": "14",
            "پانزدهم": "15", "پانزده": "15", "۱۵": "15",
            "شانزدهم": "16", "شانزده": "16", "۱۶": "16",
            "هفدهم": "17", "هفده": "17", "۱۷": "17",
        }
        lesson_number = persian_to_num.get(lesson_text, lesson_text.zfill(2))
        print(f"📚 [ANALYSIS] درس شناسایی شد: {lesson_number}")

    detected_section = None
    for pattern, section_type in section_patterns.items():
        if re.search(pattern, query, re.IGNORECASE):
            detected_section = section_type
            print(f"🎯 [ANALYSIS] بخش شناسایی شد: {section_type}")
            break

    results = []
    search_strategy = "semantic"

    if lesson_number and detected_section:
        search_strategy = "exact_match"
        print(f"\n🎯 [STRATEGY] استراتژی: Exact Match (درس={lesson_number}, بخش={detected_section})\n")
        response = questions.query.fetch_objects(
            filters=(
                Filter.by_property("source").equal(f"lesson_{lesson_number}")
                & Filter.by_property("section_type").equal(detected_section)
            ),
            limit=limit,
        )
        results = response.objects

    elif lesson_number:
        search_strategy = "filtered_semantic"
        print(f"\n🔍 [STRATEGY] استراتژی: Filtered Semantic (درس={lesson_number})\n")
        response = questions.query.near_text(
            query=query,
            filters=Filter.by_property("source").equal(f"lesson_{lesson_number}"),
            limit=limit,
            return_metadata=["distance"],
        )
        results = response.objects

    elif detected_section:
        search_strategy = "type_filtered_semantic"
        print(f"\n🔍 [STRATEGY] استراتژی: Type Filtered Semantic (بخش={detected_section})\n")
        response = questions.query.near_text(
            query=query,
            filters=Filter.by_property("section_type").equal(detected_section),
            limit=limit,
            return_metadata=["distance"],
        )
        results = response.objects

    else:
        search_strategy = "pure_semantic"
        print(f"\n🔍 [STRATEGY] استراتژی: Pure Semantic Search\n")
        response = questions.query.near_text(
            query=query, limit=limit, return_metadata=["distance"]
        )
        results = response.objects

    print(f"📦 [RESULTS] {len(results)} نتیجه با استراتژی '{search_strategy}' پیدا شد\n")

    if not results:
        print("❌ نتیجه‌ای پیدا نشد\n")
        client.close()
        return f"❌ نتیجه‌ای برای '{query}' پیدا نشد."

    formatted_results = []
    related_ids = set()

    for idx, obj in enumerate(results, 1):
        distance = obj.metadata.distance if hasattr(obj.metadata, "distance") else "N/A"
        chunk_data = {
            "content": obj.properties.get("content", ""),
            "section_type": obj.properties.get("section_type", "unknown"),
            "source": obj.properties.get("source", ""),
            "chunk_id": obj.properties.get("chunk_id", ""),
        }
        formatted_results.append(chunk_data)

        print(f"📄 [CHUNK {idx}] ✅ MATCHED")
        print(f"   ├─ نوع: {chunk_data['section_type']}")
        print(f"   ├─ منبع: {chunk_data['source']}")
        print(f"   ├─ فاصله: {distance}")
        print(f"   └─ محتوا: {chunk_data['content'][:80]}...\n")

        related = obj.properties.get("related_chunks", [])
        if related:
            related_ids.update(related)

    if related_ids:
        print(f"🔗 [RELATED] بازیابی {len(related_ids)} چانک مرتبط...\n")
        for related_id in related_ids:
            related_response = questions.query.fetch_objects(
                filters=Filter.by_property("chunk_id").equal(related_id), limit=1
            )
            if related_response.objects:
                obj = related_response.objects[0]
                formatted_results.append({
                    "content": obj.properties.get("content", ""),
                    "section_type": obj.properties.get("section_type", "unknown"),
                    "source": obj.properties.get("source", ""),
                    "is_related": True,
                })

    client.close()

    main = [r for r in formatted_results if not r.get("is_related")]
    related = [r for r in formatted_results if r.get("is_related")]

    print(f"✅ [SUMMARY] {len(main)} اصلی + {len(related)} مرتبط\n")
    print(f"{'=' * 60}\n")

    output_parts = ["📌 **نتایج:**\n"]
    for i, r in enumerate(main):
        output_parts.append(
            f"**بخش {i + 1}** ({r['section_type']} - {r['source']}):\n{r['content']}\n"
        )

    if related:
        output_parts.append("\n🔗 **مرتبط:**\n")
        for i, r in enumerate(related):
            output_parts.append(f"**{i + 1}** ({r['section_type']}):\n{r['content']}\n")

    return "\n---\n".join(output_parts)


# ==================== Telegram Tool ====================

def send_telegram_message(message: str) -> str:
    """ارسال پیام از طریق ربات تلگرام"""
    print(f"\n📱 [TELEGRAM] ارسال پیام: {message[:50]}...")

    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"

    try:
        response = requests.post(
            url, json={"chat_id": chat_id, "text": message, "parse_mode": "HTML"}
        )

        if response.status_code == 200:
            print("✅ [TELEGRAM] ارسال موفق\n")
            return "✅ پیام با موفقیت به تلگرام ارسال شد"
        else:
            print(f"❌ [TELEGRAM] خطا: {response.text}\n")
            return f"❌ خطا: {response.text}"
    except Exception as e:
        print(f"❌ [TELEGRAM] Exception: {str(e)}\n")
        return f"❌ خطا: {str(e)}"


# ==================== Tool Definitions ====================

tool_weaviate = StructuredTool.from_function(
    name="semantic_search",
    func=intelligent_search,
    description="""
    ابزار تخصصی و ضروری برای جستجوی معنایی در پایگاه داده دروس فارسی کلاس دوم.

    از این ابزار **حتماً** استفاده کن وقتی که:
    1. کاربر سوالی درباره محتوای یک درس خاص (داستان، شعر، تمرین) می‌پرسد.
    2. نیاز به یافتن **جزئیات دقیق** مانند نام نویسنده، تاریخ یا جملات کامل متن درس داری.

    ورودی:
        query: **فقط کلمات کلیدی یا عبارت کلیدی بسیار کوتاه و دقیق** برای جستجوی معنایی.
        limit: تعداد نتایج (پیش‌فرض 3)

    خروجی:
        متن شامل نتایج مرتبط و **کامل** از بخش‌های درس.
    """,
)

tool_telegram = StructuredTool.from_function(
    name="send_telegram_message",
    func=send_telegram_message,
    description="""ارسال پیام از طریق تلگرام برای اطلاع‌رسانی فوری""",
)

tools = [tool_weaviate, tool_telegram]


# ==================== 🆕 تابع فیلتر کردن History (نسخه نهایی اصلاح شده) ====================

def filter_messages_for_llm(messages: list, max_pairs: int = 5) -> list:
    """
    پیام‌ها را فیلتر می‌کند:
    1. پیام‌های 'system' (context RAG و Ephemeral) را حذف می‌کند.
    2. تاریخچه را به آخرین N پیام محدود می‌کند تا توکن کاهش یابد.
    """
    print(f"\n🧹 [FILTER] فیلتر کردن history...")
    print(f"   ├─ تعداد پیام‌های ورودی: {len(messages)}")

    # 1. حذف پیام‌های System و Ephemeral (Contextهای RAG)
    filtered_messages = []
    for msg in messages:
        # حذف System Messages و پیام‌هایی که به عنوان موقت (ephemeral) نشانه‌گذاری شده‌اند
        if msg.type == "system" or (hasattr(msg, "additional_kwargs") and msg.additional_kwargs.get("ephemeral")):
            continue
        filtered_messages.append(msg)

    print(f"   ├─ بعد از حذف system/context: {len(filtered_messages)}")

    # 2. محدود کردن به آخرین N پیام (حفظ یکپارچگی Tool Call Chain)
    
    # ما باید همیشه مطمئن شویم که اگر یک Tool Call Chain (AI + Tool + AI) در تاریخچه برش خورده باشد،
    # آن پیام‌های ناقص AI/Tool حذف شوند. با این رویکرد، ما فقط آخرین پیام‌ها را حفظ می‌کنیم.
    
    # max_history_length: تعداد کل پیام‌هایی که مجاز به حفظ آن‌ها هستیم.
    # 5 جفت (Human + AI) = 10 پیام. پیام Human جدید را حساب نمی‌کنیم زیرا همیشه آخرین است.
    max_history_length = max_pairs * 2 # برای 5 جفت قبلی
    
    # آخرین پیام Human (ورودی فعلی) را همیشه در لیست نهایی خواهیم داشت.
    
    # از لیست فیلتر شده، فقط آخرین max_history_length پیام را انتخاب می‌کنیم.
    # این ساده‌ترین راه برای حذف تاریخچه قدیمی و حفظ پیام‌های Tool/AI لازم برای تکمیل حلقه است.
    
    # ما یک Human Message در انتهای filtered_messages داریم که ورودی فعلی است.
    final_messages = filtered_messages[-max_history_length:]

    print(f"   └─ نهایی (حداکثر {max_history_length} پیام): {len(final_messages)}")
    
    return final_messages



# ==================== LangGraph Setup ====================

class State(TypedDict):
    messages: Annotated[list, add_messages]


db_path = "langgraph_weaviate.db"
conn = sqlite3.connect(db_path, check_same_thread=False)
sql_memory = SqliteSaver(conn)

MODEL_NAME = "gpt-4o-mini"
llm = ChatOpenAI(
    base_url=os.getenv("METIS_BASE_URL"),
    api_key=os.getenv("METIS_API_KEY"),
    model=MODEL_NAME,
    temperature=0.3,
)

llm_with_tools = llm.bind_tools(tools)

SYSTEM_PROMPT = """تو یک دستیار آموزشی هوشمند، مهربان و دقیق برای دانش‌آموزان کلاس دوم ابتدایی هستی.
لحن و بیان تو باید همیشه آرام، ساده و در سطح درک کودک باشد.

وظایف اصلی:
1. **اولویت با ابزار است:** برای پاسخ به هر سوال درسی که نیاز به محتوای مشخص دارد (شعر، نام نویسنده، متن درس، تمرین)، **فقط** از نتایج ابزار `semantic_search` استفاده کن.
2. از اطلاعات موجود در مکالمه قبلی استفاده کن و نیازی به جستجوی مجدد نیست مگر موضوع جدیدی مطرح شود.
"""


def route_after_start(state: State) -> str:
    """
    ✅ بدون تغییر - همان کد قبلی
    """
    import re

    messages = state.get("messages", [])
    if not messages:
        return "search"

    last_message = messages[-1].content.strip()
    score = 0

    lesson_pattern = r"(درس|فصل)\s*(اول|دوم|سوم|چهارم|\d+)"
    section_keywords = [
        "بیاموز و بگو", "واژه سازی", "بخوان و حفظ کن",
        "درست و نادرست", "بازی", "گوش کن و بگو",
        "فکر کن و بگو", "پیدا کن و بگو", "ایستگاه اندیشه",
        "بخوان و بیندیش", "شعر",
    ]
    
    if re.search(lesson_pattern, last_message):
        score += 3
    if any(kw in last_message for kw in section_keywords):
        score += 3

    pronouns = ["این", "اون", "همین", "فعالیتش", "ادامه", "اون قسمت"]
    if any(word in last_message for word in pronouns):
        score -= 2

    if len(last_message.split()) < 4:
        score -= 1

    if "?" in last_message or "چیه" in last_message or "بگو" in last_message:
        score += 1

    if score >= 2:
        print(f"🟢 [ROUTER] تصمیم: جستجو انجام شود (امتیاز={score})")
        return "search"
    else:
        print(f"🟡 [ROUTER] تصمیم: گفتگو ادامه یابد (امتیاز={score})")
        return "skip_search"


def mandatory_search(state: State):
    """
    ?📌 تغییر اصلی #2: اضافه کردن نشانگر "ephemeral" به context
    
    ?چیکار می‌کنه:
    ?- یک metadata اضافه می‌کنه که بعداً براساسش پیام رو حذف کنیم
    ?- این پیام فقط برای یکبار استفاده است
    
    ?چرا این کار رو می‌کنیم:
    ?- نتایج جستجو باید فقط یکبار استفاده بشن
    ?- نباید در history بعدی باقی بمونن
    """
    messages = state["messages"]
    last_user_message = None

    for msg in reversed(messages):
        if msg.type == "human":
            last_user_message = msg.content
            break

    if not last_user_message:
        return state

    print(f"\n{'🔄' * 30}")
    print(f"💬 [USER INPUT] سوال کاربر: '{last_user_message}'")
    print(f"{'🔄' * 30}")

    search_result = intelligent_search(last_user_message, limit=3)

    # 🆕 اضافه کردن metadata برای شناسایی context موقت
    from langchain_core.messages import SystemMessage
    
    context_message = SystemMessage(
        content=f"📚 نتایج جستجو:\n\n{search_result}\n\n⚠️ از این اطلاعات استفاده کن.",
        additional_kwargs={"ephemeral": True}  # 🆕 نشانگر موقت بودن
    )

    print(f"✅ [CONTEXT] Context موقت به مدل ارسال شد\n")

    return {"messages": [context_message]}


def chatbot(state: State):
    messages = state["messages"]

    # 1. پیام‌های خام را فیلتر می‌کند (Context RAG حذف می‌شود، تاریخچه قدیمی برش می‌خورد)
    filtered_msgs = filter_messages_for_llm(messages, max_pairs=5)
    
    # 2. System Prompt را اضافه می‌کند
    from langchain_core.messages import SystemMessage
    final_messages = [SystemMessage(content=SYSTEM_PROMPT)] + filtered_msgs
    
    # 3. به LLM ارسال می‌شود
    response = llm_with_tools.invoke(final_messages)
    
    return {"messages": [response]}


# ==================== Build Graph ====================

graph_builder = StateGraph(State)
graph_builder.add_node("mandatory_search", mandatory_search)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", ToolNode(tools=tools))

graph_builder.add_conditional_edges(
    START, route_after_start, {"search": "mandatory_search", "skip_search": "chatbot"}
)
graph_builder.add_edge("mandatory_search", "chatbot")
graph_builder.add_conditional_edges("chatbot", tools_condition)
graph_builder.add_edge("tools", "chatbot")

graph = graph_builder.compile(checkpointer=sql_memory)

try:
    display(Image(graph.get_graph().draw_mermaid_png()))
except:
    print("⚠️ نمودار گراف قابل نمایش نیست")


# ==================== Gradio Interface ====================

config = {"configurable": {"thread_id": "1"}}


def chat(user_input: str, history):
    """
    ✅ بدون تغییر - همان کد قبلی
    """
    try:
        print(f"\n{'🎯' * 30}")
        print(f"🚀 [SESSION START] شروع پردازش درخواست جدید")
        print(f"{'🎯' * 30}\n")

        result = graph.invoke(
            {"messages": [{"role": "user", "content": user_input}]}, config=config
        )

        final_response = result["messages"][-1].content

        print(f"\n{'✨' * 30}")
        print(f"✅ [SESSION END] پاسخ نهایی آماده شد")
        print(f"📝 [RESPONSE] {final_response[:100]}...")
        print(f"{'✨' * 30}\n")

        return final_response
    except Exception as e:
        print(f"\n❌ [ERROR] خطای کلی: {str(e)}\n")
        return f"❌ خطا: {str(e)}"


if __name__ == "__main__":
    interface = gr.ChatInterface(
        chat,
        type="messages",
        title="🎓 دستیار آموزشی هوشمند",
        description="سوالات درسی خود را بپرسید یا املا بخواهید!",
        examples=[
            "یک املا از درس اول برام بساز",
            "درس اول درباره چی بود؟",
            "شعر درس اول رو برام بخون",
            "تمرین درست و نادرست درس اول",
        ],
        theme="soft",
    )
    interface.launch(share=False)

