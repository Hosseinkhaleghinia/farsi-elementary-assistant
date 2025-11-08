"""
اسکریپت Setup برای Weaviate
این فایل را فقط یکبار یا هنگام آپدیت داده‌ها اجرا کنید
"""

import weaviate
import os
import re
from typing import List
from uuid import uuid4
from weaviate.classes.config import Configure, Property, DataType


# ==================== تابع چانک کردن پیشرفته ====================

def chunk_by_semantic_sections(text: str, lesson_name: str = "unknown") -> List[dict]:
    """تقسیم متن درس بر اساس ساختار معنایی و ارتباط بین بخش‌ها"""
    
    chunks = []
    lines = text.strip().split('\n')
    
    current_section = []
    section_type = "unknown"
    importance = "medium"
    lesson_id = f"lesson_{lesson_name}"
    
    for line in lines:
        line = line.strip()
        if not line:
            # جداکننده بخش
            if current_section and len('\n'.join(current_section).strip()) > 10:
                chunks.append({
                    "id": str(uuid4()),
                    "lesson_id": lesson_id,
                    "content": '\n'.join(current_section),
                    "section_type": section_type,
                    "importance": importance,
                    "related_chunks": []
                })
                current_section = []
                section_type = "unknown"
                importance = "medium"
            continue
        
        # تشخیص فصل
        if line.startswith("فصل"):
            if current_section:
                chunks.append({
                    "id": str(uuid4()),
                    "lesson_id": lesson_id,
                    "content": '\n'.join(current_section),
                    "section_type": section_type,
                    "importance": importance,
                    "related_chunks": []
                })
                current_section = []
            current_section.append(line)
            section_type = "chapter_title"
            importance = "high"
        
        # تشخیص عنوان درس
        elif re.search(r"درس\s+\S+", line):
            if current_section:
                chunks.append({
                    "id": str(uuid4()),
                    "lesson_id": lesson_id,
                    "content": '\n'.join(current_section),
                    "section_type": section_type,
                    "importance": importance,
                    "related_chunks": []
                })
                current_section = []
            current_section.append(line)
            section_type = "lesson_title"
            importance = "high"
        
        # تشخیص بخش‌های خاص
        elif "درست" in line and "نادرست" in line:
            section_type = "exercise_true_false"
            importance = "medium"
            current_section.append(line)
        
        elif "گوش کن و بگو" in line:
            section_type = "listen_and_speak"
            importance = "medium"
            current_section.append(line)
        
        elif "پیدا کن و بگو" in line:
            section_type = "find_and_say"
            importance = "medium"
            current_section.append(line)
        
        elif "فکر کن و بگو" in line:
            section_type = "think_and_say"
            importance = "medium"
            current_section.append(line)
        
        elif "ایستگاه اندیشه" in line:
            section_type = "thinking_station"
            importance = "high"
            current_section.append(line)
        
        elif "بخوان و بیندیش" in line:
            section_type = "read_and_think"
            importance = "high"
            current_section.append(line)
        
        elif "واژه سازی" in line or "واژه‌سازی" in line:
            section_type = "word_formation"
            importance = "high"
            current_section.append(line)
        
        elif "بیاموز و بگو" in line:
            section_type = "learn_and_say"
            importance = "high"
            current_section.append(line)
        
        elif "بازی" in line:
            section_type = "game_activity"
            importance = "medium"
            current_section.append(line)
        
        elif "بخوان و حفظ کن" in line:
            section_type = "poem"
            importance = "high"
            current_section.append(line)
        
        
        # متن اصلی درس
        elif len(line) > 50 and not section_type.startswith("exercise"):
            if section_type in ["unknown", "lesson_title"]:
                section_type = "main_story"
                importance = "high"
            current_section.append(line)
        
        else:
            current_section.append(line)
    
    # افزودن آخرین بخش
    if current_section:
        chunks.append({
            "id": str(uuid4()),
            "lesson_id": lesson_id,
            "content": '\n'.join(current_section),
            "section_type": section_type,
            "importance": importance,
            "related_chunks": []
        })
    
    # افزودن روابط بین چانک‌ها
    chunks = build_relations(chunks)
    return chunks


# ==================== تابع ساخت روابط ====================

def build_relations(chunks: List[dict]) -> List[dict]:
    """افزودن روابط بین چانک‌ها بر اساس قواعد آموزشی"""
    for chunk in chunks:
        if chunk["section_type"] == "exercise_true_false":
            chunk["related_chunks"] = find_related(chunks, chunk["lesson_id"], "main_story")
        
        elif chunk["section_type"] in ["listen_and_speak", "find_and_say", "think_and_say"]:
            chunk["related_chunks"] = find_related(chunks, chunk["lesson_id"], "main_story")
        
        elif chunk["section_type"] == "thinking_station":
            chunk["related_chunks"] = find_related(chunks, chunk["lesson_id"], "read_and_think")
    
    return chunks


def find_related(chunks: List[dict], lesson_id: str, target_type: str) -> List[str]:
    """پیدا کردن شناسه‌ی چانک‌های مرتبط در همان درس"""
    return [c["id"] for c in chunks if c["lesson_id"] == lesson_id and c["section_type"] == target_type]


# ==================== Setup Weaviate ====================

def setup_weaviate_collection():
    """ساخت Collection با تنظیمات embedding"""
    print("🔄 در حال اتصال به Weaviate...")
    client = weaviate.connect_to_local(host="localhost", port=8080)
    print("✅ اتصال برقرار شد")

    # حذف کالکشن قدیمی
    try:
        client.collections.delete("Question")
        print("⚠️ Collection قبلی حذف شد")
    except:
        print("ℹ️ Collection قبلی وجود نداشت")

    # ساخت کالکشن جدید
    print("🔧 در حال ساخت Collection با مدل bge-m3...")

    client.collections.create(
        name="Question",
        vectorizer_config=Configure.Vectorizer.text2vec_ollama(
            api_endpoint="http://host.docker.internal:11434",
            model="bge-m3:latest"
        ),
        properties=[
            Property(name="content", data_type=DataType.TEXT, description="محتوای اصلی"),
            Property(name="section_type", data_type=DataType.TEXT, description="نوع بخش"),
            Property(name="importance", data_type=DataType.TEXT, description="سطح اهمیت"),
            Property(name="source", data_type=DataType.TEXT, description="منبع درس"),
            Property(name="lesson_id", data_type=DataType.TEXT, description="شناسه‌ی درس"),
            Property(name="chunk_id", data_type=DataType.TEXT, description="شناسه‌ی چانک"),
            Property(name="related_chunks", data_type=DataType.TEXT_ARRAY, description="شناسه‌ی چانک‌های مرتبط"),
        ]
    )

    client.close()
    print("✅ Collection با موفقیت ساخته شد")


# ==================== Import Lessons ====================

def import_lessons():
    """خواندن تمام فایل‌های درسی و وارد کردن به Weaviate"""
    lessons_dir = "./lessons"
    if not os.path.exists(lessons_dir):
        print("❌ پوشه lessons پیدا نشد!")
        return

    client = weaviate.connect_to_local(host="localhost", port=8080)
    questions = client.collections.get("Question")

    lesson_files = [f for f in os.listdir(lessons_dir) if f.endswith(".txt")]

    for lesson_file in lesson_files:
        lesson_path = os.path.join(lessons_dir, lesson_file)
        lesson_name = os.path.splitext(lesson_file)[0]

        print(f"\n📘 در حال پردازش {lesson_name} ...")
        with open(lesson_path, "r", encoding="utf-8") as f:
            content = f.read()

        chunks_data = chunk_by_semantic_sections(content, lesson_name=lesson_name)
        print(f"✂️ {len(chunks_data)} بخش شناسایی شد")

        with questions.batch.dynamic() as batch:
            for chunk in chunks_data:
                batch.add_object({
                    "content": chunk["content"],
                    "section_type": chunk["section_type"],
                    "importance": chunk["importance"],
                    "source": lesson_name,
                    "lesson_id": chunk["lesson_id"],
                    "chunk_id": chunk["id"],
                    "related_chunks": chunk["related_chunks"],
                })

        print(f"✅ {lesson_name}: {len(chunks_data)} بخش وارد شد")

    client.close()
    print("\n🎉 همه‌ی دروس با موفقیت وارد Weaviate شدند ✅")


# ==================== Main ====================

if __name__ == "__main__":
    print("=" * 60)
    print("🚀 Setup Weaviate - مرحله اولیه")
    print("=" * 60)

    setup_weaviate_collection()
    import_lessons()

    print("\n🎯 عملیات Setup کامل شد ✅")
