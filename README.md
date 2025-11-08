# 📚 Farsi Elementary Assistant

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![LangGraph](https://img.shields.io/badge/LangGraph-Powered-green.svg)](https://github.com/langchain-ai/langgraph)

> An intelligent educational assistant for Iranian elementary school students, focusing on Farsi language learning through AI-powered semantic search and contextual understanding.

[**فارسی**](./README.fa.md) | **English**

---

## 📖 Overview

**Farsi Elementary Assistant** is an intelligent educational system designed to help Iranian elementary school students (specifically 2nd grade) with their Farsi language lessons. Using RAG (Retrieval-Augmented Generation), intelligent semantic search, and LangGraph agent architecture, it provides accurate, age-appropriate responses.

### 🎯 Target Audience
- **Primary**: Parents and educators who want to assist children with homework
- **Secondary**: Elementary students (with parental supervision)

---

## ✨ Key Features

### 🔍 Multi-Layered Intelligent Search
- **3-Stage Search Strategy**:
  1. **Exact Match**: Direct lookup when lesson + section are specified
  2. **Metadata Filtering**: Filter by lesson or section type
  3. **Semantic Search**: Vector-based similarity search using BGE-M3
- **Smart Query Understanding**: Automatically detects lesson numbers (Persian/English digits) and section types
- **Related Content Discovery**: Finds contextually related chunks (e.g., exercises linked to main story)

### 🤖 LangGraph Agent Architecture
- **Intelligent Routing**: Decides whether to search or use conversation history
- **Tool Integration**: Extensible design - add custom tools easily
- **Persistent Memory**: SQLite-based conversation history with smart context management
- **Ephemeral Contexts**: Temporary RAG contexts that don't pollute conversation history

### 🧠 Advanced RAG Pipeline
- **Weaviate Vector Database**: Efficient semantic search with BGE-M3 embeddings
- **Semantic Chunking**: Lessons are intelligently split by sections (poems, exercises, stories)
- **Context Filtering**: Removes old conversations and RAG contexts to manage token limits
- **Cross-Reference System**: Related chunks are automatically linked (e.g., exercises → main text)

### 💬 User-Friendly Interface
- **Gradio Chat UI**: Simple, intuitive chat interface
- **Example Queries**: Pre-built examples to get started quickly
- **Conversational Flow**: Natural dialogue suitable for educational contexts

---

## 🏗️ Architecture

<img width="308" height="372" alt="image" src="https://github.com/user-attachments/assets/017c2f18-5a1d-4971-ba87-42a032231602" />


**Key Components**:
- **Router**: Analyzes query to determine if vector search is needed
- **Semantic Search Tool**: Queries Weaviate with intelligent filtering
- **LLM Agent**: GPT-4o-mini (or compatible) with tool-calling capabilities
- **Memory**: SQLite checkpointer for conversation persistence

---

## 📋 Prerequisites

### Required
- **Python**: 3.10 or higher
- **Docker**: For running Weaviate locally
- **Ollama**: For local embeddings (BGE-M3 model)
- **LLM API Key**: OpenAI-compatible API (OpenAI, OpenRouter, Groq, etc.)

### Optional
- **Weaviate Cloud**: Alternative to local Docker setup
- **OpenAI Embeddings**: Alternative to local Ollama embeddings

---

## 🚀 Quick Start

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/yourusername/farsi-elementary-assistant.git
cd farsi-elementary-assistant
```

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

**Required packages**:
- `langgraph>=0.2.0`
- `langchain-openai>=0.2.0`
- `weaviate-client>=4.0.0`
- `gradio>=4.0.0`
- `python-dotenv`
- `requests`

### 3️⃣ Setup Environment Variables
Create a `.env` file in the root directory:

```bash
# LLM Configuration (Choose one or multiple)
METIS_API_KEY=your_api_key_here
METIS_BASE_URL=https://api.openai.com/v1

# Alternative: OpenAI Direct
# OPENAI_API_KEY=sk-...
# OPENAI_API_BASE=https://api.openai.com/v1

# Alternative: OpenRouter
# OPENROUTER_API_KEY=sk-or-v1-...
# OPENROUTER_BASE_URL=https://openrouter.ai/api/v1

# Alternative: Local Ollama
# OLLAMA_BASE_URL=http://localhost:11434/
# OLLAMA_API_KEY=ollama

# Ollama for Embeddings (if using local embeddings)
OLLAMA_BASE_URL=http://localhost:11434/
OLLAMA_API_KEY=ollama
```

> 💡 **Note**: You can use any OpenAI-compatible API provider. The code uses `METIS_*` as default, but you can modify `main_agent.py` to use your preferred provider.

### 4️⃣ Setup Weaviate

#### Option A: Local Docker (Recommended for Development)
```bash
# Start Weaviate container
docker-compose up -d

# Verify it's running
curl http://localhost:8080/v1/meta
```

#### Option B: Weaviate Cloud
See [Weaviate Cloud Documentation](https://weaviate.io/developers/weaviate/installation/weaviate-cloud-services) for setup instructions. Update connection details in `setup_weaviate.py` and `main_agent.py`.

### 5️⃣ Setup Ollama (For Local Embeddings)

```bash
# Install Ollama (if not already installed)
# Visit: https://ollama.ai/download

# Pull BGE-M3 model
ollama pull bge-m3:latest

# Verify
ollama list
```

> 💡 **Alternative**: Use OpenAI embeddings by modifying the vectorizer config in `setup_weaviate.py` to use `text2vec-openai` instead of `text2vec-ollama`.

### 6️⃣ Import Lesson Data
```bash
# This will:
# - Create the Weaviate collection
# - Chunk and import lesson_01.txt and lesson_02.txt
# - Build semantic relationships between chunks

python setup_weaviate.py
```

**Expected Output**:
```
🔄 در حال اتصال به Weaviate...
✅ اتصال برقرار شد
🔧 در حال ساخت Collection با مدل bge-m3...
✅ Collection با موفقیت ساخته شد

📘 در حال پردازش lesson_01 ...
✂️ 12 بخش شناسایی شد
✅ lesson_01: 12 بخش وارد شد

📘 در حال پردازش lesson_02 ...
✂️ 15 بخش شناسایی شد
✅ lesson_02: 15 بخش وارد شد

🎉 همه‌ی دروس با موفقیت وارد Weaviate شدند ✅
```

### 7️⃣ Run the Agent
```bash
python main_agent.py
```

The Gradio interface will launch at `http://localhost:7860`

---

## ⚙️ Configuration

### LLM Provider Options

The system supports any OpenAI-compatible API. Here are common configurations:

#### OpenAI Direct
```python
# In main_agent.py, modify:
llm = ChatOpenAI(
    base_url="https://api.openai.com/v1",
    api_key=os.getenv("OPENAI_API_KEY"),
    model="gpt-4o-mini",
    temperature=0.3,
)
```

#### OpenRouter (Access Multiple Models)
```python
llm = ChatOpenAI(
    base_url=os.getenv("OPENROUTER_BASE_URL"),  # https://openrouter.ai/api/v1
    api_key=os.getenv("OPENROUTER_API_KEY"),
    model="anthropic/claude-3-haiku",  # or any other model
    temperature=0.3,
)
```

#### Local Ollama
```python
llm = ChatOpenAI(
    base_url=os.getenv("OLLAMA_BASE_URL"),  # http://localhost:11434/v1
    api_key="ollama",  # Ollama doesn't require real API key
    model="llama3.2",  # or any model you have pulled
    temperature=0.3,
)
```

> 📚 **Need API Keys?** We'll provide a guide soon on obtaining API keys from various providers. For now, check:
> - [OpenAI Platform](https://platform.openai.com/)
> - [OpenRouter](https://openrouter.ai/)
> - [Groq](https://console.groq.com/)

### Embedding Options

#### Local Ollama (Default)
```python
# In setup_weaviate.py
vectorizer_config=Configure.Vectorizer.text2vec_ollama(
    api_endpoint="http://host.docker.internal:11434",
    model="bge-m3:latest"
)
```

#### OpenAI Embeddings
```python
vectorizer_config=Configure.Vectorizer.text2vec_openai(
    model="text-embedding-3-small"
)
# Don't forget to set OPENAI_API_KEY in environment
```

---

## 📚 Adding More Lessons

Lessons are stored as plain text files in the `lessons/` directory.

### Step 1: Create Lesson File
Create a new file: `lessons/lesson_03.txt` (or any number)

### Step 2: Format Your Lesson
Follow this structure:

```
فصل [Chapter Name]

درس [Number]: [Lesson Title]
[Main story content...]

[Section Headers:]
- بیاموز و بگو (Learn and Say)
- واژه سازی (Word Formation)
- بخوان و حفظ کن (Poem)
- درست، نادرست (True/False Exercise)
- بازی، بازی، بازی (Games)
- گوش کن و بگو (Listen and Speak)
- فکر کن و بگو (Think and Say)
- پیدا کن و بگو (Find and Say)
- ایستگاه اندیشه (Thinking Station)
- بخوان و بیندیش (Read and Think)
```

### Step 3: Re-run Setup
```bash
python setup_weaviate.py
```

The script will automatically detect and import new lessons!

---

## 🎯 Usage Examples

### Example Queries

**Question about a specific lesson:**
```
درس اول درباره چی بود؟
(What was lesson 1 about?)
```

**Request for poem:**
```
شعر درس اول رو برام بخون
(Read me the poem from lesson 1)
```

**Exercise help:**
```
تمرین درست و نادرست درس دوم
(True/False exercise from lesson 2)
```

**Dictation request:**
```
یک املا از درس سوم برام بساز
(Create a dictation from lesson 3)
```

---

## 🖼️ Screenshots

> 📸 **TODO**: Add screenshots here

### Recommended Screenshots:
1. **Main Chat Interface** (`screenshots/chat-interface.png`)
   - Show the Gradio UI with a sample conversation
   
2. **Search Results Example** (`screenshots/search-results.png`)
   - Terminal output showing the 3-stage search process
   
3. **Weaviate Dashboard** (`screenshots/weaviate-dashboard.png`)
   - Optional: Show the Weaviate UI with imported data

4. **Agent Graph Visualization** (`screenshots/agent-graph.png`)
   - The LangGraph structure (automatically generated in code)

**How to add**:
```markdown
![Chat Interface](screenshots/chat-interface.png)
*The main Gradio chat interface with example queries*
```

---

## 🔧 Customization & Extensibility

### Adding Custom Tools

The system uses LangGraph's tool-calling architecture. You can easily add new tools:

```python
from langchain_core.tools import StructuredTool

def my_custom_tool(query: str) -> str:
    """Your custom functionality"""
    # Implementation
    return result

tool_custom = StructuredTool.from_function(
    name="custom_tool",
    func=my_custom_tool,
    description="Description for the LLM to understand when to use this tool"
)

# Add to tools list
tools = [tool_weaviate, tool_custom]
```

### Adjusting Search Behavior

Modify `intelligent_search()` in `main_agent.py`:
- **Change result limit**: `limit=3` → `limit=5`
- **Add new section patterns**: Extend `section_patterns` dictionary
- **Adjust routing logic**: Modify `route_after_start()` scoring system

### Conversation History Management

In `filter_messages_for_llm()`:
- **Change history length**: `max_pairs=5` → keep last 5 conversation pairs
- **Modify filtering rules**: Adjust which messages to keep/remove

---

## 🗺️ Roadmap

### Current Version (v0.1)
- ✅ Semantic search with Weaviate + BGE-M3
- ✅ LangGraph agent with tool calling
- ✅ Gradio chat interface
- ✅ 2 sample lessons (Farsi 2nd grade)

### Upcoming Features
- 🔜 **More Lessons**: Complete 2nd grade Farsi curriculum
- 🔜 **Enhanced UI**: Custom web interface beyond Gradio
- 🔜 **Multi-Subject Support**: Math, Science, etc.
- 🔜 **Voice Interaction**: Text-to-speech for poems and stories
- 🔜 **Progress Tracking**: Student performance analytics
- 🔜 **Mobile App**: iOS/Android applications

### Future Considerations
- 📱 Telegram bot integration (in development)
- 🎮 Gamification elements
- 👥 Multi-user support with authentication
- 🌐 Web-based deployment (Weaviate Cloud + OpenAI embeddings)

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

### Ways to Contribute
1. **Add More Lessons**: Format and contribute new lesson files
2. **Improve Chunking Logic**: Enhance semantic section detection
3. **Add Features**: New tools, UI improvements, etc.
4. **Fix Bugs**: Report or fix issues
5. **Documentation**: Improve README, add tutorials

### Contribution Process
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Guidelines
- Follow existing code style
- Add comments for complex logic (Farsi or English)
- Test your changes locally
- Update documentation if needed

---

## 🙏 Acknowledgments

- **LangChain/LangGraph**: For the agent framework
- **Weaviate**: For vector database capabilities
- **BGE-M3**: For multilingual embeddings
- **Gradio**: For rapid UI prototyping

---

## 📧 Contact & Support

- **Issues**: [GitHub Issues](https://github.com/Hosseinkhaleghinia/farsi-elementary-assistant/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Hosseinkhaleghinia/farsi-elementary-assistant/discussions)

---

**Made with ❤️ for Iranian students and educators**
