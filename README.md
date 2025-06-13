# Octopus_Books
An App for Book Club Lovers

**Ask intelligent questions about your uploaded ebooks.**  
Ebook Tutor is a Streamlit app powered by LangChain, OpenAI GPT-4o, and ChromaDB that allows you to chat with any PDF book using Retrieval-Augmented Generation (RAG). Upload a PDF and instantly start asking deep, contextual questions based on the content.

---

## 🚀 Features

- 📤 Upload and analyze any PDF file  
- 🧠 Chunk, embed, and persist vector representations using `ChromaDB`  
- 🤖 Chat interface powered by OpenAI’s GPT-4o-mini  
- 📝 Maintains session memory and chat history  
- 💾 Vector store persistence to avoid reprocessing  
- 🌙 Beautiful dark-mode inspired UI with custom CSS styling  

---

## 🖥️ Demo

> *(Add GIF or screenshot here if applicable)*

---

## 📦 Installation

1. **Clone the repository**

```bash
git clone https://github.com/yourusername/ebook-tutor.git
cd ebook-tutor
```

2. **Create a virtual environment**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Set up OpenAI API key**

Create a `.streamlit/secrets.toml` file:

```toml
OPENAI_API_KEY = "your-openai-api-key"
```

Alternatively, use environment variables or a `credentials.yml` setup.

---

## 🧠 How It Works

1. The user uploads a `.pdf` file.
2. It’s split into 1000-character chunks using LangChain’s `CharacterTextSplitter`.
3. Text is embedded using `text-embedding-3-large` via OpenAI.
4. The embeddings are stored and persisted using `ChromaDB`.
5. User queries are passed through a RAG chain using a structured prompt.
6. The assistant returns answers grounded in your PDF.

---

## 📁 Project Structure

```
📦 ebook-tutor
├── ebook_tutor.py             # Main Streamlit app
├── requirements.txt           # Python dependencies
├── Practice_Folder_V1/
│   └── data/                  # Stores Chroma vectorstore folders
├── .streamlit/
│   └── secrets.toml           # OpenAI API key (not committed)
└── README.md
```

---

## ✅ Requirements

- Python 3.9+
- Streamlit
- LangChain
- OpenAI
- ChromaDB
- pysqlite3 (used for custom SQLite bindings)

---

## 🧪 Example Prompt

> **User:** "What marketing strategies were mentioned in Chapter 2?"  
> **AI:** "Chapter 2 highlights the importance of email segmentation, targeted promotions, and storytelling for brand identity..."

---

## 🛠️ Development Notes

- Make sure your system SQLite version is ≥ 3.35.0.  
- If using in cloud environments (like Streamlit Community Cloud), ensure `pysqlite3` is properly patched and `sys.modules["sqlite3"] = pysqlite3` is declared before imports.

---

## 📬 Future Improvements

- Sidebar with search/filter capabilities  
- Source document citation viewer  
- Memory persistence between sessions  
- Multi-PDF support  
- Flashcard or highlight export mode  

---

## 👤 Author

Built with ❤️ by Lannon Khau(https://github.com/LannontheCannon)
