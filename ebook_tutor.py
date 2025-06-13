import sys
import pysqlite3
sys.modules["sqlite3"] = pysqlite3

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import streamlit as st
import os
import tempfile

# === CONFIG ===
st.set_page_config(page_title="📘 Ebook Tutor", page_icon="📖", layout="wide")
st.markdown("""
    <style>
    body {
        background: linear-gradient(to bottom right, #0f172a, #1e293b, #0f172a);
        color: #e2e8f0;
        font-family: 'Inter', sans-serif;
    }
    .stButton > button {
        background-color: #4f46e5;
        color: white;
        border-radius: 0.5rem;
        padding: 0.5rem 1.25rem;
        transition: all 0.2s ease-in-out;
    }
    .stButton > button:hover {
        background-color: #4338ca;
        transform: scale(1.03);
    }
    .uploadedFileName {
        font-size: 0.9rem;
        color: #94a3b8;
        margin-top: 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)

st.title("📘 DataForge Ebook Tutor")
st.caption("Ask intelligent questions from your uploaded ebook PDF, powered by GPT-4o + LangChain.")
st.divider()

# === API KEY ===
openai_key = st.secrets["OPENAI_API_KEY"]

# === FORMAT CHAT HISTORY ===
def format_chat_history(messages):
    history = ""
    for msg in messages:
        role = "User" if msg["role"] == "human" else "AI"
        history += f"{role}: {msg['content']}\n"
    return history

# === FILE UPLOADER ===
load_pdf = st.file_uploader("📤 Upload your PDF here", type="pdf")

if load_pdf:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(load_pdf.read())
        tmp_path = tmp_file.name

    pdf_name = os.path.splitext(load_pdf.name)[0]
    persist_path = f"../Practice_Folder_V1/data/{pdf_name}"

    loader = PyPDFLoader(tmp_path)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000)
    chunks = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-large", openai_api_key=openai_key)

    if not os.path.exists(persist_path):
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=persist_path
        )
        vectorstore.persist()
    else:
        vectorstore = Chroma(
            embedding_function=embeddings,
            persist_directory=persist_path
        )

    retriever = vectorstore.as_retriever()

    os.remove(tmp_path)

    st.success(f"✅ '{load_pdf.name}' processed successfully!")

    # === RAG PROMPT ===
    template = """You are a marketing assistant helping a business owner.
Use only the context below and the chat history so far to answer the next question.

Context:
{context}

Chat History:
{chat_history}

Current Question:
{question}
"""
    prompt = ChatPromptTemplate.from_template(template)
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0.3, openai_api_key=openai_key)
    rag_chain = prompt | model | StrOutputParser()

    # === SESSION STATE ===
    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "ai", "content": "Hey there! What would you like to know about the book?"}]

    # === CHAT HISTORY DISPLAY ===
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # === CHAT INPUT ===
    if question := st.chat_input("Ask something about the book..."):
        with st.chat_message("human"):
            st.markdown(question)

        with st.spinner("Thinking..."):
            chat_history = format_chat_history(st.session_state.messages)
            docs = retriever.invoke(question)

            response = rag_chain.invoke({
                "question": question,
                "context": docs,
                "chat_history": chat_history
            })

        with st.chat_message("ai"):
            st.markdown(response)

        # Update memory
        st.session_state.messages.append({"role": "human", "content": question})
        st.session_state.messages.append({"role": "ai", "content": response})

else:
    st.info("👆 Start by uploading a PDF to begin.")