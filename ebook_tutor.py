from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import streamlit as st 
import yaml
import os

def format_chat_history(messages):
    history = ""
    for msg in messages:
        role = "User" if msg["role"] == "human" else "AI"
        history += f"{role}: {msg['content']}\n"
    return history

# === CONFIG
os.environ["OPENAI_API_KEY"] = yaml.safe_load(open("../credentials.yml"))['openai']

# === LOAD PDF & SPLIT
import os
import tempfile



# === RAG PROMPT + MODEL
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
model = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

rag_chain = (
    prompt
    | model
    | StrOutputParser()
)

# === STREAMLIT APP SETUP
load_pdf = st.file_uploader("Please upload your pdf here:", type="pdf")

if load_pdf:
    # Save uploaded file to a temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(load_pdf.read())
        tmp_path = tmp_file.name

    # Use the filename (without extension) as vectorstore folder name
    pdf_name = os.path.splitext(load_pdf.name)[0]
    persist_path = f"../Practice_Folder_V1/data/{pdf_name}"

    # Load and process PDF
    loader = PyPDFLoader(tmp_path)
    documents = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=1000)
    chunks = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

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

    # Optional cleanup
    os.remove(tmp_path)


# === SESSION STATE MEMORY
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "ai", "content": "Hey there! What would you like to know about the book?"}]

# === DISPLAY CHAT HISTORY
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# === USER INPUT
if question := st.chat_input("Ask something about the book..."):
    

    with st.chat_message("human"):
        st.markdown(question)
    
    with st.spinner("Thinking..."):
        chat_history = format_chat_history(st.session_state.messages)
        docs = retriever.invoke(question)  # ← do this manually

        inputs = {
            "question": question,
            "context": docs,
            "chat_history": chat_history
        }

        response = rag_chain.invoke(inputs)

    with st.chat_message("ai"):
        st.markdown(response)

    # === SAVE TO MEMORY
    st.session_state.messages.append({"role": "human", "content": question})
    st.session_state.messages.append({"role": "ai", "content": response})