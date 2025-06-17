import os
import streamlit as st
from pathlib import Path
import tempfile
from typing import List

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# ─── Helpers ────────────────────────────────────────────────────────────────────

def load_uploaded_documents(files) -> List[Document]:
    docs: List[Document] = []
    for file in files:
        suffix = Path(file.name).suffix.lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(file.read())
            tmp_path = tmp.name

        if suffix == ".pdf":
            loader = PyPDFLoader(tmp_path)
        elif suffix == ".docx":
            loader = Docx2txtLoader(tmp_path)
        elif suffix == ".txt":
            loader = TextLoader(tmp_path)
        else:
            st.warning(f"Unsupported file type: {file.name}")
            continue

        docs.extend(loader.load())
    return docs

def split_documents(docs: List[Document], chunk_size=1000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(docs)

def build_retriever(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="rag_collection",
        persist_directory="./chroma_db"
    )
    return vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 5, "fetch_k": 10})

def get_llm():
    return ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)

def get_answer_prompt():
    return ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Answer ONLY using the context. "
                   "If it’s not in the context, say 'Answer not available in the context'."),
        ("system", "Context:\n{context}"),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

def get_rewrite_prompt():
    return ChatPromptTemplate.from_messages([
        ("system", "Rewrite the latest user query as a standalone question given chat history."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

def build_rag_chain(llm, retriever):
    rewrite_prompt = get_rewrite_prompt()
    history_aware = create_history_aware_retriever(llm, retriever, rewrite_prompt)
    qa_prompt = get_answer_prompt()
    doc_chain = create_stuff_documents_chain(llm, qa_prompt)
    return create_retrieval_chain(history_aware, doc_chain)

def handle_chat(rag_chain):
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_input = st.chat_input("Ask something about your uploaded files…")
    if user_input:
        with st.spinner("Thinking…"):
            result = rag_chain.invoke({
                "input": user_input,
                "chat_history": st.session_state.chat_history
            })
            answer = result["answer"]

        st.session_state.chat_history.append(HumanMessage(content=user_input))
        st.session_state.chat_history.append(AIMessage(content=answer))

    for msg in st.session_state.chat_history:
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        with st.chat_message(role):
            st.write(msg.content)

# ─── Main App Flow ────────────────────────────────────────────────────────────────
def main():
    load_dotenv()
    os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY", "")

    st.set_page_config(page_title="RAG Chatbot", layout="wide")
    st.title("📄🧠 RAG Chatbot with Multiple File Upload & Submit")
    uploaded_files = st.file_uploader(
        "Upload your documents (PDF, DOCX, TXT)", 
        type=["pdf", "docx", "txt"], 
        accept_multiple_files=True
    )

    # Add a button to trigger processing
    if uploaded_files:
        if st.button("📚 Process Documents"):
            with st.spinner("Indexing documents…"):
                docs = load_uploaded_documents(uploaded_files)
                chunks = split_documents(docs)
                retriever = build_retriever(chunks)
                llm = get_llm()
                st.session_state.rag_chain = build_rag_chain(llm, retriever)
            st.success("✅ Documents processed! You can now chat below.")

    # Once processed, show chat interface
    if "rag_chain" in st.session_state:
        handle_chat(st.session_state.rag_chain)
    elif not uploaded_files:
        st.info("Upload files and click ‘Process Documents’ to start.")
if __name__=="__main__":
    main()
