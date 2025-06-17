# RAG Chatbot with PDFs, DOCX & TXT

This project is a **Retrieval-Augmented Generation (RAG)** chatbot built with:

- **Streamlit** for the interactive web interface
- **LangChain** framework for RAG pipeline
- **Google Generative AI (Gemini)** for embeddings and LLM
- **Chroma** as the local vector database
- **PyPDFLoader**, **Docx2txtLoader**, **TextLoader** for document ingestion

---

## 📂 Project Structure

```
gemini_pdf_chatbot/
├── app.py                  # Streamlit application entry point
├── requirements.txt        # Project dependencies
├── .env                    # Environment variables (e.g., API key)
├── chroma_db/              # Persisted Chroma vector store
└── README.md               # This file
```

---

## ⚙️ Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/sumangithazra/Chat_with_multiple_file.git 
   ```

2. **Create a virtual environment**

   ```bash
   python -m venv venv
   # Windows:
   venv\Scripts\activate
   # macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**

   - Create a `.env` file in the project root:
     ```env
     GOOGLE_API_KEY=AIza...your_api_key_here
     ```
   - This key must have access to Google's Generative Language API (Gemini).

---

## 🚀 Running the App

Start the Streamlit app from your project directory:

```bash
streamlit run app.py
```

Then open your browser to `http://localhost:8501`.

---

## 🛠️ Usage

1. **Upload documents**:
   - Click the **Browse files** button and select one or more `.pdf`, `.docx`, or `.txt` files.
2. **Process documents**:
   - Click **Process Documents** to ingest and index the uploaded files.
3. **Chat**:
   - Enter your question in the chat box. The model will answer based on the uploaded content.
   - The conversation history appears below and is used for context in follow-up questions.

---

## 🔍 How It Works

1. **Document Ingestion**: Uploaded files are saved temporarily and loaded via the appropriate loader.
2. **Chunking**: Text is split into overlapping chunks (\~1000 characters) for embedding.
3. **Embedding & Indexing**: Chunks are embedded with `GoogleGenerativeAIEmbeddings` and stored in ChromaDB.
4. **RAG Chain**:
   - **History-aware retriever** rewrites user queries based on conversation history.
   - **Chroma retriever** fetches the most relevant chunks with Maximal Marginal Relevance (MMR).
   - **LLM prompt** injects retrieved context and produces an answer.
5. **Chat Interface**: Streamlit displays user and assistant messages turn by turn.

---

## 📦 Dependencies

- `streamlit`
- `python-dotenv`
- `langchain`, `langchain-core`, `langchain-community`
- `langchain-google-genai`
- `langchain-chroma`, `chromadb`
- `pypdf`, `docx2txt`

See `requirements.txt` 

---

## 💡 Extending

- Add support for more file types (e.g., `.md`, `.pptx`).
- Deploy on cloud platforms (e.g., Streamlit Cloud, Hugging Face Spaces).
- Integrate user authentication and file upload limits.
- Replace Google Gemini with other LLMs or hybrid retrieval.

---

## 🔗 References

- [LangChain Documentation](https://langchain.com)
- [Google Generative AI (Gemini) API](https://cloud.google.com/generative-ai)
- [ChromaDB](https://www.trychroma.com)

---


