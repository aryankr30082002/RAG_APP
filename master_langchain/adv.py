# app.py
import os, tempfile
import streamlit as st
from dotenv import load_dotenv

# LangChain + Cohere
from langchain_cohere import ChatCohere, CohereEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import Chroma

# PDF + OCR
from langchain_community.document_loaders import PyPDFLoader, UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import fitz  # PyMuPDF
from PIL import Image
import pytesseract

# ---------------- CONFIG ----------------
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
load_dotenv()

COHERE_API_KEY = os.getenv("COHERE_API_KEY")
if not COHERE_API_KEY:
    st.error("❌ Add COHERE_API_KEY to your .env or environment variables")
    st.stop()

LLM_MODEL, EMBED_MODEL = "command-a-03-2025", "embed-english-v3.0"
llm, embeddings = ChatCohere(model=LLM_MODEL, temperature=0.2), CohereEmbeddings(model=EMBED_MODEL)
CHROMA_DIR = "./chroma_pdf_store"

# ---------------- HELPERS ----------------
def load_pdf(path: str):
    """Load PDF → PyPDF → Unstructured → OCR fallback."""
    for Loader in (PyPDFLoader, UnstructuredPDFLoader):
        try:
            docs = Loader(path).load()
            if any(d.page_content.strip() for d in docs):
                return docs
        except:
            pass

    # OCR fallback
    pdf, docs = fitz.open(path), []
    for i, page in enumerate(pdf):
        pix = page.get_pixmap(dpi=200)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        text = pytesseract.image_to_string(img).strip()
        if text:
            docs.append(Document(page_content=text, metadata={"page": i}))
    return docs

def create_index(docs, chunk_size=800, chunk_overlap=100):
    """Split docs into chunks and build retriever."""
    if not isinstance(docs, list):
        docs = list(docs)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = [c for c in splitter.split_documents(docs) if c.page_content.strip()]
    if not chunks:
        return None, None

    db = Chroma.from_documents(chunks, embeddings, persist_directory=CHROMA_DIR)
    return db, db.as_retriever(search_kwargs={"k": 4})

def make_chain(retriever):
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        output_key="answer"
    )

# ---------------- STREAMLIT APP ----------------
st.set_page_config(page_title="PDF AI Assistant", layout="wide")
st.title("📘 General PDF AI Assistant")
st.write("Upload any PDF and ask questions. The assistant retrieves relevant chunks and answers based only on the document.")

if uploaded := st.file_uploader("📂 Upload your PDF", type=["pdf"]):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded.read())
        pdf_path = tmp.name

    with st.spinner("Extracting text..."):
        docs = load_pdf(pdf_path)
    if not docs:
        st.error("⚠️ Could not extract text.")
        st.stop()
    st.success(f"✅ Extracted {len(docs)} pages.")

    with st.spinner("Splitting and indexing..."):
        db, retriever = create_index(docs)
    if not db:
        st.error("⚠️ No chunks created.")
        st.stop()

    conv_chain = make_chain(retriever)

    st.markdown("---")
    query = st.text_input("💬 Ask questions about the PDF:")
    if query:
        with st.spinner("Thinking..."):
            try:
                result = conv_chain({"question": query})
            except:
                result = conv_chain({"query": query})

        st.success("✅ Answer:")
        st.write(result["answer"])

        if sources := result.get("source_documents", []):
            st.markdown("**📖 Sources:**")
            for i, src in enumerate(sources, 1):
                snippet = src.page_content[:500].strip().replace("\n", " ")
                st.markdown(f"**Source {i} (p.{src.metadata.get('page','?')}):** {snippet}...")
