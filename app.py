import os
import io
import json
import fitz  # PyMuPDF
from PIL import Image
import streamlit as st
import pandas as pd
import easyocr
import torch
import numpy as np
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_community.chat_message_histories import FileChatMessageHistory
from langchain_groq import ChatGroq

st.set_page_config(page_title="Hybrid Search Multimodal RAG", layout="wide")

# --- Environment and Constant Setup ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY", st.secrets.get("GROQ_API_KEY", ""))
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

TEXT_MODEL = "llama3-70b-8192"
MAX_RETRIEVAL_CHUNKS = 10
TEMP_DIR = "temp_documents"
VECTOR_STORE_DIR = "faiss_vector_store"
CHAT_HISTORY_DIR = "chat_history"
IMAGE_DIR = "extracted_images"  # Directory to store extracted images

@st.cache_resource
def get_ocr_reader(langs):
    use_gpu = torch.cuda.is_available()
    return easyocr.Reader(langs, gpu=use_gpu)

def run_ocr(image_bytes: bytes, reader) -> str:
    try:
        result = reader.readtext(image_bytes, detail=0, paragraph=True)
        return " ".join(result)
    except Exception as e:
        return f"OCR failed: {str(e)}"

@st.cache_resource
def get_embeddings_model():
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

def get_adaptive_chunk_settings(num_pages):
    if num_pages < 10: return 1500, 200
    elif num_pages < 50: return 2500, 350
    return 4000, 500

def load_and_process_files(file_paths, ocr_reader):
    docs = []
    os.makedirs(IMAGE_DIR, exist_ok=True)
    
    for path in file_paths:
        filename = os.path.basename(path)
        try:
            if path.endswith(".pdf"):
                doc = fitz.open(path)
                chunk_size, overlap = get_adaptive_chunk_settings(len(doc))
                text_docs = PyPDFLoader(path).load()
                splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
                for d in text_docs:
                    for part in splitter.split_documents([d]):
                        part.metadata.update({'source': filename, 'page': d.metadata.get('page', 0)})
                        docs.append(part)
                
                for page_num in range(len(doc)):
                    page = doc.load_page(page_num)
                    for img_index, img_tuple in enumerate(page.get_images(full=True)):
                        xref = img_tuple[0] 
                        
                        base_image = doc.extract_image(xref) 
                        
                        img_bytes = base_image.get("image")
                        img_ext = base_image.get("ext")
                        
                        img_filename = f"{os.path.splitext(filename)[0]}_p{page_num + 1}_img{img_index}.{img_ext}"
                        img_path = os.path.join(IMAGE_DIR, img_filename)

                        if img_bytes:
                            with open(img_path, "wb") as img_file:
                                img_file.write(img_bytes)
                        else:
                            st.warning(f"Could not extract image data for image {img_index} from {filename} page {page_num+1}. Skipping.")
                            continue

                        img_ocr = run_ocr(img_bytes, ocr_reader)
                        caption = f"Image from {filename} on page {page_num + 1} with OCR text: {img_ocr if img_ocr else 'No OCR text extracted.'}"
                        
                        docs.append(Document(
                            page_content=caption, 
                            metadata={
                                'source': filename, 
                                'page': page_num + 1, 
                                'content_type': 'image',
                                'image_path': img_path
                            }
                        ))
            elif path.endswith(".txt"):
                docs.extend(TextLoader(path).load())
            elif path.endswith(".docx"):
                docs.extend(UnstructuredWordDocumentLoader(path).load())
        except Exception as e:
            st.error(f"Failed to process {filename}: {str(e)}")
    return docs

def get_vector_store(docs, embeddings):
    if not docs:
        st.warning("No documents to build vector store from.")
        return None

    if os.path.exists(VECTOR_STORE_DIR):
        try:
            return FAISS.load_local(VECTOR_STORE_DIR, embeddings, allow_dangerous_deserialization=True)
        except Exception as e:
            st.warning(f"Could not load existing vector store: {e}. Rebuilding...")
    
    vector_store = FAISS.from_documents(docs, embeddings)
    vector_store.save_local(VECTOR_STORE_DIR)
    return vector_store

def get_rag_chain(vector_store, docs):
    llm = ChatGroq(model_name=TEXT_MODEL, temperature=0.1)
    
    if vector_store is None:
        st.error("Vector store not initialized. Cannot create RAG chain.")
        return None

    semantic = vector_store.as_retriever(search_kwargs={"k": 5}) 
    keyword = BM25Retriever.from_documents(docs)
    keyword.k = 5 
    ensemble = EnsembleRetriever(retrievers=[semantic, keyword], weights=[0.7, 0.3])

    history_aware = create_history_aware_retriever(llm, ensemble, ChatPromptTemplate.from_messages([
        ("system", "Rewrite the user's question to be a standalone question based on the chat history."), 
        MessagesPlaceholder("chat_history"), 
        ("human", "{input}")
    ]))

    combine_chain = create_stuff_documents_chain(llm, ChatPromptTemplate.from_messages([
        ("system", """You are an expert document analysis assistant. Answer the user's question based on the context provided, which includes text, OCR data, and image captions. 
        When you use information from a source, cite it clearly, for example: [source_name, page X].
        If the context includes relevant images, explicitly refer to them in your answer (e.g., "As shown in the diagram on page 5..."). Use Markdown formatting for your response.
        Context: {context}"""),
        MessagesPlaceholder("chat_history"), 
        ("human", "{input}")
    ]))

    return create_retrieval_chain(history_aware, combine_chain)

# Moved get_chat_history definition before main to resolve NameError
def get_chat_history(session_id):
    os.makedirs(CHAT_HISTORY_DIR, exist_ok=True)
    return FileChatMessageHistory(os.path.join(CHAT_HISTORY_DIR, f"{session_id}.json"))

def main():
    st.title("Hybrid Multimodal RAG with Image Retrieval")

    langs = st.sidebar.multiselect("Select OCR Language(s)", ['en', 'hi', 'ta', 'fr', 'de'], default=['en'])
    ocr_reader = get_ocr_reader(langs)
    embeddings = get_embeddings_model()

    uploaded_files = st.sidebar.file_uploader("Upload PDFs/DOCX/TXT", type=["pdf", "docx", "txt"], accept_multiple_files=True)

    if st.sidebar.button("Process Documents") and uploaded_files:
        with st.spinner("Processing documents, performing OCR, and building index..."):
            paths = []
            for f in uploaded_files:
                os.makedirs(TEMP_DIR, exist_ok=True)
                path = os.path.join(TEMP_DIR, f.name)
                with open(path, "wb") as out: out.write(f.getbuffer())
                paths.append(path)
            
            docs = load_and_process_files(paths, ocr_reader)
            st.session_state.docs = docs 
            
            vector = get_vector_store(docs, embeddings)
            if vector:
                st.session_state.rag_chain = get_rag_chain(vector, docs)
                st.success("Documents processed and ready. You can now ask questions!")
            else:
                st.error("Failed to create vector store. Please check document processing.")

    if st.sidebar.button("Reset Session"):
        st.session_state.clear()
        import shutil
        for dir_path in [TEMP_DIR, VECTOR_STORE_DIR, IMAGE_DIR, CHAT_HISTORY_DIR]:
            if os.path.exists(dir_path):
                shutil.rmtree(dir_path)
        st.experimental_rerun()

    if 'docs' in st.session_state:
        total_docs = len(st.session_state.docs)
        image_docs_count = len([doc for doc in st.session_state.docs if doc.metadata.get('content_type') == 'image'])
        text_docs_count = total_docs - image_docs_count
        
        st.sidebar.markdown("---")
        st.sidebar.subheader("Document Statistics (for debugging)")
        st.sidebar.write(f"Total Documents Processed: {total_docs}")
        st.sidebar.write(f"  - Image Documents: {image_docs_count}")
        st.sidebar.write(f"  - Text Documents: {text_docs_count}")
        st.sidebar.markdown("---")

    if "rag_chain" in st.session_state and st.session_state.rag_chain is not None:
        session_id = "main_chat"
        history = get_chat_history(session_id) # This call is now correctly positioned
        for msg in history.messages:
            with st.chat_message(msg.type): st.markdown(msg.content)

        prompt = st.chat_input("Ask about your documents")
        if prompt:
            st.chat_message("human").markdown(prompt)
            with st.chat_message("ai"):
                with st.spinner("Thinking..."):
                    try:
                        response = st.session_state.rag_chain.invoke({"input": prompt, "chat_history": history.messages})
                        answer = response.get("answer", "I couldn't find an answer.")
                        st.markdown(answer)

                        retrieved_docs = response.get("context", [])
                        if st.sidebar.checkbox("Show retrieved document metadata"):
                            st.write("---")
                            st.subheader("Retrieved Document Metadata (for debugging)")
                            for i, doc in enumerate(retrieved_docs):
                                st.write(f"**Document {i+1}:**")
                                st.json(doc.metadata)
                            st.write("---")

                        image_paths = set() 
                        for doc in retrieved_docs:
                            if doc.metadata.get('content_type') == 'image' and 'image_path' in doc.metadata:
                                image_paths.add(doc.metadata['image_path'])
                        
                        if image_paths:
                            st.markdown("---")
                            st.markdown("**Retrieved Images:**")
                            for path in sorted(list(image_paths)):
                                if os.path.exists(path):
                                    st.image(path, caption=os.path.basename(path))
                                else:
                                    st.warning(f"Image not found at path: {path}")

                        history.add_user_message(prompt)
                        history.add_ai_message(answer)
                    except Exception as e:
                        st.error(f"An error occurred: {str(e)}")

        if st.button("⬇ Download Chat Transcript"):
            transcript = "\n\n".join([
                f"User: {m.content}" if m.type == "human" else f"AI: {m.content}" for m in history.messages
            ])
            st.download_button("Download Chat", transcript, file_name="chat_transcript.txt")

if __name__ == "__main__":
    main()

