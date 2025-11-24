import os
import cv2
import numpy as np
import pytesseract
from tqdm import tqdm
from pdf2image import convert_from_path
import pdfplumber

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from sentence_transformers import SentenceTransformer
from langchain.embeddings.base import Embeddings


# ---------------------------------------
# PATH CONFIG
# ---------------------------------------
DATA_PATH = "data/qatar.pdf"
VECTOR_PATH = "vector_store"

# Poppler and Tesseract Paths
POPLER_PATH = r"F:\Release-25.11.0-0\poppler-25.11.0\Library\bin"
pytesseract.pytesseract.tesseract_cmd = r"F:\Tesseract-OCR\tesseract.exe"

os.makedirs(VECTOR_PATH, exist_ok=True)


# ---------------------------------------
# EMBEDDING MODEL
# ---------------------------------------
embedder = SentenceTransformer("all-MiniLM-L6-v2")


# Wrapper to use SentenceTransformer inside LangChain
class STEmbeddingWrapper(Embeddings):
    def embed_documents(self, texts):
        return embedder.encode(texts).tolist()

    def embed_query(self, query):
        return embedder.encode([query]).tolist()[0]


embed_model = STEmbeddingWrapper()


# ---------------------------------------
# TEXT EXTRACTION FROM PDF
# ---------------------------------------
def extract_text_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()       # list of Document objects
    return pages


# ---------------------------------------
# TABLE EXTRACTION
# ---------------------------------------
def extract_tables_pdf(pdf_path):
    tables_text = []

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables()

            for table in tables or []:
                cleaned_rows = []
                for row in table:
                    cleaned_row = [cell if cell is not None else "" for cell in row]
                    cleaned_rows.append(" | ".join(cleaned_row))

                table_text = "\n".join(cleaned_rows)
                tables_text.append(table_text)

    return tables_text


# ---------------------------------------
# OCR EXTRACTION
# ---------------------------------------
def extract_images_ocr(pdf_path):
    images_text = []

    print("\nConverting PDF → Images (via Poppler)...")
    pages = convert_from_path(pdf_path, 300, poppler_path=POPLER_PATH)

    print(f"Total pages detected: {len(pages)}")
    print("Running OCR on each page...\n")

    for img in tqdm(pages, desc="OCR Progress", unit="page"):
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        text = pytesseract.image_to_string(img_cv)
        images_text.append(text)

    print("\nOCR Completed Successfully!\n")
    return images_text


# ---------------------------------------
# BUILD VECTOR STORE
# ---------------------------------------
def build_vector_store():
    print("Extracting text...")
    text_pages = extract_text_pdf(DATA_PATH)

    print("Extracting tables...")
    tables = extract_tables_pdf(DATA_PATH)
    table_docs = [Document(page_content=t) for t in tables]

    print("Extracting OCR text...")
    ocr_texts = extract_images_ocr(DATA_PATH)
    ocr_docs = [Document(page_content=t) for t in ocr_texts]

    # Merge all documents
    all_docs = text_pages + table_docs + ocr_docs

    # Chunking
    print("Chunking documents...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(all_docs)

    docs_text = [chunk.page_content for chunk in chunks]

    # Build FAISS
    print("Embedding & Saving FAISS store...")
    db = FAISS.from_texts(docs_text, embed_model)
    db.save_local(VECTOR_PATH)

    print("\n🎉 Vector store build completed successfully!")


if __name__ == "__main__":
    build_vector_store()
