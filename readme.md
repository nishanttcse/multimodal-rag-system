# Multimodal RAG System

A complete **Retrieval-Augmented Generation (RAG)** system that extracts:
- Text from PDFs  
- Tables from PDFs  
- OCR from images inside PDFs  

It embeds all data using **Sentence Transformers**, stores vectors in **FAISS**, and provides:
- A **FastAPI Backend**
- A **Gradio UI**
- A **Streamlit UI**

---

## 🚀 Features
- Multi-modal ingestion: text + tables + OCR
- Vector store powered by FAISS
- High-quality embeddings using MiniLM-L6-v2
- LangChain-compatible pipeline
- FastAPI endpoint for QA
- Gradio + Streamlit UI for easy usage

---

## 📁 Project Structure

multimodal-rag-system/
│── ingest.py # PDF → chunks → embeddings → FAISS
│── rag_chain.py # Retriever + LLM pipeline
│── app_fastapi.py # Backend API
│── ui_gradio.py # Gradio interface
│── ui_streamlit.py # Streamlit interface
│── requirements.txt # Python dependencies
│── technical_report.pdf # 2-page official report
│── video_script.txt # 3–5 minute demo script
│── data/
│ └── qatar.pdf # Input document
│── vector_store/ # FAISS storage

yaml
Copy code

---

## 📦 Installation

```bash
git clone https://github.com/nishanttcse/multimodal-rag-system
cd multimodal-rag-system
python -m venv venv
venv\Scripts\activate   # Windows
pip install -r requirements.txt
🏗️ Build Vector Store
bash
Copy code
python ingest.py
▶️ Run FastAPI backend
bash
Copy code
uvicorn app_fastapi:app --reload
🎨 Run Gradio UI
bash
Copy code
python ui_gradio.py
🌐 Run Streamlit UI
bash
Copy code
streamlit run ui_streamlit.py
📹 Demo Video
See video_script.txt for the exact recording flow.

📄 Report
See technical_report.pdf (included in this repo).

🤝 Contributing
Pull requests are welcome!

yaml
Copy code

---

# ✅ **5. Git Commands to Upload the Project**

Run these EXACT commands inside your `multimodal-rag-system` folder:

```bash
git init
git add .
git commit -m "Initial commit: Multimodal RAG System with ingestion, FAISS, FastAPI, Gradio, Streamlit, report, and scripts"
git branch -M main
git remote add origin https://github.com/nishanttcse/multimodal-rag-system.git
git push -u origin main
