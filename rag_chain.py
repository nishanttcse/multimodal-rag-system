from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
# Use simple Python string formatting instead of langchain's PromptTemplate
from langchain_community.llms import HuggingFaceHub  # or OpenAI / Gemini

VECTOR_PATH = "vector_store"

embedder = SentenceTransformer("all-MiniLM-L6-v2")
db = FAISS.load_local(VECTOR_PATH, embedder)

prompt_template = """
You are a QA assistant. Answer ONLY using the context below.
ALWAYS include page numbers if possible.

Context:
{context}

Question:
{question}

Answer:
"""


def rag_answer(query):
    docs = db.similarity_search(query, k=4)
    context = "\n\n".join(docs)

    # build the prompt using plain string formatting
    prompt = prompt_template.format(context=context, question=query)

    llm = HuggingFaceHub(repo_id="google/flan-t5-large")

    return llm(prompt)
