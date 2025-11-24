from fastapi import FastAPI
from rag_chain import rag_answer

app = FastAPI()


@app.get("/ask")
def ask(q: str):
    answer = rag_answer(q)
    return {"answer": answer}
