import gradio as gr
from rag_chain import rag_answer

def chat(query):
    return rag_answer(query)

iface = gr.Interface(
    fn=chat,
    inputs="text",
    outputs="text",
    title="📄 Multi-Modal RAG QA Bot",
    description="Ask questions about the IMF Qatar Report"
)

iface.launch()
