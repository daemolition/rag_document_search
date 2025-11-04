# RAG Document Search

---

A simple document search system using RAG and ReAct Agent to retrieve documents.
Documents can be either from a directory, *.pdf, *.txt files or urls.

The documents will be saved in a vectorstore (FAISS) and be retrieved with a simple similiratiy search.
The Project is uses a local mistral model with a langchain wrapper.

## Agentic Workflow

The project showcases the use of modern agentic ai patterns using langgraph to define a workflow.

## Simple UI

The UI is simple using streamlit.
The document retrieve gives the answer and the documents used.

## How to use

1. Clone the repository
2. `uv sync`
3. `streamlit run streamlit_app.py`
