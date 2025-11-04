from typing import List
from pydantic import BaseModel
from langchain_core.documents import Document

class RAGState(BaseModel):
    """State object for RAG workflow"""
    
    question: str
    retrieved_documents: List[Document] = []
    answer: str = ""