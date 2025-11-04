from typing import List

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

class VectoreStore:
    """Manages vectore store application"""
    
    def __init__(self, embedding_model:str = 'all-MiniLM-L6-v2'):
        self.embedding=HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device':'cuda'}
        )
        self.vectorestore=None
        self.retriever=None
        
    
    def create_vectorstore(self, documents: List[Document]):
        """
        Create vectorestore from documents
        
        Args:
            documents: Lost of documents to embed
        """
        self.vectorestore = FAISS.from_documents(documents, self.embedding)
        self.retriever = self.vectorestore.as_retriever()


    def get_retriever(self):
        """
        Get the retriever instance
        
        Returns: 
            Retriever instance
        """       
        
        if self.retriever is None:
            raise ValueError("Vectorstore no initialized. Call create_vectorstore first")
        return self.retriever
    
    
    def retrieve(self, query: str, k: int = 5) -> List[Document]:
        """
        Retrieve relevant documents for a query
        
        Args:
            query: Search query
            k: Number of documents to retrieve
        
        Returns:
            List of relevant documents
        """
        
        if self.retriever is None:
            raise ValueError("Vectorstore not initialized. Call create_vectore_store first")
        return self.retriever.invoke(query)
    
    
    