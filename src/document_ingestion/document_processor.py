from pathlib import Path
from typing import List, Union

from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader, PyPDFDirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


class DocumentProcessor:
    """Handles document loading and processing"""
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """Initializes the DocumentProcessor with chunk size and overlap
        
        Args:
            chunk (int): The maximum number of characters
            chunk_overlap (int): The number of characters to overlap
        """    
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        
    def load_from_url(self, url: str ) -> List[Document]:
        """Load Documents from urls"""
        loader = WebBaseLoader(url)
        return loader.load()
    
 
    def load_from_pdf_dir(self, directory: Union[str, Path] ) -> List[Document]:
        """Load all PDFs inside a directory"""
        loader = PyPDFDirectoryLoader(directory)
        return loader.load()   
    
    
    def load_from_text(self, text: Union[str, Path]) -> List[Document]:
        """Load document(s) from text file"""
        loader = TextLoader(text)
        return loader.load()
    

    def load_from_pdf(self, directory: Union[str, Path] ) -> List[Document]:
        """Load all PDFs inside a directory"""
        loader = PyPDFLoader(directory)
        return loader.load()   
    
    
    def load_documents(self, sources: List[str]) -> List[Document]:
        """
        Load documents from URLs, PDF directories, or TXT files
        
        Args:
            sources: List of URLs, PDF folder paths, or TXT file paths
            
        Returns:
            List of Document objects loaded from the given sources
        """
        
        docs = []
        
        for src in sources:
            if src.startswith('http://') or src.startswith('https://'):
                docs.extend(self.load_from_url(src))
            
            path = Path("data")
            if path.is_dir():
                docs.extend(self.load_from_pdf_dir(path))
            elif path.suffix.lower() == '.pdf':
                docs.extend(self.load_from_pdf(src))
            elif path.suffix.lower() == '.txt':
                docs.extend(self.load_from_text(path))
            else:
                raise ValueError(
                    f"Unsupported source type: {src}."
                    "Usr URL, .txt file, or PDF Directory"
                )
            
        return docs
    
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks
        
        Args: 
            documents: List of documents to split
        
        Returns:
            List of split documents
        """
        
        return self.splitter.split_documents(documents)
    
    
    def process_urls(self, urls: List[str]) -> List[Document]:
        """
        Complete pipeline to load and split documents
        
        Args:
            urls: List of URLs to process
            
        Returns:
            List of processed document chunks
        """
        docs = self.load_documents(urls)
        return self.split_documents(docs)
    