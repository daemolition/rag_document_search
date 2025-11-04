from src.states.rag_state import RAGState


class RAGNodes:
    """Contains node function for RAG workflows"""
    
    def __init__(self, retriever, llm):
        """
        Initialize RAG nodes
        
        Args:
            retriever: Document retriever instance
            llm: Language model instance
        """
        
        self.retriever = retriever
        self.llm = llm
        
    
    def retrieve_docs(self, state: RAGState) -> RAGState:
        """
        Retreives relevant documents node
        
        Args:
            state: Current RAG State
            
        Returns:
            Updated RAG state with retreived documents
        """    
        
        docs = self.retriever.invoke(state.question)
        
        return RAGState(
            question=state.question,
            retrieved_documents=docs
        )
    
    def generate_answer(self, state: RAGState) -> RAGState:
        """
        Generate the answer from retrieved documents node
        
        Args:
            state: Current RAG state woth retrieved documents
            
        Returns:
            Updated RAG state with generated answer
        """    
        
        #Combin retreived documents into context
        context = "\n\n".join([doc.page_content for doc in state.retrieved_documents])
        
        # Create prompt
        prompt = f"""
            Answer the question based on the context
            
            Context:
            {context}
            
            Question:
            {state.question}
        """
        
        # Generate response
        response = self.llm.invoke(prompt)
        
        return RAGState(
            question=state.question,
            retrieved_documents=state.retrieved_documents,
            answer=response.content
        )
        
    