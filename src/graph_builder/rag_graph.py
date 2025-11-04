from langgraph.graph import StateGraph, START, END
from src.states.rag_state import RAGState
from src.nodes.react_node import RAGNodes

class GraphBuilder:
    """Builds and manages the Langgraph Workflow"""
    
    def __init__(self, retriever, llm):
        """
        Initialize the graph builder
        
        Args:
            retriever: Document retrieval instance
            llm: Language model instance
        """
        
        self.nodes=RAGNodes(retriever, llm)
        self.graph=None
        
    
    def build(self):
        """
        Build the RAG workflow graph
        
        Returns:
            Compiled graph instance
        """    
        
        # Create state graph
        builder = StateGraph(RAGState)
        
        # Add nodes
        builder.add_node("retriever", self.nodes.retrieve_docs)
        builder.add_node("responder", self.nodes.generate_answer)
        
        # Set entrypoint
        builder.set_entry_point("retriever")
        
        # Add edges
        builder.add_edge("retriever", "responder")
        builder.add_edge("responder", END)
        
        # Compile and return the graph
        self.graph = builder.compile()
        return self.graph
    
    
    def run(self, question: str) -> dict:
        """
        Run the RAG workflow
        
        Args: 
            question: User question
            
        Returns:
            Final state with answer
        """
        
        if self.graph is None:
            self.build()
            
        initial_state = RAGState(question=question)
        return self.graph.invoke(initial_state)