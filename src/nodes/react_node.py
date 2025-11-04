from typing import Optional, List, Any
from src.states.rag_state import RAGState

from langchain_core.documents import Document
from langchain_core.tools import Tool
from langchain_core.messages import HumanMessage
from langchain.agents import create_agent
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun


class RAGNodes:
    """Contains the node functions for RAG workfllow"""
    
    def __init__(self, retriever, llm):
        self.retriever=retriever
        self.llm=llm
        self._agent=None # Lazy Init
        
    
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
        
    def _build_tools(self) -> List[Tool]:
        """Build retriever and wikipedia tool"""
        
        def retriever_tool_fn(query: str) -> str:
            docs: List[Document] = self.retriever.invoke(query)
            
            if not docs:
                return "No documents found."
            
            merged = []
            
            for i, doc in enumerate(docs[:8], start=1):
                meta = doc.metadata if hasattr(doc, "metadata") else {}
                title = meta.get("title") or meta.get("source") or f"doc_{i}"
                merged.append(f"[{i}] {title}\n{doc.page_content}")
            return "\n\n".join(merged)
        
        retriever_tool = Tool(
            name="retriever",
            description="Fetch passages from indexed vectorstore.",
            func=retriever_tool_fn
        )
        
        wiki = WikipediaQueryRun(
            api_wrapper=WikipediaAPIWrapper(
                top_k_results=3, 
                lang="en",
                wiki_client=Any
            )
        )
        
        wikipedia_tool = Tool(
            name="wikipedia",
            description="Search Wikipedia for general knwoledge.",
            func=wiki.run,
        )
        
        return [retriever_tool, wikipedia_tool]                
        
    
    def _build_agent(self):
        """ReAct agent with tools"""
        
        tools = self._build_tools()
        system_prompt = """
            You are a helpful RAG agent.
            Prefer 'retriever' for user-provided docs; use 'wikipedia' for general knowledge.
            Return only the final useful answer.
        """
        self._agent =create_agent(
            self.llm, 
            tools=tools, 
            system_prompt=system_prompt)
        
    
    def generate_answer(self, state: RAGState) -> RAGState:
        """
        Generate the answer from retrieved documents node
        
        Args:
            state: Current RAG state woth retrieved documents
            
        Returns:
            Updated RAG state with generated answer
        """    
        
        if self._agent is None:
            self._build_agent()
            
        result = self._agent.invoke({"messages": [HumanMessage(content=state.question)]})
        
        messages = result.get("messages", [])
        answer: Optional[str] = None
        
        if messages:
            answer_msg = messages[-1]
            answer = getattr(answer_msg, "content", None)
            
        return RAGState(
            question=state.question,
            retrieved_documents=state.retrieved_documents,
            answer=answer or "Could not generate answer"
        )
        
        
    