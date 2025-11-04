from langchain_ollama import ChatOllama

class LLM:
    """Class for initializing the llm"""
    
    def __init__(self, model: str = ""):
        self.model=model
        self._llm=None
        
    @property
    def get_llm(self):
        self._llm = ChatOllama(
            model=self.model,
            num_ctx=8192
        )
        return self._llm