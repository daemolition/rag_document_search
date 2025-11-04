import os
from dotenv import load_dotenv

# Laoding environments from file


class Config:
    """Configuration class for RAG system"""
    
    # Model
    LLM_MODEL="mistral-nemo:12b"

    # Document Processing
    CHUNK_SIZE=500
    CHUNK_OVERLAP=50

    DEFAULT_URLS = [
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
        "https://lilianweng.github.io/posts/2024-04-12-diffusion-video"
    ]