from openai import OpenAI
from rag.config import RAGConfig

def get_client(config: RAGConfig) -> OpenAI:
    """Returns an initialized OpenAI client pointing to the configured base URL."""
    return OpenAI(
        base_url=config.client_base_url,
        api_key=config.client_api_key
    )
