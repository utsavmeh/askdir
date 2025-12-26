import os
import yaml
from typing import List, Optional
from pathlib import Path
from dataclasses import dataclass, asdict

CONFIG_FILE_NAME = "rag.yaml"
INDEX_DIR_NAME = ".rag_index"

@dataclass
class RAGConfig:
    folder_path: str
    chunk_size: int = 1000
    overlap: int = 200
    embedding_model: str = "nomic-embed-text"
    chat_model: str = "deepseek-r1:1.5b"
    ignore_dirs: List[str] = None
    client_base_url: str = "http://localhost:11434/v1"
    client_api_key: str = "ollama"

    def __post_init__(self):
        if self.ignore_dirs is None:
            self.ignore_dirs = [".git", "node_modules", "venv", "__pycache__", ".rag_index"]

def load_config() -> Optional[RAGConfig]:
    """Loads configuration from the current working directory."""
    if not os.path.exists(CONFIG_FILE_NAME):
        return None
    
    with open(CONFIG_FILE_NAME, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    
    return RAGConfig(**data)

def save_config(config: RAGConfig):
    """Saves configuration to the current working directory."""
    with open(CONFIG_FILE_NAME, "w", encoding="utf-8") as f:
        yaml.dump(asdict(config), f, default_flow_style=False)
