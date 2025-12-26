import os
import json
import faiss
import numpy as np
from typing import List, Dict, Tuple
from rich.progress import track
from rag.config import INDEX_DIR_NAME

INDEX_FILE = "faiss.index"
META_FILE = "metadata.json"

def get_embeddings(client, texts: List[str], model: str) -> np.ndarray:
    """Generates embeddings for a list of text strings via the client."""
    embeddings = []
    
    # Process sequentially with a progress bar
    for text in track(texts, description="Generating embeddings..."):
        try:
            clean_text = text.replace("\n", " ")
            response = client.embeddings.create(
                input=[clean_text],
                model=model
            )
            emb = response.data[0].embedding
            embeddings.append(emb)
        except Exception as e:
            # In a real app, might want to retry or log error better
            print(f"Error embedding chunk: {e}")
            # Append zero vector or skip? 
            # Skipping breaks alignment with chunks list.
            # We'll fail hard here for the MVP to ensure integrity.
            raise e
            
    return np.array(embeddings, dtype="float32")

def build_index(client, chunks: List[Dict], config):
    """Creates FAISS index and saves it along with metadata."""
    if not chunks:
        print("No content found to index.")
        return

    text_list = [c["text"] for c in chunks]
    embeddings = get_embeddings(client, text_list, config.embedding_model)
    
    if len(embeddings) == 0:
        print("No embeddings generated.")
        return

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    
    # Ensure output dir exists
    if not os.path.exists(INDEX_DIR_NAME):
        os.makedirs(INDEX_DIR_NAME)
        
    # Save index
    faiss.write_index(index, os.path.join(INDEX_DIR_NAME, INDEX_FILE))
    
    # Save metadata
    with open(os.path.join(INDEX_DIR_NAME, META_FILE), "w", encoding="utf-8") as f:
        json.dump(chunks, f)

def load_faiss_index() -> Tuple[faiss.Index, List[Dict]]:
    """Loads the FAISS index and metadata from disk."""
    index_path = os.path.join(INDEX_DIR_NAME, INDEX_FILE)
    meta_path = os.path.join(INDEX_DIR_NAME, META_FILE)
    
    if not os.path.exists(index_path) or not os.path.exists(meta_path):
        return None, None
        
    index = faiss.read_index(index_path)
    
    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
        
    return index, metadata
