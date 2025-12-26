import faiss
import numpy as np
from typing import List, Dict

def retrieve_context(client, query: str, index: faiss.Index, metadata: List[Dict], model: str, k: int = 5) -> List[Dict]:
    """Retrieves the top k most similar chunks for the query."""
    try:
        query = query.replace("\n", " ")
        response = client.embeddings.create(
            input=[query],
            model=model
        )
        query_vec = np.array([response.data[0].embedding], dtype="float32")
        
        # Search index
        distances, indices = index.search(query_vec, k)
        
        results = []
        for idx in indices[0]:
            if idx != -1 and idx < len(metadata):
                results.append(metadata[idx])
                
        return results
    except Exception as e:
        print(f"Error during retrieval: {e}")
        return []
