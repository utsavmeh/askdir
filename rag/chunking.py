from typing import List, Dict

def chunk_text(documents: List[Dict[str, str]], chunk_size: int, overlap: int) -> List[Dict]:
    """Splits document content into character-based chunks with overlap."""
    chunks = []
    
    for doc in documents:
        text = doc["content"]
        total_len = len(text)
        
        start = 0
        while start < total_len:
            end = min(start + chunk_size, total_len)
            chunk_str = text[start:end]
            
            chunks.append({
                "text": chunk_str,
                "source": doc["path"]
            })
            
            # Move window forward, accounting for overlap
            start += (chunk_size - overlap)
            
            if start >= total_len:
                break
                
    return chunks
