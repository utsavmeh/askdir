import os
import sys
import json
import faiss
import numpy as np
from openai import OpenAI
from pypdf import PdfReader
from typing import List, Dict, Tuple

# ------------------------------------------------------------------------------
# SETUP INSTRUCTIONS
# ------------------------------------------------------------------------------
# 1. Install dependencies:
#    pip install openai faiss-cpu numpy pypdf
#
# 2. Install & Start Ollama:
#    - Download: https://ollama.com/
#    - Pull the model: `ollama pull functiongemma`
#    - Ensure Ollama is running (`ollama serve`)
#
# 3. Run the script:
#    python rag_chatbot.py /path/to/your/documents
# ------------------------------------------------------------------------------

# Configuration
# Pointing to local Ollama instance via OpenAI-compatible API
CLIENT_BASE_URL = "http://localhost:11434/v1"
CLIENT_API_KEY = "ollama"  # Not required for local, but needed by client init

# Use nomic-embed-text for embeddings and deepseek-r1 for chat
EMBEDDING_MODEL = "nomic-embed-text"
CHAT_MODEL = "deepseek-r1:1.5b"

CHUNK_SIZE = 1000  # Characters (approx 200-300 words)
CHUNK_OVERLAP = 200
INDEX_DIR = ".rag_index"
INDEX_FILE = "faiss.index"
META_FILE = "metadata.json"
IGNORE_DIRS = {".git", "node_modules", "venv", "__pycache__"}

# Initialize Client pointing to Local Ollama
client = OpenAI(
    base_url=CLIENT_BASE_URL,
    api_key=CLIENT_API_KEY
)

def load_files(folder_path: str) -> List[Dict[str, str]]:
    """Recursively loads .txt, .md, and .pdf files from the directory."""
    documents = []
    print(f"Scanning '{folder_path}'...")

    for root, dirs, files in os.walk(folder_path):
        # Filter out ignored directories
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]

        for file in files:
            file_path = os.path.join(root, file)
            ext = os.path.splitext(file)[1].lower()
            
            content = ""
            try:
                if ext in [".txt", ".md"]:
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()
                elif ext == ".pdf":
                    reader = PdfReader(file_path)
                    for page in reader.pages:
                        text = page.extract_text()
                        if text:
                            content += text + "\n"
                else:
                    continue  # Skip unsupported files

                if content.strip():
                    documents.append({"path": file_path, "content": content})
                    print(f"  Loaded: {file}")

            except Exception as e:
                print(f"  Error loading {file}: {e}")

    return documents

def chunk_text(documents: List[Dict[str, str]]) -> List[Dict]:
    """Splits document content into character-based chunks with overlap."""
    chunks = []
    
    print("Chunking text...")
    for doc in documents:
        text = doc["content"]
        total_len = len(text)
        
        start = 0
        while start < total_len:
            end = min(start + CHUNK_SIZE, total_len)
            chunk_str = text[start:end]
            
            chunks.append({
                "text": chunk_str,
                "source": doc["path"]
            })
            
            # Move window forward, accounting for overlap
            start += (CHUNK_SIZE - CHUNK_OVERLAP)
            
            if start >= total_len:
                break
                
    return chunks

def get_embeddings(text_chunks: List[str]) -> np.ndarray:
    """Generates embeddings for a list of text strings via Ollama."""
    embeddings = []
    # Ollama processes one by one or small batches. 
    # We'll do simple serial processing here to ensure stability.
    
    print(f"Generating embeddings for {len(text_chunks)} chunks using {EMBEDDING_MODEL}...")
    
    for i, text in enumerate(text_chunks):
        try:
            # Clean text to avoid issues
            clean_text = text.replace("\n", " ")
            response = client.embeddings.create(
                input=[clean_text],
                model=EMBEDDING_MODEL
            )
            emb = response.data[0].embedding
            embeddings.append(emb)
            
            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{len(text_chunks)}", end="\r")
                
        except Exception as e:
            print(f"\nError generating embedding for chunk {i}: {e}")
            sys.exit(1)
            
    print("") # Newline
    return np.array(embeddings, dtype="float32")

def create_index(chunks: List[Dict]):
    """Creates FAISS index and saves it along with metadata."""
    if not chunks:
        print("No content found to index.")
        return

    text_list = [c["text"] for c in chunks]
    embeddings = get_embeddings(text_list)
    
    if len(embeddings) == 0:
        print("No embeddings generated.")
        return

    # FAISS expects float32
    dimension = embeddings.shape[1]
    
    # Create L2 index
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    
    # Ensure output dir exists
    if not os.path.exists(INDEX_DIR):
        os.makedirs(INDEX_DIR)
        
    # Save index
    faiss.write_index(index, os.path.join(INDEX_DIR, INDEX_FILE))
    
    # Save metadata
    with open(os.path.join(INDEX_DIR, META_FILE), "w", encoding="utf-8") as f:
        json.dump(chunks, f)
        
    print(f"Index created with {len(chunks)} chunks and saved to '{INDEX_DIR}/'.")

def load_index() -> Tuple[faiss.Index, List[Dict]]:
    """Loads the FAISS index and metadata from disk."""
    index_path = os.path.join(INDEX_DIR, INDEX_FILE)
    meta_path = os.path.join(INDEX_DIR, META_FILE)
    
    if not os.path.exists(index_path) or not os.path.exists(meta_path):
        return None, None
        
    print("Loading existing index...")
    index = faiss.read_index(index_path)
    
    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
        
    return index, metadata

def retrieve_context(query: str, index: faiss.Index, metadata: List[Dict], k: int = 5) -> List[Dict]:
    """Retrieves the top k most similar chunks for the query."""
    try:
        query = query.replace("\n", " ")
        response = client.embeddings.create(
            input=[query],
            model=EMBEDDING_MODEL
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

def chat_loop(index: faiss.Index, metadata: List[Dict]):
    """Runs the REPL chat loop."""
    print("\n--- Local RAG Chatbot (Ollama: functiongemma) ---")
    print("Type 'exit' to quit.\n")
    
    while True:
        try:
            query = input("You: ").strip()
            if not query:
                continue
            if query.lower() in ["exit", "quit"]:
                break
                
            # Retrieve
            matches = retrieve_context(query, index, metadata)
            
            # Build Context
            context_text = "\n\n".join(
                [f"Source: {m['source']}\nContent: {m['text']}" for m in matches]
            )
            
            # Strict Prompt
            system_prompt = (
                "You are a helpful assistant. Use the following context to answer the user's question. "
                "If the answer is not contained in the context, say 'I don't know'."
            )
            
            user_message = f"Context:\n{context_text}\n\nQuestion: {query}"
            
            # Generate Answer
            response = client.chat.completions.create(
                model=CHAT_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.0
            )
            
            answer = response.choices[0].message.content
            print(f"Bot: {answer}\n")
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")

def main():
    if len(sys.argv) < 2:
        # If no arg provided, verify if we have a saved index we can use
        if os.path.exists(os.path.join(INDEX_DIR, INDEX_FILE)):
            print("No folder path provided, but found existing index.")
            use_existing = input("Use existing index? (y/n): ").lower()
            if use_existing != 'y':
                print("Usage: python rag_chatbot.py <folder_path>")
                sys.exit(1)
        else:
            print("Usage: python rag_chatbot.py <folder_path>")
            sys.exit(1)
        folder_path = None
    else:
        folder_path = sys.argv[1]

    # Initialize or Load
    index = None
    metadata = None

    # If folder is provided, rebuild (or check if user wants to rebuild)
    if folder_path:
        if os.path.exists(os.path.join(INDEX_DIR, INDEX_FILE)):
            print("Existing index found.")
            rebuild = input("Rebuild index? (y/n): ").lower()
            if rebuild == 'y':
                docs = load_files(folder_path)
                chunks = chunk_text(docs)
                create_index(chunks)
        else:
            docs = load_files(folder_path)
            chunks = chunk_text(docs)
            create_index(chunks)
            
    # Load index to memory
    index, metadata = load_index()
    
    if index and metadata:
        chat_loop(index, metadata)
    else:
        print("Could not load index. Exiting.")

if __name__ == "__main__":
    main()
