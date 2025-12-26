# RAG CLI Tool

A simple, opinionated, and local-first Retrieval-Augmented Generation (RAG) tool for your terminal. It turns any folder of documents (`.txt`, `.md`, `.pdf`) into a searchable chatbot using local AI models.

## Features
- **100% Local**: Uses Ollama for privacy and offline capability.
- **Fast Search**: Powered by FAISS vector search.
- **Simple CLI**: Easy commands (`init`, `chat`, `rebuild`).
- **Strict Answers**: The bot answers *only* from your documents, reducing hallucinations.

## Prerequisites

1.  **Python 3.9+**
2.  **Ollama**: Download from [ollama.com](https://ollama.com).

### Model Setup
You need two models: one for understanding text (embeddings) and one for answering (chat).

```bash
# Pull the models (approx 2GB total)
ollama pull nomic-embed-text
ollama pull deepseek-r1:1.5b
```

Ensure Ollama is running (`ollama serve`).

## Installation

1.  **Install Python Dependencies**:
    ```bash
    python3 -m pip install "typer[all]" pyyaml rich openai faiss-cpu numpy pypdf
    ```

2.  **Make the CLI Executable**:
    ```bash
    chmod +x rag-cli
    ```

## Usage

### 1. Initialize
Scan a folder and build the search index. This creates a `rag.yaml` config file and a hidden `.rag_index` directory.

```bash
./rag-cli init /path/to/your/documents
```

### 2. Chat
Start the interactive chat session.

```bash
./rag-cli chat
```
*Type `exit` to quit.*

### 3. Rebuild
If you add new files to your folder or change settings in `rag.yaml`, rebuild the index:

```bash
./rag-cli rebuild
```

## Configuration
After running `init`, a `rag.yaml` file is created. You can edit this file to customize the behavior:

```yaml
folder_path: /path/to/your/documents
chunk_size: 1000        # Size of text chunks (characters)
overlap: 200            # Overlap between chunks to preserve context
embedding_model: nomic-embed-text
chat_model: deepseek-r1:1.5b
ignore_dirs:            # Folders to skip
  - .git
  - node_modules
```

## Troubleshooting

**"Permission denied" errors during install:**
This usually happens on macOS with the system Python. Use the provided `./rag-cli` wrapper script instead of trying to install it globally.

**"Connection refused" or Model errors:**
Make sure Ollama is running in a separate terminal:
```bash
ollama serve
```

**"I don't know":**
The bot is strictly instructed to only answer using the provided context. If the answer isn't in your files, it will refuse to answer.
