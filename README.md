# 🤖 Local RAG CLI & Web UI

A privacy-focused, 100% local Retrieval-Augmented Generation (RAG) tool. It turns any folder of documents (`.txt`, `.md`, `.pdf`) into a searchable knowledge base that you can chat with via your terminal or a web browser.

**Stack:** Python, Ollama, FAISS, FastAPI, Typer.

## ✨ Features

*   **100% Local**: No data leaves your machine. Powered by Ollama.
*   **Dual Interface**:
    *   💻 **CLI**: Fast, keyboard-centric terminal chat.
    *   🌐 **Web UI**: Clean, simple browser interface with source citations.
*   **Strict Answers**: The bot answers *only* from your documents to reduce hallucinations.
*   **Fast Retrieval**: Uses FAISS for efficient vector similarity search.

---

## 🚀 Prerequisites

1.  **Python 3.9+** installed.
2.  **Ollama**: Download from [ollama.com](https://ollama.com).

### Model Setup
You need to pull the specific models this tool uses (Embeddings + Chat):

```bash
# Pull the models (approx 2GB total)
ollama pull nomic-embed-text
ollama pull deepseek-r1:1.5b
```
*Ensure Ollama is running in the background (`ollama serve`).*

---

## 📦 Installation

1.  **Install Python Dependencies**:
    ```bash
    python3 -m pip install "typer[all]" pyyaml rich openai faiss-cpu numpy pypdf fastapi uvicorn
    ```

2.  **Enable the Wrapper Script**:
    ```bash
    chmod +x rag-cli
    ```

---

## 🛠 Usage

### 1. Initialize Index
Scan a folder and build the search index. This creates a `rag.yaml` config file and a hidden `.rag_index` directory in your current path.

```bash
./rag-cli init /path/to/your/documents
```

### 2. Chat (Web UI)
Launch the local web server to chat in your browser.

```bash
./rag-cli ui
```
> Open **[http://localhost:8000](http://localhost:8000)** in your browser.
> 
> **New Features:**
> *   **Thinking Indicator**: See when the bot is processing your request.
> *   **Rebuild from UI**: Click the "Rebuild Index" button to refresh your knowledge base without restarting the server.

### 3. Chat (Terminal)
Start a quick interactive session directly in the terminal.

```bash
./rag-cli chat
```

### 4. Rebuild Index
If you add new files to your folder or modify `rag.yaml`, update the index:

```bash
./rag-cli rebuild
```

---

## ⚙️ Configuration
After running `init`, a `rag.yaml` file is generated. You can customize it:

```yaml
folder_path: /path/to/your/documents
chunk_size: 1000        # Character count per chunk
overlap: 200            # Overlap to preserve context
embedding_model: nomic-embed-text
chat_model: deepseek-r1:1.5b
client_base_url: http://localhost:11434/v1
ignore_dirs:
  - .git
  - node_modules
```

---

## 📂 Project Structure

```text
.
├── rag-cli             # Executable wrapper script
├── rag.yaml            # Configuration (auto-generated)
├── .rag_index/         # FAISS index & metadata storage
├── rag/                # Source Code
│   ├── cli.py          # Command line entry points
│   ├── server.py       # FastAPI backend & HTML frontend
│   ├── ingest.py       # File loader
│   ├── index.py        # Vector database logic
│   └── chat.py         # Generation logic
└── README.md
```

## ❓ Troubleshooting

*   **"Permission denied"**: Use `./rag-cli` instead of trying to run `rag` directly if you haven't installed the package globally.
*   **"Connection refused"**: Make sure `ollama serve` is running.
*   **"I don't know"**: The bot is instructed to be strict. If the answer isn't in your files, it won't make one up.