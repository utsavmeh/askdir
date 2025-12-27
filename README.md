# 🤖 Local RAG CLI & Web UI

A professional, 100% local Retrieval-Augmented Generation (RAG) system. Turn any directory of documents (`.pdf`, `.md`, `.txt`) into a searchable knowledge base with a clean Web UI and a powerful CLI.

**Built for developers who value privacy and simplicity.**

## ✨ Features

*   **100% Local**: No data ever leaves your machine. Powered by Ollama.
*   **Modern Web UI**: Clean, developer-tool aesthetic with "thinking" states and markdown support.
*   **Dual Interface**: Switch between a terminal-based REPL and a browser-based chat.
*   **Strict Context**: Instructs the LLM to answer *only* from your data to prevent hallucinations.
*   **On-the-fly Rebuilding**: Refresh your index directly from the UI or CLI.
*   **Opinionated Defaults**: Pre-configured with `deepseek-r1` and `nomic-embed-text`.

---

## 🚀 Prerequisites

1.  **Python 3.9+**
2.  **Ollama**: Install from [ollama.com](https://ollama.com).

### Default Model Setup
By default, this tool expects the following models:
```bash
ollama pull nomic-embed-text
ollama pull deepseek-r1:1.5b
```
*Ensure Ollama is running (`ollama serve`).*

---

## 📦 Installation

1.  **Install Dependencies**:
    ```bash
    python3 -m pip install "typer[all]" pyyaml rich openai faiss-cpu numpy pypdf fastapi uvicorn
    ```

2.  **Prepare the Launcher**:
    ```bash
    chmod +x rag-cli
    ```

---

## 🛠 Usage

### 1. Initialize
Scan a folder to build the initial vector index. This creates a `rag.yaml` config and a `.rag_index/` folder.
```bash
./rag-cli init /path/to/your/docs
```

### 2. Chat via Web UI (Recommended)
Launch the browser-based interface.
```bash
./rag-cli ui
```
> Access at: **[http://localhost:8000](http://localhost:8000)**

### 3. Chat via Terminal
For quick CLI access:
```bash
./rag-cli chat
```

### 4. Rebuild Index
If you add or change documents, sync the index:
```bash
./rag-cli rebuild
```
*(You can also do this via the "Rebuild Index" button in the Web UI).*

---

## ⚙️ Customizing Models

You can use **any model** supported by Ollama. To switch models:

1.  **Pull the new model**:
    ```bash
    ollama pull llama3
    ```
2.  **Update `rag.yaml`**:
    Open the generated `rag.yaml` file and change the model names:
    ```yaml
    embedding_model: nomic-embed-text  # Model for vector search
    chat_model: llama3                 # Model for generating answers
    ```
3.  **Rebuild**:
    If you changed the `embedding_model`, you **must** rebuild the index:
    ```bash
    ./rag-cli rebuild
    ```

---

## 📂 Project Structure

```text
.
├── rag-cli             # CLI Launcher
├── rag.yaml            # Local Configuration (Auto-generated)
├── .rag_index/         # FAISS Index & Metadata (Auto-generated)
└── rag/                # Source Code
    ├── cli.py          # CLI commands (init, chat, ui)
    ├── server.py       # FastAPI backend & Web UI
    ├── chat.py         # LLM interaction logic
    ├── index.py        # FAISS & Embedding logic
    ├── ingest.py       # File scanning & PDF parsing
    ├── chunking.py     # Text splitting logic
    ├── config.py       # YAML Configuration handler
    └── utils.py        # Shared helpers
```

## ❓ FAQ

**Can I use it for my resume?**
Yes. Point it at a folder with your resume PDF, run `init`, and ask questions like "What are the user's skills?"

**Why does it say "I don't know"?**
The system prompt is very strict. It will refuse to answer if the information is not explicitly found in your documents.

**Does it support folders?**
Yes, it recursively scans all sub-folders, ignoring common junk like `.git` and `node_modules`.
