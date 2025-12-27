import os
import threading
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Optional

from rag.config import load_config, save_config
from rag.index import load_faiss_index, build_index
from rag.utils import get_client
from rag.retrieve import retrieve_context
from rag.chat import generate_response
from rag.ingest import load_files
from rag.chunking import chunk_text

app = FastAPI(title="Local RAG UI")

# Global State
class State:
    config = None
    index = None
    metadata = None
    client = None
    rebuild_status = "idle" # idle, running, success, error
    rebuild_message = ""

state = State()

# Load resources on startup
@app.on_event("startup")
def startup_event():
    state.config = load_config()
    if not state.config:
        print("Warning: No rag.yaml found. Please run 'rag init'.")
        return
    
    state.index, state.metadata = load_faiss_index()
    if not state.index:
        print("Warning: No index found. Please run 'rag init' or 'rag rebuild'.")
        # We don't return here so UI can still load to allow rebuilding
        
    state.client = get_client(state.config)
    print("RAG System Loaded Successfully.")

# Data Models
class ChatRequest(BaseModel):
    query: str

class ChatResponse(BaseModel):
    answer: str
    sources: List[str]

class RebuildRequest(BaseModel):
    folder_path: Optional[str] = None

class RebuildStatusResponse(BaseModel):
    status: str
    message: str

# Rebuild Logic
def run_rebuild_task(folder_path: Optional[str]):
    try:
        state.rebuild_status = "running"
        state.rebuild_message = "Starting rebuild..."
        
        # 1. Update config if path provided
        if folder_path and folder_path.strip():
            state.config.folder_path = folder_path.strip()
            save_config(state.config)
            state.rebuild_message = f"Updated config to: {state.config.folder_path}"
        
        # 2. Load
        state.rebuild_message = "Scanning files..."
        docs = load_files(state.config.folder_path, state.config.ignore_dirs)
        if not docs:
            state.rebuild_status = "error"
            state.rebuild_message = "No valid documents found."
            return

        # 3. Chunk
        state.rebuild_message = f"Chunking {len(docs)} documents..."
        chunks = chunk_text(docs, state.config.chunk_size, state.config.overlap)
        
        # 4. Index
        state.rebuild_message = f"Indexing {len(chunks)} chunks..."
        build_index(state.client, chunks, state.config)
        
        # 5. Reload
        state.rebuild_message = "Reloading index..."
        state.index, state.metadata = load_faiss_index()
        
        state.rebuild_status = "success"
        state.rebuild_message = "Index rebuilt successfully!"
        
    except Exception as e:
        state.rebuild_status = "error"
        state.rebuild_message = f"Error: {str(e)}"
        print(f"Rebuild failed: {e}")

# Endpoints
@app.post("/rebuild")
def trigger_rebuild(request: RebuildRequest, background_tasks: BackgroundTasks):
    if state.rebuild_status == "running":
        raise HTTPException(status_code=400, detail="Rebuild already in progress")
    
    if not state.config and not request.folder_path:
        raise HTTPException(status_code=400, detail="No config found. Please provide folder path.")
        
    background_tasks.add_task(run_rebuild_task, request.folder_path)
    return {"status": "started"}

@app.get("/rebuild/status", response_model=RebuildStatusResponse)
def get_rebuild_status():
    return RebuildStatusResponse(status=state.rebuild_status, message=state.rebuild_message)

@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    if state.rebuild_status == "running":
         raise HTTPException(status_code=503, detail="Index is currently rebuilding. Please wait.")
         
    if not state.client or not state.index:
        raise HTTPException(status_code=500, detail="RAG system not initialized. Please rebuild index.")

    # Retrieve
    matches = retrieve_context(
        state.client, 
        request.query, 
        state.index, 
        state.metadata, 
        state.config.embedding_model
    )
    
    # Build Context
    context_text = "\n\n".join(
        [f"Source: {m['source']}\nContent: {m['text']}" for m in matches]
    )
    
    # Generate
    answer = generate_response(
        state.client, 
        state.config.chat_model, 
        context_text, 
        request.query
    )
    
    # Extract Sources
    sources = [os.path.basename(m['source']) for m in matches]
    
    return ChatResponse(answer=answer, sources=sources)

HTML_CONTENT = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Local RAG</title>
    <style>
        :root {
            --bg-body: #ffffff;
            --bg-subtle: #f6f8fa;
            --border-default: #d0d7de;
            --border-subtle: #eff2f5;
            --text-primary: #1f2328;
            --text-secondary: #656d76;
            --accent-blue: #0969da;
            --accent-blue-bg: #ddf4ff;
            --danger: #cf222e;
            --font-stack: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
            --shadow-sm: 0 1px 2px rgba(0,0,0,0.05);
        }

        * { box-sizing: border-box; }

        body {
            font-family: var(--font-stack);
            background-color: var(--bg-body);
            color: var(--text-primary);
            margin: 0;
            display: flex;
            flex-direction: column;
            height: 100vh;
            font-size: 14px;
            line-height: 1.5;
        }

        /* --- Header --- */
        header {
            height: 60px;
            padding: 0 24px;
            border-bottom: 1px solid var(--border-default);
            display: flex;
            justify-content: space-between;
            align-items: center;
            background: #fff;
            position: sticky;
            top: 0;
            z-index: 10;
        }

        h1 {
            margin: 0;
            font-size: 16px;
            font-weight: 600;
            color: var(--text-primary);
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        h1::before {
            content: '';
            display: block;
            width: 12px;
            height: 12px;
            background: var(--text-primary);
            border-radius: 50%;
        }

        .rebuild-btn {
            background: transparent;
            border: 1px solid var(--border-default);
            color: var(--text-secondary);
            padding: 6px 12px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 12px;
            font-weight: 500;
            transition: all 0.2s ease;
        }

        .rebuild-btn:hover {
            background: var(--bg-subtle);
            color: var(--text-primary);
            border-color: #b1bac4;
        }

        /* --- Main Layout --- */
        #chat-container {
            flex: 1;
            display: flex;
            flex-direction: column;
            max-width: 800px; /* Constrained width for readability */
            width: 100%;
            margin: 0 auto;
            position: relative;
            background: var(--bg-body);
        }

        #messages {
            flex: 1;
            overflow-y: auto;
            padding: 24px;
            scroll-behavior: smooth;
        }

        /* --- Messages --- */
        .message {
            margin-bottom: 24px;
            max-width: 100%;
            animation: fadeIn 0.3s ease;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(5px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .message-content {
            display: inline-block;
            padding: 12px 16px;
            border-radius: 8px;
            font-size: 14px;
            position: relative;
            max-width: 85%;
        }

        /* User Message */
        .user-msg {
            text-align: right;
        }
        .user-msg .message-content {
            background-color: var(--accent-blue-bg);
            color: var(--text-primary);
            border: 1px solid transparent;
        }

        /* Bot Message */
        .bot-msg {
            text-align: left;
        }
        .bot-msg .message-content {
            background-color: var(--bg-subtle);
            color: var(--text-primary);
            border: 1px solid var(--border-subtle);
        }

        .sources {
            margin-top: 8px;
            font-size: 11px;
            color: var(--text-secondary);
            font-family: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace;
            padding-top: 8px;
            border-top: 1px solid rgba(0,0,0,0.05);
        }

        /* --- Input Area --- */
        #input-area {
            padding: 24px;
            border-top: 1px solid transparent; /* Cleaner look without hard border */
            background: var(--bg-body); /* Sticky bottom needs bg */
            display: flex;
            gap: 12px;
            position: sticky;
            bottom: 0;
        }

        .input-wrapper {
            position: relative;
            flex: 1;
        }

        input {
            width: 100%;
            padding: 12px 16px;
            border: 1px solid var(--border-default);
            border-radius: 6px;
            font-size: 14px;
            outline: none;
            transition: border-color 0.2s, box-shadow 0.2s;
            background: var(--bg-body);
            color: var(--text-primary);
            box-shadow: var(--shadow-sm);
        }

        input:focus {
            border-color: var(--accent-blue);
            box-shadow: 0 0 0 3px rgba(9, 105, 218, 0.15);
        }

        button#sendBtn {
            padding: 0 20px;
            background: var(--text-primary);
            color: #fff;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-size: 14px;
            font-weight: 500;
            transition: opacity 0.2s;
            box-shadow: var(--shadow-sm);
        }

        button#sendBtn:hover { opacity: 0.9; }
        button:disabled { opacity: 0.5; cursor: not-allowed; }

        /* --- Status & Indicators --- */
        #thinkingIndicator {
            display: none;
            padding: 0 24px;
            margin-bottom: 12px;
            color: var(--text-secondary);
            font-size: 12px;
            align-items: center;
            gap: 6px;
        }
        
        .pulse-dot {
            width: 8px;
            height: 8px;
            background-color: var(--text-secondary);
            border-radius: 50%;
            opacity: 0.4;
            animation: pulse 1.5s infinite ease-in-out;
        }

        @keyframes pulse {
            0%, 100% { opacity: 0.4; transform: scale(0.8); }
            50% { opacity: 1; transform: scale(1.1); }
        }

        #status-bar {
            padding: 8px 24px;
            font-size: 12px;
            font-weight: 500;
            text-align: center;
            display: none;
        }
        .status-success { background: #dafbe1; color: #1a7f37; }
        .status-error { background: #ffebe9; color: var(--danger); }
        .status-info { background: var(--bg-subtle); color: var(--text-secondary); }

        /* --- Modal --- */
        .modal {
            display: none;
            position: fixed;
            z-index: 1000;
            left: 0; top: 0;
            width: 100%; height: 100%;
            background-color: rgba(255,255,255,0.8);
            backdrop-filter: blur(2px);
            align-items: center;
            justify-content: center;
        }

        .modal-content {
            background-color: #fff;
            padding: 32px;
            border-radius: 12px;
            width: 400px;
            box-shadow: 0 8px 24px rgba(0,0,0,0.12);
            border: 1px solid var(--border-default);
        }

        .modal h3 { margin: 0 0 16px 0; font-size: 18px; }
        .modal p { color: var(--text-secondary); margin-bottom: 16px; }
        
        .modal input { margin-bottom: 24px; }

        .modal-actions {
            display: flex;
            justify-content: flex-end;
            gap: 12px;
        }
        
        .btn-secondary {
            background: transparent;
            border: none;
            color: var(--text-secondary);
            cursor: pointer;
            font-weight: 500;
        }
        .btn-secondary:hover { color: var(--text-primary); }

        .btn-primary {
            background: var(--text-primary);
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 6px;
            cursor: pointer;
            font-weight: 500;
        }

    </style>
</head>
<body>

    <header>
        <h1>Local RAG</h1>
        <button class="rebuild-btn" onclick="openModal()">Rebuild Index</button>
    </header>

    <div id="status-bar"></div>

    <div id="chat-container">
        <div id="messages">
            <!-- Messages will appear here -->
        </div>

        <div id="thinkingIndicator">
            <div class="pulse-dot"></div>
            <span>Thinking...</span>
        </div>

        <div id="input-area">
            <div class="input-wrapper">
                <input type="text" id="queryInput" placeholder="Ask a question..." autocomplete="off" autofocus>
            </div>
            <button id="sendBtn" onclick="sendMessage()">Send</button>
        </div>
    </div>

    <!-- Rebuild Modal -->
    <div id="rebuildModal" class="modal">
        <div class="modal-content">
            <h3>Rebuild Index</h3>
            <p>Enter a new folder path to re-index, or leave empty to rebuild using the current configuration.</p>
            <input type="text" id="folderPath" placeholder="/path/to/documents">
            <div class="modal-actions">
                <button class="btn-secondary" onclick="closeModal()">Cancel</button>
                <button class="btn-primary" onclick="startRebuild()">Start Rebuild</button>
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
    <script>
        const input = document.getElementById('queryInput');
        const messagesDiv = document.getElementById('messages');
        const sendBtn = document.getElementById('sendBtn');
        const thinkingIndicator = document.getElementById('thinkingIndicator');
        const modal = document.getElementById('rebuildModal');
        const statusBar = document.getElementById('status-bar');

        // Configure marked to be safe
        marked.setOptions({
            breaks: true,
            gfm: true
        });

        // --- Chat Logic ---

        input.addEventListener('keypress', function (e) {
            if (e.key === 'Enter') sendMessage();
        });

        async function sendMessage() {
            const query = input.value.trim();
            if (!query) return;

            // UI Updates
            appendMessage(query, 'user');
            input.value = '';
            input.disabled = true;
            sendBtn.disabled = true;
            thinkingIndicator.style.display = 'flex';
            scrollToBottom();

            try {
                const response = await fetch('/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ query: query })
                });

                if (response.status === 503) {
                    throw new Error("System is rebuilding. Please wait.");
                }
                if (!response.ok) {
                    throw new Error("Failed to get response");
                }

                const data = await response.json();
                appendMessage(data.answer, 'bot', data.sources);

            } catch (error) {
                appendMessage("Error: " + error.message, 'bot');
            } finally {
                input.disabled = false;
                sendBtn.disabled = false;
                thinkingIndicator.style.display = 'none';
                input.focus();
                scrollToBottom();
            }
        }

        function appendMessage(text, sender, sources = []) {
            const msgDiv = document.createElement('div');
            msgDiv.className = `message ${sender}-msg`;
            
            // Use marked for Markdown rendering
            let contentHtml = `<div class="message-content"><span>${marked.parse(text)}</span>`;
            
            if (sources && sources.length > 0) {
                const uniqueSources = [...new Set(sources)];
                contentHtml += `<div class="sources">Sources: ${uniqueSources.join(', ')}</div>`;
            }
            contentHtml += `</div>`;
            
            msgDiv.innerHTML = contentHtml;
            messagesDiv.appendChild(msgDiv);
            scrollToBottom();
        }

        function scrollToBottom() {
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
        }

        // --- Rebuild Logic ---

        function openModal() { 
            modal.style.display = 'flex'; 
            document.getElementById('folderPath').focus();
        }
        function closeModal() { modal.style.display = 'none'; }

        // Close modal on outside click
        window.onclick = function(event) {
            if (event.target == modal) {
                closeModal();
            }
        }

        async function startRebuild() {
            const folder = document.getElementById('folderPath').value.trim();
            closeModal();
            
            try {
                const response = await fetch('/rebuild', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ folder_path: folder || null })
                });
                
                if (response.ok) {
                    pollStatus();
                } else {
                    const err = await response.json();
                    showStatus(err.detail || "Failed to start", "error");
                }
            } catch (e) {
                showStatus("Network error", "error");
            }
        }

        async function pollStatus() {
            const interval = setInterval(async () => {
                try {
                    const res = await fetch('/rebuild/status');
                    const data = await res.json();
                    
                    if (data.status === 'running') {
                        showStatus(data.message, "info");
                        input.disabled = true;
                        sendBtn.disabled = true;
                    } else if (data.status === 'success') {
                        clearInterval(interval);
                        showStatus(data.message, "success");
                        input.disabled = false;
                        sendBtn.disabled = false;
                        setTimeout(() => statusBar.style.display = 'none', 5000);
                    } else if (data.status === 'error') {
                        clearInterval(interval);
                        showStatus(data.message, "error");
                        input.disabled = false;
                        sendBtn.disabled = false;
                    } else {
                        // Idle or unknown, stop polling
                        clearInterval(interval);
                    }
                } catch (e) {
                    clearInterval(interval);
                }
            }, 1000);
        }

        function showStatus(msg, type='info') {
            statusBar.style.display = 'block';
            statusBar.className = '';
            if (type === 'success') statusBar.classList.add('status-success');
            if (type === 'error') statusBar.classList.add('status-error');
            if (type === 'info') statusBar.classList.add('status-info');
            statusBar.innerHTML = msg;
        }
        
        // Check status on load: Only poll if actually running
        (async function checkExistingRebuild() {
            try {
                const res = await fetch('/rebuild/status');
                const data = await res.json();
                if (data.status === 'running') {
                    pollStatus();
                }
            } catch (e) {}
        })();

    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def get_ui():
    return HTML_CONTENT