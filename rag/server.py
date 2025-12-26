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
    <title>Local RAG Chat</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; max-width: 900px; margin: 0 auto; padding: 20px; background: #f4f4f9; display: flex; flex-direction: column; height: 95vh; }
        
        /* Header */
        header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; }
        h1 { margin: 0; font-size: 1.5rem; color: #333; }
        .rebuild-btn { background: #6c757d; color: white; border: none; padding: 8px 15px; border-radius: 4px; cursor: pointer; font-size: 0.9rem; }
        .rebuild-btn:hover { background: #5a6268; }

        /* Chat Area */
        #chat-container { background: white; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); overflow: hidden; display: flex; flex-direction: column; flex: 1; position: relative; }
        #messages { flex: 1; overflow-y: auto; padding: 20px; scroll-behavior: smooth; }
        
        .message { margin-bottom: 15px; line-height: 1.5; max-width: 80%; }
        .user-msg { margin-left: auto; text-align: right; }
        .user-msg span { background: #007bff; color: white; padding: 10px 15px; border-radius: 15px 15px 0 15px; display: inline-block; }
        .bot-msg { margin-right: auto; text-align: left; }
        .bot-msg span { background: #e9ecef; color: #333; padding: 10px 15px; border-radius: 15px 15px 15px 0; display: inline-block; }
        
        .sources { font-size: 0.75em; color: #666; margin-top: 5px; font-style: italic; }
        
        /* Input Area */
        #input-area { border-top: 1px solid #ddd; padding: 20px; display: flex; gap: 10px; background: #fff; }
        input { flex: 1; padding: 12px; border: 1px solid #ddd; border-radius: 4px; font-size: 16px; outline: none; }
        input:focus { border-color: #007bff; }
        button#sendBtn { padding: 10px 25px; background: #28a745; color: white; border: none; border-radius: 4px; cursor: pointer; font-size: 16px; font-weight: bold; }
        button#sendBtn:hover { background: #218838; }
        button:disabled { background: #ccc !important; cursor: not-allowed; }

        /* Thinking Indicator */
        .thinking { font-style: italic; color: #888; font-size: 0.9rem; margin-bottom: 10px; display: none; align-items: center; gap: 8px; }
        .thinking .dots { display: inline-block; animation: ellipsis 1.5s infinite; }
        @keyframes ellipsis {
            0% { content: '.'; }
            33% { content: '..'; }
            66% { content: '...'; }
        }

        /* Modal */
        .modal { display: none; position: fixed; z-index: 1000; left: 0; top: 0; width: 100%; height: 100%; background-color: rgba(0,0,0,0.5); align-items: center; justify-content: center; }
        .modal-content { background-color: #fefefe; padding: 25px; border-radius: 8px; width: 400px; box-shadow: 0 4px 20px rgba(0,0,0,0.2); }
        .modal h3 { margin-top: 0; }
        .modal input { width: 100%; box-sizing: border-box; margin: 15px 0; }
        .modal-actions { display: flex; justify-content: flex-end; gap: 10px; }
        .close-btn { background: #ddd; color: #333; }
        
        /* Status Bar */
        #status-bar { display: none; padding: 10px; background: #e2e3e5; color: #383d41; text-align: center; font-size: 0.9rem; border-radius: 4px; margin-bottom: 10px; }
        .status-success { background: #d4edda !important; color: #155724 !important; }
        .status-error { background: #f8d7da !important; color: #721c24 !important; }

    </style>
</head>
<body>

    <header>
        <h1>🤖 Local RAG</h1>
        <button class="rebuild-btn" onclick="openModal()">Rebuild Index</button>
    </header>

    <div id="status-bar"></div>

    <div id="chat-container">
        <div id="messages"></div>
        <div class="thinking" id="thinkingIndicator">
            <span>Thinking</span><span class="dots">...</span>
        </div>
        <div id="input-area">
            <input type="text" id="queryInput" placeholder="Ask a question..." autofocus>
            <button id="sendBtn" onclick="sendMessage()">Send</button>
        </div>
    </div>

    <!-- Rebuild Modal -->
    <div id="rebuildModal" class="modal">
        <div class="modal-content">
            <h3>Rebuild Index</h3>
            <p style="font-size: 0.9rem; color: #666;">Enter a new folder path or leave empty to use current.</p>
            <input type="text" id="folderPath" placeholder="/path/to/documents">
            <div class="modal-actions">
                <button class="close-btn" onclick="closeModal()">Cancel</button>
                <button onclick="startRebuild()" style="background: #007bff; color: white;">Start Rebuild</button>
            </div>
        </div>
    </div>

    <script>
        const input = document.getElementById('queryInput');
        const messagesDiv = document.getElementById('messages');
        const sendBtn = document.getElementById('sendBtn');
        const thinkingIndicator = document.getElementById('thinkingIndicator');
        const modal = document.getElementById('rebuildModal');
        const statusBar = document.getElementById('status-bar');

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
            messagesDiv.scrollTop = messagesDiv.scrollHeight;

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
                messagesDiv.scrollTop = messagesDiv.scrollHeight;
            }
        }

        function appendMessage(text, sender, sources = []) {
            const msgDiv = document.createElement('div');
            msgDiv.className = `message ${sender}-msg`;
            
            let content = `<span>${text.replace(/\\n/g, '<br>')}</span>`;
            
            if (sources && sources.length > 0) {
                const uniqueSources = [...new Set(sources)];
                content += `<div class="sources">Sources: ${uniqueSources.join(', ')}</div>`;
            }
            
            msgDiv.innerHTML = content;
            messagesDiv.appendChild(msgDiv);
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
        }

        // --- Rebuild Logic ---

        function openModal() { modal.style.display = 'flex'; }
        function closeModal() { modal.style.display = 'none'; }

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
                        showStatus(data.message + " <span class='dots'>...</span>");
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
            statusBar.innerHTML = msg;
        }
        
        // Check status on load in case a rebuild is running
        pollStatus();

    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def get_ui():
    return HTML_CONTENT