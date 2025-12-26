import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List

from rag.config import load_config
from rag.index import load_faiss_index
from rag.utils import get_client
from rag.retrieve import retrieve_context
from rag.chat import generate_response

app = FastAPI(title="Local RAG UI")

# Global State
class State:
    config = None
    index = None
    metadata = None
    client = None

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
        return
        
    state.client = get_client(state.config)
    print("RAG System Loaded Successfully.")

# Data Models
class ChatRequest(BaseModel):
    query: str

class ChatResponse(BaseModel):
    answer: str
    sources: List[str]

# Frontend HTML (Embedded for simplicity)
HTML_CONTENT = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Local RAG Chat</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; background: #f4f4f9; }
        #chat-container { background: white; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); overflow: hidden; display: flex; flex-direction: column; height: 80vh; }
        #messages { flex: 1; overflow-y: auto; padding: 20px; }
        .message { margin-bottom: 15px; line-height: 1.5; }
        .user-msg { text-align: right; color: #fff; }
        .user-msg span { background: #007bff; padding: 8px 12px; border-radius: 15px 15px 0 15px; display: inline-block; }
        .bot-msg { text-align: left; color: #333; }
        .bot-msg span { background: #e9ecef; padding: 8px 12px; border-radius: 15px 15px 15px 0; display: inline-block; }
        .sources { font-size: 0.8em; color: #666; margin-top: 5px; font-style: italic; }
        #input-area { border-top: 1px solid #ddd; padding: 20px; display: flex; gap: 10px; background: #fff; }
        input { flex: 1; padding: 10px; border: 1px solid #ddd; border-radius: 4px; font-size: 16px; }
        button { padding: 10px 20px; background: #28a745; color: white; border: none; border-radius: 4px; cursor: pointer; font-size: 16px; }
        button:hover { background: #218838; }
        button:disabled { background: #ccc; }
        .error { color: red; text-align: center; margin-top: 10px; }
    </style>
</head>
<body>
    <h2 style="text-align: center;">Local RAG Chat</h2>
    <div id="chat-container">
        <div id="messages"></div>
        <div id="input-area">
            <input type="text" id="queryInput" placeholder="Ask a question about your documents..." autofocus>
            <button id="sendBtn" onclick="sendMessage()">Send</button>
        </div>
    </div>

    <script>
        const input = document.getElementById('queryInput');
        const messagesDiv = document.getElementById('messages');
        const sendBtn = document.getElementById('sendBtn');

        input.addEventListener('keypress', function (e) {
            if (e.key === 'Enter') sendMessage();
        });

        async function sendMessage() {
            const query = input.value.trim();
            if (!query) return;

            // Add User Message
            appendMessage(query, 'user');
            input.value = '';
            input.disabled = true;
            sendBtn.disabled = true;

            try {
                const response = await fetch('/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ query: query })
                });

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
                input.focus();
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
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def get_ui():
    if not state.config or not state.index:
        return HTMLResponse("<h1>Error: System not initialized. Run './rag-cli init <folder>' first.</h1>", status_code=500)
    return HTML_CONTENT

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    if not state.client or not state.index:
        raise HTTPException(status_code=500, detail="RAG system not initialized")

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
