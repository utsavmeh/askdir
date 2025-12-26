import typer
import os
from typing import Optional
from rich.console import Console

from rag.config import RAGConfig, load_config, save_config
from rag.ingest import load_files
from rag.chunking import chunk_text
from rag.index import build_index, load_faiss_index
from rag.chat import start_chat
from rag.utils import get_client

app = typer.Typer(help="Local RAG CLI Tool")
console = Console()

@app.command()
def init(folder_path: str):
    """
    Initialize a new RAG index from a folder.
    """
    if os.path.exists("rag.yaml"):
        console.print("[yellow]Configuration file already exists. Use 'rag rebuild' to update or delete 'rag.yaml' to start over.[/yellow]")
        return

    # Create default config
    config = RAGConfig(folder_path=folder_path)
    save_config(config)
    console.print(f"[green]Initialized config for '{folder_path}'[/green]")
    
    # Run ingestion
    _build_pipeline(config)

@app.command()
def chat():
    """
    Start the chat interface.
    """
    config = load_config()
    if not config:
        console.print("[red]No configuration found. Run 'rag init <folder>' first.[/red]")
        raise typer.Exit(code=1)

    index, metadata = load_faiss_index()
    if not index:
        console.print("[red]No index found. Run 'rag rebuild' to create it.[/red]")
        raise typer.Exit(code=1)
        
    client = get_client(config)
    start_chat(client, config, index, metadata)

@app.command()
def ui(port: int = 8000):
    """
    Start the Web UI.
    """
    try:
        import uvicorn
        from fastapi import FastAPI
    except ImportError:
        console.print("[red]Error: 'fastapi' and 'uvicorn' are required for the UI.[/red]")
        console.print("Please install them: [bold]pip install fastapi uvicorn[/bold]")
        raise typer.Exit(code=1)
        
    config = load_config()
    if not config:
        console.print("[red]No configuration found. Run 'rag init <folder>' first.[/red]")
        raise typer.Exit(code=1)

    console.print(f"[green]Starting Web UI at http://127.0.0.1:{port}[/green]")
    console.print("Press Ctrl+C to stop.")
    
    uvicorn.run("rag.server:app", host="127.0.0.1", port=port, reload=False)

@app.command()
def rebuild():
    """
    Rebuild the index using the current configuration.
    """
    config = load_config()
    if not config:
        console.print("[red]No configuration found. Run 'rag init <folder>' first.[/red]")
        raise typer.Exit(code=1)
        
    _build_pipeline(config)

def _build_pipeline(config: RAGConfig):
    """Core logic to ingest, chunk, and index."""
    console.print("[bold]Starting ingestion pipeline...[/bold]")
    
    # 1. Load
    docs = load_files(config.folder_path, config.ignore_dirs)
    if not docs:
        console.print("[red]No valid documents found. Aborting.[/red]")
        return

    # 2. Chunk
    chunks = chunk_text(docs, config.chunk_size, config.overlap)
    console.print(f"Created {len(chunks)} chunks.")

    # 3. Index
    client = get_client(config)
    build_index(client, chunks, config)
    console.print("[bold green]Index successfully built![/bold green]")

if __name__ == "__main__":
    app()
