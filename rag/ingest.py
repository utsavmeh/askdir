import os
from typing import List, Dict, Set
from pypdf import PdfReader
from rich.console import Console

console = Console()

def load_files(folder_path: str, ignore_dirs: List[str]) -> List[Dict[str, str]]:
    """Recursively loads .txt, .md, and .pdf files from the directory."""
    documents = []
    ignore_set = set(ignore_dirs)
    
    # Verify path
    if not os.path.isdir(folder_path):
        raise ValueError(f"Path '{folder_path}' is not a directory.")

    console.print(f"[bold blue]Scanning '{folder_path}'...[/bold blue]")

    for root, dirs, files in os.walk(folder_path):
        # Filter out ignored directories in-place
        dirs[:] = [d for d in dirs if d not in ignore_set]

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

            except Exception as e:
                console.print(f"[yellow]Warning:[/yellow] Could not load {file}: {e}")

    console.print(f"[green]Found {len(documents)} supported documents.[/green]")
    return documents
