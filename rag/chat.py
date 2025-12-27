from rich.console import Console
from rich.markdown import Markdown
from rag.retrieve import retrieve_context

console = Console()

def generate_response(client, model_name: str, context_text: str, query: str) -> str:
    """Generates an answer using the LLM based on context."""
    system_prompt = (
        "You are a strict assistant that answers questions based ONLY on the provided context.\n"
        "Rules:\n"
        "1. If the user greets you (e.g., 'Hi'), respond politely.\n"
        "2. Answer the question using ONLY the text in the 'Context' section below.\n"
        "3. Do NOT use your own internal knowledge or facts from the internet.\n"
        "4. If the answer to the question is not present in the 'Context' section, you MUST respond with exactly: 'I don't know'."
    )
    
    user_message = f"Context:\n{context_text}\n\nQuestion: {query}"
    
    # Generate Answer
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ],
        temperature=0.0
    )
    
    return response.choices[0].message.content

def start_chat(client, config, index, metadata):
    """Runs the REPL chat loop."""
    console.print("\n[bold green]--- RAG Chatbot Ready ---[/bold green]")
    console.print(f"Model: {config.chat_model}")
    console.print("Type 'exit' to quit.\n")
    
    while True:
        try:
            query = console.input("[bold blue]You:[/bold blue] ").strip()
            if not query:
                continue
            if query.lower() in ["exit", "quit"]:
                break
                
            # Retrieve
            with console.status("Thinking..."):
                matches = retrieve_context(client, query, index, metadata, config.embedding_model)
                
                # Build Context
                context_text = "\n\n".join(
                    [f"Source: {m['source']}\nContent: {m['text']}" for m in matches]
                )
                
                answer = generate_response(client, config.chat_model, context_text, query)
            
            console.print("[bold purple]Bot:[/bold purple]")
            console.print(Markdown(answer))
            console.print()
            
        except KeyboardInterrupt:
            console.print("\nExiting...")
            break
        except Exception as e:
            console.print(f"[red]Error:[/red] {e}")
