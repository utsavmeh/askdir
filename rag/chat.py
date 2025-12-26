from rich.console import Console
from rich.markdown import Markdown
from rag.retrieve import retrieve_context

console = Console()

def generate_response(client, model_name: str, context_text: str, query: str) -> str:
    """Generates an answer using the LLM based on context."""
    # Strict Prompt
    system_prompt = (
        "You are a helpful assistant. Use the following context to answer the user's question. "
        "If the answer is not contained in the context, say 'I don't know'."
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
