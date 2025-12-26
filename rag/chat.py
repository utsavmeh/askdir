from rich.console import Console
from rich.markdown import Markdown
from rag.retrieve import retrieve_context

console = Console()

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
                
                # Strict Prompt
                system_prompt = (
                    "You are a helpful assistant. Use the following context to answer the user's question. "
                    "If the answer is not contained in the context, say 'I don't know'."
                )
                
                user_message = f"Context:\n{context_text}\n\nQuestion: {query}"
                
                # Generate Answer
                response = client.chat.completions.create(
                    model=config.chat_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message}
                    ],
                    temperature=0.0
                )
                
                answer = response.choices[0].message.content
            
            console.print("[bold purple]Bot:[/bold purple]")
            console.print(Markdown(answer))
            console.print()
            
        except KeyboardInterrupt:
            console.print("\nExiting...")
            break
        except Exception as e:
            console.print(f"[red]Error:[/red] {e}")
