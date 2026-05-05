"""RAG chatbot — ask about music, get recommendations with context."""

from google import genai

from config import GEMINI_API_KEY
from recommend import search_tracks


def chat(query: str) -> str:
    """Process a music query and return a recommendation response."""
    # Step 1: Search Pinecone with the user's natural language query
    recs = search_tracks(query, top_k=5)

    # Step 2: Generate response with Gemini using retrieved context
    context = "\n".join(
        f"- {r['name']} by {r['artist']} (album: {r['album']}, match score: {r['score']:.2f})"
        for r in recs
    )

    client = genai.Client(api_key=GEMINI_API_KEY)
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=query,
        config={
            "system_instruction": f"""You are a music recommendation assistant. Based on the user's request,
here are matching tracks from our database:

{context}

Recommend these tracks naturally. Explain why they match the user's mood/request.
Keep it conversational and brief. If no tracks found, say so honestly.""",
            "max_output_tokens": 500,
        },
    )

    return response.text


if __name__ == "__main__":
    from rich.console import Console
    from rich.markdown import Markdown

    console = Console()
    console.print("\n[bold]🎵 Music Recommendation Chatbot[/bold]")
    console.print("Ask me for music recommendations! Type 'quit' to exit.\n")

    while True:
        query = input("You: ").strip()
        if query.lower() in ("quit", "exit", "q"):
            break
        if not query:
            continue

        console.print("\n[dim]Thinking...[/dim]")
        response = chat(query)
        console.print(Markdown(response))
        console.print()
