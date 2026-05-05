"""Music Recommendation Engine — main app with all features."""

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt

from features import (
    explain_taste,
    track_dna,
    playlist_roast,
    mood_timeline,
    music_matchmaker,
    music_debate,
)
from recommend import search_tracks
from gemini_client import generate

console = Console()

MENU = """
[bold cyan]1[/] — Search & Recommend (natural language)
[bold cyan]2[/] — Explain My Taste
[bold cyan]3[/] — Track DNA
[bold cyan]4[/] — Playlist Roast
[bold cyan]5[/] — Mood Timeline
[bold cyan]6[/] — Music Matchmaker
[bold cyan]7[/] — Music Debate
[bold cyan]q[/] — Quit
"""


def show_menu():
    console.print(Panel(MENU, title="[bold]🎵 Music Rec Engine[/bold]", border_style="cyan"))


def handle_search():
    query = Prompt.ask("[cyan]What kind of music?[/]")
    console.print("\n[dim]Searching...[/dim]")
    recs = search_tracks(query, top_k=5)

    if not recs:
        console.print("[yellow]No matches found. Try indexing more music.[/yellow]")
        return

    context = "\n".join(
        f"- {r['name']} by {r['artist']} (score: {r['score']:.2f})"
        for r in recs
    )
    response = generate(
        prompt=query,
        system=f"""You are a music recommendation assistant. Matching tracks:

{context}

Recommend these tracks naturally. Explain why they match. Keep it brief.""",
    )
    console.print(Markdown(response))


def handle_explain_taste():
    console.print("\n[dim]Analyzing your Spotify listening data...[/dim]")
    result = explain_taste()
    console.print(Markdown(result))


def handle_track_dna():
    query = Prompt.ask("[cyan]Which track or artist?[/]")
    console.print("\n[dim]Analyzing track DNA...[/dim]")
    result = track_dna(query)
    console.print(Markdown(result))


def handle_roast():
    console.print("\n[dim]Preparing your roast...[/dim]")
    result = playlist_roast()
    console.print(Markdown(result))


def handle_mood_timeline():
    console.print("[cyan]Describe your day or planned day:[/]")
    console.print("[dim]Example: 'morning coffee, stressful meetings, gym after work, chill evening'[/dim]")
    day = Prompt.ask("[cyan]>[/]")
    console.print("\n[dim]Building your mood timeline...[/dim]")
    result = mood_timeline(day)
    console.print(Markdown(result))


def handle_matchmaker():
    console.print("[cyan]Enter your friend's favorite artists (comma-separated):[/]")
    console.print("[dim]Example: 'radiohead, boards of canada, four tet'[/dim]")
    friend_input = Prompt.ask("[cyan]>[/]")
    friend_artists = [a.strip() for a in friend_input.split(",") if a.strip()]

    if not friend_artists:
        console.print("[yellow]Need at least one artist.[/yellow]")
        return

    console.print("\n[dim]Finding your musical common ground...[/dim]")
    result = music_matchmaker(friend_artists)
    console.print(Markdown(result))


def handle_debate():
    query = Prompt.ask("[cyan]Which track or artist to debate?[/]")
    console.print("\n[dim]Court is now in session...[/dim]")
    result = music_debate(query)
    console.print(Markdown(result))


HANDLERS = {
    "1": handle_search,
    "2": handle_explain_taste,
    "3": handle_track_dna,
    "4": handle_roast,
    "5": handle_mood_timeline,
    "6": handle_matchmaker,
    "7": handle_debate,
}


def main():
    console.print("\n[bold]🎵 Music Recommendation Engine[/bold]")
    console.print("[dim]Powered by Pinecone + Gemini + Spotify[/dim]\n")

    while True:
        show_menu()
        choice = Prompt.ask("[bold]Choose[/]", choices=["1", "2", "3", "4", "5", "6", "7", "q"])

        if choice == "q":
            console.print("[dim]Later! 🎧[/dim]")
            break

        console.print()
        try:
            HANDLERS[choice]()
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")

        console.print()


if __name__ == "__main__":
    main()
