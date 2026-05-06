"""Unified chat router — slash commands + AI intent detection + free chat."""

import json
import re

import spotipy

from gemini_client import generate
from recommend import search_tracks
from taste import get_taste_profile, taste_to_text, get_all_taste_artists
from features import track_dna, mood_timeline, music_debate


# ─── Slash command parser ────────────────────────────────────────────

COMMANDS = {
    "/search": "search",
    "/recommend": "search",
    "/rec": "search",
    "/dna": "track_dna",
    "/trackdna": "track_dna",
    "/roast": "roast",
    "/taste": "taste",
    "/timeline": "mood_timeline",
    "/mytimeline": "auto_timeline",
    "/myday": "auto_timeline",
    "/matchmaker": "matchmaker",
    "/match": "matchmaker",
    "/debate": "debate",
    "/discover": "discover",
    "/explore": "discover",
}


def parse_slash_command(message: str) -> tuple[str | None, str]:
    """Parse /command from message. Returns (intent, remaining_text) or (None, message)."""
    stripped = message.strip()
    if not stripped.startswith("/"):
        return None, message

    parts = stripped.split(None, 1)
    cmd = parts[0].lower()
    args = parts[1] if len(parts) > 1 else ""

    if cmd in COMMANDS:
        return COMMANDS[cmd], args

    return None, message


# ─── AI intent detection (only for non-slash messages) ───────────────

def detect_intent(message: str, has_spotify: bool) -> dict:
    """Use Gemini to detect intent — only called when no slash command."""
    result = generate(
        prompt=message,
        system=f"""Classify this music-related message. Return ONLY a JSON object.

Intents (pick the BEST match, default to "chat" for general conversation):
- "search" — user explicitly asks for recommendations or describes a vibe they want music for
- "track_dna" — user asks WHY a specific track/artist sounds the way it does
- "roast" — user wants their music taste roasted
- "mood_timeline" — user describes a day/schedule for a playlist (WITH day description)
- "auto_timeline" — user wants a daily playlist from their history (NO day description)
- "matchmaker" — user wants to compare taste with someone
- "debate" — user wants a debate about a track/artist
- "taste" — user asks about their own listening taste
- "discover" — user wants to find new music based on their taste
- "chat" — general music discussion, questions, opinions, history, theory, anything conversational

IMPORTANT: Default to "chat" for general questions like "what do you think about X",
"tell me about X", "who influenced X", "is X good". Only use specific intents when
the user clearly wants that feature.

Return: {{"intent": "...", "query": "the relevant topic/query", "artists": ["if matchmaker"]}}
Only return the JSON.""",
        max_tokens=200,
    )

    try:
        clean = result.strip()
        if clean.startswith("```"):
            clean = clean.split("\n", 1)[1].rsplit("```", 1)[0]
        return json.loads(clean)
    except (json.JSONDecodeError, IndexError):
        return {"intent": "chat", "query": message}


# ─── Main router ────────────────────────────────────────────────────

def chat_with_context(
    message: str,
    taste_text: str = "",
    sp: spotipy.Spotify = None,
    history: list[dict] = None,
) -> dict:
    """Process a chat message with taste context and return response + tracks."""
    has_spotify = sp is not None

    # Step 1: Check for slash command
    slash_intent, slash_args = parse_slash_command(message)

    if slash_intent:
        intent = slash_intent
        query = slash_args or message
        intent_data = {"intent": intent, "query": query, "artists": []}
        # Parse artists for matchmaker: /match artist1, artist2
        if intent == "matchmaker" and slash_args:
            intent_data["artists"] = [a.strip() for a in slash_args.split(",") if a.strip()]
    else:
        # Step 2: AI intent detection for natural messages
        intent_data = detect_intent(message, has_spotify)
        intent = intent_data.get("intent", "chat")
        query = intent_data.get("query", message)

    tracks = []
    response = ""

    taste_context = ""
    if taste_text:
        taste_context = f"""
USER'S MUSIC TASTE PROFILE (use this to personalize your response):
{taste_text}
"""

    history_context = ""
    if history:
        recent = history[-6:]
        history_context = "\nRecent conversation:\n" + "\n".join(
            f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content'][:200]}"
            for m in recent
        )

    # ─── Route to feature ────────────────────────────────────────

    if intent == "search":
        tracks = search_tracks(query, top_k=5)
        tracks_text = "\n".join(
            f"- {t['name']} by {t['artist']} (score: {t['score']:.2f})"
            for t in tracks
        )
        response = generate(
            prompt=message,
            system=f"""You are a music recommendation assistant with deep knowledge.
{taste_context}
{history_context}

Matching tracks from our database:
{tracks_text}

Recommend tracks naturally. If the user has a taste profile, explain how these
relate to what they already like. Be conversational and brief.""",
        )

    elif intent == "track_dna":
        response = track_dna(query)

    elif intent == "roast" and has_spotify:
        from features import playlist_roast
        response = playlist_roast(sp=sp)

    elif intent == "taste" and has_spotify:
        from features import explain_taste
        response = explain_taste(sp=sp)

    elif intent == "mood_timeline":
        response = mood_timeline(query)

    elif intent == "auto_timeline" and has_spotify:
        from features import auto_timeline
        response = auto_timeline(sp=sp)

    elif intent == "matchmaker" and has_spotify:
        friend_artists = intent_data.get("artists", [])
        if not friend_artists:
            friend_artists = [a.strip() for a in query.split(",") if a.strip()]
        if friend_artists:
            from features import music_matchmaker
            response = music_matchmaker(friend_artists, sp=sp)
        else:
            response = ("**Two ways to match:**\n\n"
                "1. **Type artists:** `/match radiohead, tame impala, daft punk`\n"
                "2. **Share a link:** Click the 💘 button below — "
                "send it to your friend, they connect Spotify, and I compare your real data!\n\n"
                "Which do you prefer?")

    elif intent == "debate":
        response = music_debate(query)

    elif intent == "discover" and has_spotify:
        from indexer import index_discovery
        index_discovery(sp=sp)
        tracks = search_tracks(taste_context or message, top_k=10)
        tracks_text = "\n".join(
            f"- {t['name']} by {t['artist']} ({', '.join(t.get('genres', [])[:2])})"
            for t in tracks
        )
        response = generate(
            prompt=f"User asked: {message}",
            system=f"""You are a music discovery assistant. Fresh discoveries:

{tracks_text}

{taste_context}

Present these as exciting discoveries. For each track:
1. Why it connects to their existing taste
2. What's NEW and different about it
3. A "listen if you like X" comparison

Be enthusiastic but specific.""",
        )

    else:
        # General music chat — conversational, knowledgeable
        # Optionally pull relevant tracks for context
        tracks = search_tracks(message, top_k=3) if len(message) > 10 else []
        tracks_text = ""
        if tracks:
            tracks_text = "\nRelevant tracks from our database:\n" + "\n".join(
                f"- {t['name']} by {t['artist']}" for t in tracks
            )

        response = generate(
            prompt=message,
            system=f"""You are a passionate, knowledgeable music companion. You can discuss:
- Music history, genres, movements, and cultural impact
- Production techniques, music theory, songwriting
- Artist stories, album deep dives, discographies
- Opinions, hot takes, and music debates
- Recommendations woven naturally into conversation
{taste_context}
{history_context}
{tracks_text}

Be conversational, opinionated, and engaging. Share interesting facts and connections.
Personalize based on the user's taste when relevant.
Keep responses focused and concise — no walls of text.

If the user might benefit from a specific feature, mention the slash command:
/search, /dna, /roast, /taste, /timeline, /myday, /match, /debate, /discover""",
        )

    # If intent needed Spotify but user isn't logged in
    if intent in ("roast", "taste", "matchmaker", "auto_timeline", "discover") and not has_spotify:
        response = f"I'd love to do that, but I need your Spotify data. Connect your account using the button above!"
        tracks = []

    return {
        "response": response,
        "tracks": tracks,
        "intent": intent,
    }
