"""Unified chat router — detects intent and routes to the right feature."""

import json

import spotipy

from gemini_client import generate
from recommend import search_tracks
from taste import get_taste_profile, taste_to_text, get_all_taste_artists
from features import track_dna, mood_timeline, music_debate


def detect_intent(message: str, has_spotify: bool) -> dict:
    """Use Gemini to detect what the user wants."""
    result = generate(
        prompt=message,
        system=f"""Classify this music-related message into ONE intent. Return ONLY a JSON object.

Available intents:
- "search" — user wants music recommendations or is describing a vibe/mood
- "track_dna" — user wants to understand WHY a specific track/artist sounds the way it does
- "roast" — user wants their music taste roasted (keywords: roast, judge, critique my taste)
- "mood_timeline" — user describes a day/schedule and wants a playlist for it (keywords: morning, day, schedule, timeline, but WITH a description of the day)
- "auto_timeline" — user wants a personalized daily playlist based on their listening history, WITHOUT describing a specific day (keywords: my day, playlist for me, daily playlist, make me a timeline, what should I listen to today)
- "matchmaker" — user wants to compare taste with someone (keywords: friend likes, compare, match, compatibility)
- "debate" — user wants a debate about a track/artist (keywords: debate, argue, defend, overrated)
- "taste" — user asks about their own taste or wants it explained (keywords: my taste, what do I like, analyze me)
- "discover" — user wants to discover new music based on their taste, expand their library (keywords: discover, find new music, expand, something new, surprise me, explore)
- "chat" — general music chat, questions, or anything that doesn't fit above

Return: {{"intent": "...", "query": "the relevant search/topic query", "artists": ["if matchmaker, friend's artists"]}}
Only return the JSON.""",
        max_tokens=200,
    )

    try:
        clean = result.strip()
        if clean.startswith("```"):
            clean = clean.split("\n", 1)[1].rsplit("```", 1)[0]
        return json.loads(clean)
    except (json.JSONDecodeError, IndexError):
        return {"intent": "search", "query": message}


def chat_with_context(
    message: str,
    taste_text: str = "",
    sp: spotipy.Spotify = None,
    history: list[dict] = None,
) -> dict:
    """Process a chat message with taste context and return response + tracks."""
    has_spotify = sp is not None
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

    # Build conversation context from history
    history_context = ""
    if history:
        recent = history[-6:]  # last 3 exchanges
        history_context = "\nRecent conversation:\n" + "\n".join(
            f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content'][:200]}"
            for m in recent
        )

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
            # Try to extract artists from the message
            friend_artists = [a.strip() for a in query.split(",") if a.strip()]
        if friend_artists:
            from features import music_matchmaker
            response = music_matchmaker(friend_artists, sp=sp)
        else:
            response = ("**Two ways to match:**\n\n"
                "1. **Type artists:** *'compare me with someone who likes Radiohead, Tame Impala'*\n"
                "2. **Share a link:** Click the 💘 button below to generate a link — "
                "send it to your friend, they connect their Spotify, and I'll compare your real listening data!\n\n"
                "Which do you prefer?")

    elif intent == "debate":
        response = music_debate(query)

    elif intent == "discover" and has_spotify:
        # Index new discovery tracks, then search for the best ones
        from indexer import index_discovery
        index_discovery(sp=sp)
        # Now search with their taste as query
        tracks = search_tracks(taste_context or message, top_k=10)
        tracks_text = "\n".join(
            f"- {t['name']} by {t['artist']} ({', '.join(t.get('genres', [])[:2])})"
            for t in tracks
        )
        response = generate(
            prompt=f"User asked: {message}",
            system=f"""You are a music discovery assistant. You just expanded the user's catalog
with NEW music based on their taste. Here are fresh discoveries:

{tracks_text}

{taste_context}

Present these as exciting discoveries. For each track:
1. Why it connects to their existing taste
2. What's NEW and different about it — what it adds to their palette
3. A "listen if you like X" comparison to an artist they already know

Make it feel like a curated discovery, not a list. Be enthusiastic but specific.""",
        )

    else:
        # General chat with taste context
        tracks = search_tracks(message, top_k=3)
        tracks_text = ""
        if tracks:
            tracks_text = "\nRelevant tracks from our database:\n" + "\n".join(
                f"- {t['name']} by {t['artist']}" for t in tracks
            )

        response = generate(
            prompt=message,
            system=f"""You are a knowledgeable music assistant. You love discussing music
in all its forms — history, production, culture, recommendations.
{taste_context}
{history_context}
{tracks_text}

Be conversational, insightful, and personalized based on the user's taste when relevant.
If the user needs Spotify login for a feature, mention they can connect Spotify.""",
        )

    # If intent needed Spotify but user isn't logged in
    if intent in ("roast", "taste", "matchmaker", "auto_timeline", "discover") and not has_spotify:
        response = f"I'd love to {intent.replace('_', ' ')} for you, but I need access to your Spotify data first. Connect your Spotify account using the button above!"
        tracks = []

    return {
        "response": response,
        "tracks": tracks,
        "intent": intent,
    }
