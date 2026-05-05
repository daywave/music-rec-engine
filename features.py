"""All music recommendation features."""

import json

import spotipy

from gemini_client import generate
from recommend import search_tracks
from taste import get_taste_profile, taste_to_text, get_all_taste_artists


# ─── Feature 1: Explain My Taste ────────────────────────────────────

def explain_taste(sp: spotipy.Spotify = None) -> str:
    """Analyze user's Spotify listening and explain their taste personality."""
    profile = get_taste_profile(limit=10, sp=sp)
    taste_text = taste_to_text(profile)

    return generate(
        prompt=taste_text,
        system="""You are a music psychologist and critic. Analyze this person's listening data
and write a detailed, insightful taste profile. Cover:

1. Their core sonic identity — what ties their taste together
2. Emotional patterns — what moods/feelings they gravitate toward
3. How their taste has evolved (compare recent vs all-time)
4. Surprising contradictions or range in their taste
5. What their music says about their personality

Be specific about the music. Reference actual tracks and artists from their data.
Write in second person ("you"). Be insightful, not generic. Mix analytical with playful.""",
        max_tokens=1500,
    )


# ─── Feature 2: Track DNA ───────────────────────────────────────────

def track_dna(track_query: str) -> str:
    """Explain why a track sounds the way it does and suggest explorations."""
    similar = search_tracks(track_query, top_k=5)

    similar_text = "\n".join(
        f"- {r['name']} by {r['artist']} (similarity: {r['score']:.2f})"
        for r in similar
    )

    return generate(
        prompt=f"Analyze this track: {track_query}",
        system=f"""You are a music producer and musicologist. The user wants to understand
a track's DNA — what makes it sound the way it does.

Similar tracks from our database:
{similar_text}

Cover:
1. What makes this track sonically distinctive (production, rhythm, texture, mood)
2. The musical lineage — what genres/movements it draws from
3. Why these similar tracks share DNA with it
4. A "if you like this, explore next" path — 3 directions to go deeper,
   each progressively more adventurous

Be specific and technical but accessible. Use vivid descriptions of sound.""",
        max_tokens=1200,
    )


# ─── Feature 3: Playlist Roast ──────────────────────────────────────

def playlist_roast(sp: spotipy.Spotify = None) -> str:
    """Roast the user's music taste based on their listening data."""
    profile = get_taste_profile(limit=10, sp=sp)
    taste_text = taste_to_text(profile)
    artists = get_all_taste_artists(profile)

    return generate(
        prompt=taste_text,
        system=f"""You are a brutally funny music critic roasting someone's Spotify.
Their top artists include: {', '.join(artists)}

Rules:
- Be genuinely funny, not mean-spirited. Think comedy roast, not bullying
- Make specific jokes about the actual artists and tracks, not generic insults
- Point out funny patterns (e.g., "90% sad music, 10% gym bangers")
- Include at least one backhanded compliment
- End with a genuine recommendation to "save" their taste — suggest 2-3 tracks
  that would add range to their library
- Keep it under 400 words

Reference specific tracks and artists from their data. Generic roasts are boring.""",
        max_tokens=800,
    )


# ─── Feature 4: Mood Timeline ───────────────────────────────────────

def mood_timeline(day_description: str) -> str:
    """Generate a playlist that evolves through the user's described day."""
    segments_raw = generate(
        prompt=day_description,
        system="""Break this day description into 3-5 mood segments for a playlist.
Return ONLY a JSON array of objects with "time" and "mood" fields.
Example: [{"time": "morning", "mood": "calm waking up coffee"}, {"time": "commute", "mood": "energetic driving"}]
Return only the JSON, no other text.""",
        max_tokens=300,
    )

    try:
        clean = segments_raw.strip()
        if clean.startswith("```"):
            clean = clean.split("\n", 1)[1].rsplit("```", 1)[0]
        segments = json.loads(clean)
    except (json.JSONDecodeError, IndexError):
        segments = [
            {"time": "start", "mood": "easy going"},
            {"time": "middle", "mood": day_description},
            {"time": "end", "mood": "wind down"},
        ]

    all_results = []
    for seg in segments:
        results = search_tracks(seg["mood"], top_k=3)
        all_results.append({
            "time": seg["time"],
            "mood": seg["mood"],
            "tracks": results,
        })

    playlist_data = ""
    for seg in all_results:
        tracks = ", ".join(
            f"{t['name']} by {t['artist']}" for t in seg["tracks"]
        )
        playlist_data += f"\n[{seg['time'].upper()}] Mood: {seg['mood']}\nTracks: {tracks}\n"

    return generate(
        prompt=f"Day description: {day_description}\n\nPlaylist segments:{playlist_data}",
        system="""You are a DJ and playlist curator. The user described their day and
we've matched tracks to each segment.

Create a cohesive playlist narrative:
1. Present each time segment with its tracks
2. Explain WHY each track fits that moment
3. Describe how the playlist flows and transitions between segments
4. Add a creative playlist title

Make it feel like a curated experience, not a random list.""",
        max_tokens=1200,
    )


# ─── Feature 5: Music Matchmaker ────────────────────────────────────

def music_matchmaker(friend_artists: list[str], sp: spotipy.Spotify = None) -> str:
    """Find musical common ground between the user and a friend."""
    profile = get_taste_profile(limit=10, sp=sp)
    user_taste = taste_to_text(profile)
    user_artists = get_all_taste_artists(profile)

    combined_query = " ".join(user_artists[:5] + friend_artists[:5])
    overlap_tracks = search_tracks(combined_query, top_k=10)

    overlap_text = "\n".join(
        f"- {t['name']} by {t['artist']} (score: {t['score']:.2f})"
        for t in overlap_tracks
    )

    friend_text = ", ".join(friend_artists)

    return generate(
        prompt=f"""User's taste:
{user_taste}

Friend's favorite artists: {friend_text}

Tracks in the overlap zone:
{overlap_text}""",
        system="""You are a music matchmaker. Two people want to find common ground.

Analyze:
1. Where their tastes overlap (shared vibes, genres, energy)
2. Where they diverge
3. A "bridge playlist" — 5 tracks that both would enjoy, explaining why each works
4. One track each person should steal from the other's taste
5. A fun compatibility score (X/10) with a witty reason

Be specific about the music. Make it fun and social.""",
        max_tokens=1200,
    )


# ─── Feature 6: Music Debate ────────────────────────────────────────

def music_debate(track_query: str) -> str:
    """Two AI critics debate whether a track/artist is good."""
    similar = search_tracks(track_query, top_k=5)
    similar_text = "\n".join(
        f"- {t['name']} by {t['artist']}" for t in similar
    )

    return generate(
        prompt=f"Debate topic: {track_query}",
        system=f"""You are hosting a music debate show called "SOUND COURT" between two critics:

🎤 DEFENDER (passionate fan): Argues why this music is brilliant, important, and underappreciated
🎤 PROSECUTOR (skeptic): Argues why this music is overrated, derivative, or problematic

Similar/related tracks for context:
{similar_text}

Format the debate as:
1. OPENING STATEMENTS (each critic, 2-3 sentences)
2. ROUND 1: Musical merit (production, composition, innovation)
3. ROUND 2: Cultural impact and influence
4. ROUND 3: Hot takes and personal attacks (on taste, not people)
5. VERDICT: A fair judge summarizes both sides and gives a ruling

Rules:
- Both sides must make SPECIFIC musical arguments (not just "it's good/bad")
- Reference actual songs, albums, production techniques
- Keep it entertaining — sharp wit, not academic
- The debate should teach the reader something about the music
- 500-700 words total""",
        max_tokens=1500,
    )
