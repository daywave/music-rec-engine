"""Build taste profiles from Spotify listening data."""

import spotipy
from spotify_client import get_user_top_tracks, get_user_saved_tracks


def get_taste_profile(limit: int = 10, sp: spotipy.Spotify = None) -> dict:
    """Build a taste profile from user's Spotify data."""
    top_short = get_user_top_tracks("short_term", limit=limit, sp=sp)
    top_medium = get_user_top_tracks("medium_term", limit=limit, sp=sp)
    top_long = get_user_top_tracks("long_term", limit=limit, sp=sp)
    saved = get_user_saved_tracks(limit=limit, sp=sp)

    return {
        "recent_favorites": top_short,
        "medium_term": top_medium,
        "all_time": top_long,
        "saved": saved,
    }


def taste_to_text(profile: dict) -> str:
    """Convert taste profile to readable text for AI prompts."""
    sections = []

    if profile["recent_favorites"]:
        tracks = ", ".join(
            f"{t['name']} by {t['artist']}" for t in profile["recent_favorites"]
        )
        sections.append(f"Recently obsessed with: {tracks}")

    if profile["medium_term"]:
        tracks = ", ".join(
            f"{t['name']} by {t['artist']}" for t in profile["medium_term"]
        )
        sections.append(f"Regular rotation (last 6 months): {tracks}")

    if profile["all_time"]:
        tracks = ", ".join(
            f"{t['name']} by {t['artist']}" for t in profile["all_time"]
        )
        sections.append(f"All-time favorites: {tracks}")

    if profile["saved"]:
        tracks = ", ".join(
            f"{t['name']} by {t['artist']}" for t in profile["saved"]
        )
        sections.append(f"Liked songs: {tracks}")

    return "\n".join(sections)


def get_all_taste_artists(profile: dict) -> list[str]:
    """Extract unique artist names from taste profile."""
    artists = set()
    for key in profile:
        for track in profile[key]:
            artists.add(track["artist"])
    return sorted(artists)
