"""Fetch tracks and audio features from Spotify."""

import os
import spotipy
from spotipy.oauth2 import SpotifyOAuth

from config import SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET

SCOPES = " ".join([
    "playlist-read-private",
    "playlist-read-collaborative",
    "user-top-read",
    "user-library-read",
])

REDIRECT_URI = os.getenv("SPOTIFY_REDIRECT_URI", "http://127.0.0.1:8888/callback")


def get_auth_manager(cache_path: str = ".cache") -> SpotifyOAuth:
    """Get a SpotifyOAuth manager with a specific cache path."""
    return SpotifyOAuth(
        client_id=SPOTIFY_CLIENT_ID,
        client_secret=SPOTIFY_CLIENT_SECRET,
        redirect_uri=REDIRECT_URI,
        scope=SCOPES,
        cache_path=cache_path,
    )


def get_spotify_client(cache_path: str = ".cache") -> spotipy.Spotify:
    """Get Spotify client for CLI usage (local cache)."""
    return spotipy.Spotify(auth_manager=get_auth_manager(cache_path))


def get_spotify_client_with_token(token_info: dict) -> spotipy.Spotify:
    """Get Spotify client using a token dict (for web sessions)."""
    return spotipy.Spotify(auth=token_info["access_token"])


def get_playlist_tracks(playlist_id: str, sp: spotipy.Spotify = None) -> list[dict]:
    """Pull tracks from a Spotify playlist."""
    if sp is None:
        sp = get_spotify_client()
    results = sp.playlist_tracks(playlist_id, limit=10)
    tracks = []

    for item in results["items"]:
        track = item.get("track")
        if not track or not track.get("id"):
            continue
        tracks.append({
            "id": track["id"],
            "name": track["name"],
            "artist": ", ".join(a["name"] for a in track["artists"]),
            "album": track["album"].get("name", "Unknown"),
            "popularity": track.get("popularity", 0),
        })

    return tracks


def search_tracks(query: str, limit: int = 10) -> list[dict]:
    """Search Spotify for tracks by query."""
    sp = get_spotify_client()
    results = sp.search(q=query, type="track", limit=limit)
    tracks = []

    for track in results["tracks"]["items"]:
        tracks.append({
            "id": track["id"],
            "name": track["name"],
            "artist": ", ".join(a["name"] for a in track["artists"]),
            "album": track["album"].get("name", "Unknown"),
            "popularity": track.get("popularity", 0),
        })

    return tracks


def search_tracks_with_features(queries: list[str], limit: int = 10) -> list[dict]:
    """Search multiple queries and return track metadata."""
    sp = get_spotify_client()
    all_tracks = []
    seen_ids = set()

    for query in queries:
        results = sp.search(q=query, type="track", limit=limit)
        for track in results["tracks"]["items"]:
            if track["id"] in seen_ids:
                continue
            seen_ids.add(track["id"])
            all_tracks.append({
                "id": track["id"],
                "name": track["name"],
                "artist": ", ".join(a["name"] for a in track["artists"]),
                "album": track["album"].get("name", "Unknown"),
                "popularity": track.get("popularity", 0),
            })

    return all_tracks


def get_user_top_tracks(time_range: str = "medium_term", limit: int = 10, sp: spotipy.Spotify = None) -> list[dict]:
    """Get user's top tracks. time_range: short_term, medium_term, long_term."""
    if sp is None:
        sp = get_spotify_client()
    results = sp.current_user_top_tracks(limit=limit, time_range=time_range)
    tracks = []
    for track in results["items"]:
        tracks.append({
            "id": track["id"],
            "name": track["name"],
            "artist": ", ".join(a["name"] for a in track["artists"]),
            "album": track["album"].get("name", "Unknown"),
            "popularity": track.get("popularity", 0),
        })
    return tracks


def get_user_saved_tracks(limit: int = 10, sp: spotipy.Spotify = None) -> list[dict]:
    """Get user's liked/saved tracks."""
    if sp is None:
        sp = get_spotify_client()
    results = sp.current_user_saved_tracks(limit=limit)
    tracks = []
    for item in results["items"]:
        track = item["track"]
        tracks.append({
            "id": track["id"],
            "name": track["name"],
            "artist": ", ".join(a["name"] for a in track["artists"]),
            "album": track["album"].get("name", "Unknown"),
            "popularity": track.get("popularity", 0),
        })
    return tracks
