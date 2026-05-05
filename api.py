"""FastAPI backend for Music Recommendation Engine."""

import os
import uuid

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, RedirectResponse
from pydantic import BaseModel

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
from spotify_client import get_auth_manager, get_spotify_client_with_token, REDIRECT_URI

app = FastAPI(title="Music Rec Engine")

# In-memory session store (use Redis in production)
sessions: dict[str, dict] = {}

SPOTIFY_REDIRECT = os.getenv("SPOTIFY_REDIRECT_URI", "http://127.0.0.1:8000/auth/callback")


# ─── Helpers ─────────────────────────────────────────────────────────

def get_session_token(request: Request) -> dict | None:
    """Get Spotify token from session cookie."""
    session_id = request.cookies.get("session_id")
    if not session_id or session_id not in sessions:
        return None
    token_info = sessions[session_id]
    # Refresh if expired
    auth = get_auth_manager(cache_path=f".cache-{session_id}")
    if auth.is_token_expired(token_info):
        token_info = auth.refresh_access_token(token_info["refresh_token"])
        sessions[session_id] = token_info
    return token_info


def require_spotify(request: Request):
    """Get Spotify client from session or raise error."""
    token = get_session_token(request)
    if not token:
        raise HTTPException(status_code=401, detail="Spotify login required")
    return get_spotify_client_with_token(token)


# ─── Auth routes ─────────────────────────────────────────────────────

@app.get("/auth/login")
def auth_login():
    """Redirect to Spotify OAuth."""
    auth = get_auth_manager()
    auth.redirect_uri = SPOTIFY_REDIRECT
    url = auth.get_authorize_url()
    return RedirectResponse(url)


@app.get("/auth/callback")
def auth_callback(request: Request, code: str = None, error: str = None):
    """Handle Spotify OAuth callback."""
    if error:
        return RedirectResponse("/?auth_error=" + error)

    session_id = str(uuid.uuid4())
    auth = get_auth_manager(cache_path=f".cache-{session_id}")
    auth.redirect_uri = SPOTIFY_REDIRECT

    token_info = auth.get_access_token(code)
    sessions[session_id] = token_info

    response = RedirectResponse("/?auth=success")
    response.set_cookie("session_id", session_id, httponly=True, max_age=3600)
    return response


@app.get("/auth/status")
def auth_status(request: Request):
    """Check if user is logged in."""
    token = get_session_token(request)
    if token:
        sp = get_spotify_client_with_token(token)
        try:
            user = sp.current_user()
            return {
                "logged_in": True,
                "user": user.get("display_name", "Unknown"),
                "image": user.get("images", [{}])[0].get("url") if user.get("images") else None,
            }
        except Exception:
            pass
    return {"logged_in": False}


@app.get("/auth/logout")
def auth_logout(request: Request):
    """Clear session."""
    session_id = request.cookies.get("session_id")
    if session_id and session_id in sessions:
        del sessions[session_id]
    response = RedirectResponse("/")
    response.delete_cookie("session_id")
    return response


# ─── Request models ─────────────────────────────────────────────────

class SearchRequest(BaseModel):
    query: str
    top_k: int = 5


class TrackDNARequest(BaseModel):
    query: str


class MoodTimelineRequest(BaseModel):
    day: str


class MatchmakerRequest(BaseModel):
    friend_artists: list[str]


class DebateRequest(BaseModel):
    query: str


# ─── API routes ─────────────────────────────────────────────────────

@app.post("/api/search")
def api_search(req: SearchRequest):
    recs = search_tracks(req.query, top_k=req.top_k)
    if not recs:
        return {"response": "No matches found. Try indexing more music.", "tracks": []}

    context = "\n".join(
        f"- {r['name']} by {r['artist']} (score: {r['score']:.2f})"
        for r in recs
    )
    response = generate(
        prompt=req.query,
        system=f"""You are a music recommendation assistant. Matching tracks:

{context}

Recommend these tracks naturally. Explain why they match. Keep it brief.""",
    )
    return {"response": response, "tracks": recs}


@app.post("/api/taste")
def api_taste(request: Request):
    sp = require_spotify(request)
    result = explain_taste(sp=sp)
    return {"response": result}


@app.post("/api/track-dna")
def api_track_dna(req: TrackDNARequest):
    result = track_dna(req.query)
    return {"response": result}


@app.post("/api/roast")
def api_roast(request: Request):
    sp = require_spotify(request)
    result = playlist_roast(sp=sp)
    return {"response": result}


@app.post("/api/mood-timeline")
def api_mood_timeline(req: MoodTimelineRequest):
    result = mood_timeline(req.day)
    return {"response": result}


@app.post("/api/matchmaker")
def api_matchmaker(req: MatchmakerRequest, request: Request):
    sp = require_spotify(request)
    result = music_matchmaker(req.friend_artists, sp=sp)
    return {"response": result}


@app.post("/api/debate")
def api_debate(req: DebateRequest):
    result = music_debate(req.query)
    return {"response": result}


@app.post("/api/index/bulk")
def api_index_bulk():
    from indexer import index_bulk
    index_bulk()
    return {"status": "ok", "message": "Bulk indexing complete"}


# ─── Serve frontend ─────────────────────────────────────────────────

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def serve_frontend():
    return FileResponse("static/index.html")
