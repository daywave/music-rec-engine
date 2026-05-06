"""FastAPI backend for Music Recommendation Engine."""

import os
import uuid

from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, RedirectResponse
from pydantic import BaseModel

from chat_router import chat_with_context
from spotify_client import get_auth_manager, get_spotify_client_with_token, REDIRECT_URI
from taste import get_taste_profile, taste_to_text

app = FastAPI(title="Music Rec Engine")

# In-memory session store: {session_id: {token_info, taste_text, taste_profile}}
sessions: dict[str, dict] = {}

# Match sessions: {match_id: {creator_session_id, creator_name, creator_taste, friend_taste, friend_name}}
match_sessions: dict[str, dict] = {}

SPOTIFY_REDIRECT = os.getenv("SPOTIFY_REDIRECT_URI", "http://127.0.0.1:8000/auth/callback")


# ─── Helpers ─────────────────────────────────────────────────────────

def get_session(request: Request) -> dict | None:
    """Get session data from cookie."""
    session_id = request.cookies.get("session_id")
    if not session_id or session_id not in sessions:
        return None
    session = sessions[session_id]
    # Refresh token if expired
    token_info = session.get("token_info")
    if token_info:
        auth = get_auth_manager(cache_path=f".cache-{session_id}")
        if auth.is_token_expired(token_info):
            token_info = auth.refresh_access_token(token_info["refresh_token"])
            session["token_info"] = token_info
    return session


# ─── Auth routes ─────────────────────────────────────────────────────

@app.get("/auth/login")
def auth_login(match_id: str = None):
    auth = get_auth_manager()
    auth.redirect_uri = SPOTIFY_REDIRECT
    # Pass match_id through OAuth state so we know where to redirect after
    state = f"match:{match_id}" if match_id else "normal"
    url = auth.get_authorize_url(state=state)
    return RedirectResponse(url)


@app.get("/auth/callback")
def auth_callback(request: Request, code: str = None, error: str = None, state: str = "normal"):
    if error:
        return RedirectResponse("/?auth_error=" + error)

    session_id = str(uuid.uuid4())
    auth = get_auth_manager(cache_path=f".cache-{session_id}")
    auth.redirect_uri = SPOTIFY_REDIRECT
    token_info = auth.get_access_token(code)

    # Auto-analyze taste on login
    sp = get_spotify_client_with_token(token_info)
    user_name = "Unknown"
    try:
        user = sp.current_user()
        user_name = user.get("display_name", "Unknown")
        profile = get_taste_profile(limit=10, sp=sp)
        taste_text = taste_to_text(profile)
    except Exception:
        profile = {}
        taste_text = ""

    sessions[session_id] = {
        "token_info": token_info,
        "taste_text": taste_text,
        "taste_profile": profile,
        "user_name": user_name,
    }

    # If this is a match callback, store friend's taste in the match session
    if state and state.startswith("match:"):
        match_id = state.split(":", 1)[1]
        if match_id in match_sessions:
            match_sessions[match_id]["friend_taste"] = taste_text
            match_sessions[match_id]["friend_name"] = user_name
            response = RedirectResponse(f"/match/{match_id}")
            response.set_cookie("session_id", session_id, httponly=True, max_age=3600)
            return response

    response = RedirectResponse("/?auth=success")
    response.set_cookie("session_id", session_id, httponly=True, max_age=3600)
    return response


@app.get("/auth/status")
def auth_status(request: Request):
    session = get_session(request)
    if session and session.get("token_info"):
        sp = get_spotify_client_with_token(session["token_info"])
        try:
            user = sp.current_user()
            return {
                "logged_in": True,
                "user": user.get("display_name", "Unknown"),
                "image": user.get("images", [{}])[0].get("url") if user.get("images") else None,
                "has_taste": bool(session.get("taste_text")),
            }
        except Exception:
            pass
    return {"logged_in": False}


@app.get("/auth/logout")
def auth_logout(request: Request):
    session_id = request.cookies.get("session_id")
    if session_id and session_id in sessions:
        del sessions[session_id]
    response = RedirectResponse("/")
    response.delete_cookie("session_id")
    return response


# ─── Chat endpoint ──────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    history: list[dict] = []


@app.post("/api/chat")
def api_chat(req: ChatRequest, request: Request):
    session = get_session(request)
    sp = None
    taste_text = ""

    if session:
        token_info = session.get("token_info")
        if token_info:
            sp = get_spotify_client_with_token(token_info)
        taste_text = session.get("taste_text", "")

    result = chat_with_context(
        message=req.message,
        taste_text=taste_text,
        sp=sp,
        history=req.history,
    )

    return result


# ─── Bulk index ─────────────────────────────────────────────────────

@app.post("/api/index/bulk")
def api_index_bulk():
    from indexer import index_bulk
    index_bulk()
    return {"status": "ok", "message": "Bulk indexing complete"}


@app.post("/api/index/discover")
def api_index_discover(request: Request):
    """Index new music based on user's taste — finds music they DON'T already know."""
    session = get_session(request)
    if not session or not session.get("token_info"):
        raise HTTPException(status_code=401, detail="Spotify login required for discovery indexing")
    sp = get_spotify_client_with_token(session["token_info"])
    from indexer import index_discovery
    index_discovery(sp=sp)
    return {"status": "ok", "message": "Discovery indexing complete"}


# ─── Matchmaker ─────────────────────────────────────────────────────

@app.post("/api/match/create")
def create_match(request: Request):
    """Create a match session and return a shareable link."""
    session = get_session(request)
    if not session or not session.get("token_info"):
        raise HTTPException(status_code=401, detail="Spotify login required")

    session_id = request.cookies.get("session_id")
    match_id = str(uuid.uuid4())[:8]

    match_sessions[match_id] = {
        "creator_session_id": session_id,
        "creator_name": session.get("user_name", "Unknown"),
        "creator_taste": session.get("taste_text", ""),
        "friend_taste": None,
        "friend_name": None,
    }

    return {"match_id": match_id}


@app.get("/api/match/{match_id}")
def get_match_status(match_id: str):
    """Check match status and return results if both users connected."""
    if match_id not in match_sessions:
        raise HTTPException(status_code=404, detail="Match not found")

    match = match_sessions[match_id]

    if not match["friend_taste"]:
        return {
            "status": "waiting",
            "creator_name": match["creator_name"],
        }

    # Both connected — generate comparison
    from gemini_client import generate
    from recommend import search_tracks

    # Search for overlap
    combined = f"{match['creator_taste'][:200]} {match['friend_taste'][:200]}"
    overlap_tracks = search_tracks(combined, top_k=10)
    overlap_text = "\n".join(
        f"- {t['name']} by {t['artist']} ({', '.join(t.get('genres', [])[:2])})"
        for t in overlap_tracks
    )

    result = generate(
        prompt=f"""{match['creator_name']}'s taste:
{match['creator_taste']}

{match['friend_name']}'s taste:
{match['friend_taste']}

Tracks in the overlap zone:
{overlap_text}""",
        system="""You are a music matchmaker comparing two real people's Spotify data.

Create a fun, detailed compatibility report:
1. **Compatibility Score** (X/10) with a witty one-liner
2. **Where You Overlap** — shared vibes, genres, energy levels
3. **Where You Diverge** — what's unique to each person
4. **Bridge Playlist** — 5 tracks from our database both would enjoy, with why
5. **Steal This Track** — one track each person should borrow from the other
6. **Verdict** — a fun summary of the musical relationship

Use their actual names. Be specific about the music. Make it social and shareable.""",
        max_tokens=1500,
    )

    return {
        "status": "complete",
        "creator_name": match["creator_name"],
        "friend_name": match["friend_name"],
        "response": result,
        "tracks": overlap_tracks,
    }


@app.get("/match/{match_id}")
def serve_match_page(match_id: str):
    """Serve the match page."""
    return FileResponse("static/match.html")


# ─── Serve frontend ─────────────────────────────────────────────────

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def serve_frontend():
    return FileResponse("static/index.html")
