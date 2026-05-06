"""Index tracks into Pinecone with AI-enriched descriptions."""

import json
import sys
import time

from pinecone import Pinecone

from config import PINECONE_API_KEY, PINECONE_INDEX_NAME
from gemini_client import generate
from spotify_client import get_playlist_tracks, search_tracks_with_features


# ─── Enrichment ─────────────────────────────────────────────────────

def enrich_tracks(tracks: list[dict]) -> list[dict]:
    """Use Gemini to generate rich descriptions and metadata tags for tracks."""
    # Build a batch prompt for efficiency
    tracks_list = "\n".join(
        f'{i+1}. "{t["name"]}" by {t["artist"]} (album: {t["album"]})'
        for i, t in enumerate(tracks)
    )

    result = generate(
        prompt=tracks_list,
        system="""For each numbered track, generate a rich description and metadata tags.
Return ONLY a JSON array where each element has:
- "index": the track number (1-based)
- "description": a rich 2-3 sentence description covering genre, mood, sonic texture, era, and what kind of listener would enjoy it. Be specific about the sound.
- "genres": array of 2-4 genre tags (lowercase)
- "moods": array of 2-4 mood tags (lowercase, e.g., "aggressive", "melancholic", "euphoric", "introspective")
- "energy": one of "low", "medium", "high"
- "decade": the decade string like "2000s", "2010s", "2020s"

Be musically accurate. If you don't know a track, infer from the artist and album.
Return only the JSON array, no other text.""",
        max_tokens=3000,
    )

    try:
        clean = result.strip()
        if clean.startswith("```"):
            clean = clean.split("\n", 1)[1].rsplit("```", 1)[0]
        enrichments = json.loads(clean)
    except (json.JSONDecodeError, IndexError):
        # Fallback: basic descriptions
        enrichments = []

    # Merge enrichments back into tracks
    enrich_map = {}
    for e in enrichments:
        idx = e.get("index", 0) - 1
        if 0 <= idx < len(tracks):
            enrich_map[idx] = e

    for i, track in enumerate(tracks):
        if i in enrich_map:
            e = enrich_map[i]
            track["rich_description"] = e.get("description", "")
            track["genres"] = e.get("genres", [])
            track["moods"] = e.get("moods", [])
            track["energy"] = e.get("energy", "medium")
            track["decade"] = e.get("decade", "unknown")
        else:
            track["rich_description"] = ""
            track["genres"] = []
            track["moods"] = []
            track["energy"] = "medium"
            track["decade"] = "unknown"

    return tracks


def build_description(track: dict) -> str:
    """Build the final description for Pinecone embedding."""
    parts = [f"{track['name']} by {track['artist']}."]
    if track.get("rich_description"):
        parts.append(track["rich_description"])
    if track.get("genres"):
        parts.append(f"Genres: {', '.join(track['genres'])}.")
    if track.get("moods"):
        parts.append(f"Mood: {', '.join(track['moods'])}.")
    return " ".join(parts)


# ─── Upsert ─────────────────────────────────────────────────────────

def upsert_tracks(tracks: list[dict]):
    """Enrich and upsert tracks into Pinecone."""
    # Enrich in batches of 15 (to fit Gemini context)
    print("Enriching tracks with AI descriptions...")
    enriched = []
    batch_size = 15
    for i in range(0, len(tracks), batch_size):
        batch = tracks[i:i + batch_size]
        print(f"  Enriching batch {i // batch_size + 1} ({len(batch)} tracks)...")
        enriched.extend(enrich_tracks(batch))
        time.sleep(1)  # Rate limit

    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX_NAME)

    records = []
    for track in enriched:
        record = {
            "_id": track["id"],
            "description": build_description(track),
            "name": track["name"],
            "artist": track["artist"],
            "album": track["album"],
            "popularity": track.get("popularity", 0),
            "energy": track.get("energy", "medium"),
            "decade": track.get("decade", "unknown"),
        }
        if track.get("genres"):
            record["genres"] = track["genres"]
        if track.get("moods"):
            record["moods"] = track["moods"]
        records.append(record)

    # Upsert in batches of 96
    for i in range(0, len(records), 96):
        batch = records[i:i + 96]
        index.upsert_records(namespace="tracks", records=batch)
        print(f"Upserted batch {i // 96 + 1} ({len(batch)} tracks)")

    print(f"Done! Indexed {len(records)} enriched tracks.")


# ─── Index modes ────────────────────────────────────────────────────

def index_playlist(playlist_id: str):
    """Fetch playlist tracks and upsert into Pinecone."""
    print(f"Fetching tracks from playlist: {playlist_id}")
    tracks = get_playlist_tracks(playlist_id)
    print(f"Got {len(tracks)} tracks")
    upsert_tracks(tracks)


def index_searches(queries: list[str], limit: int = 10):
    """Search tracks by queries and upsert into Pinecone."""
    print(f"Searching {len(queries)} queries, {limit} tracks each...")
    tracks = search_tracks_with_features(queries, limit=limit)
    print(f"Got {len(tracks)} unique tracks")
    upsert_tracks(tracks)


BULK_QUERIES = [
    # Electronic
    "crystal castles", "aphex twin", "machine girl", "boards of canada",
    "autechre", "flying lotus", "oneohtrix point never", "burial",
    "arca", "sophie electronic", "gesaffelstein", "justice electronic",
    # Rock / Alt
    "radiohead", "my bloody valentine", "sonic youth", "pixies",
    "the strokes", "arctic monkeys", "tame impala", "king gizzard",
    # Hip Hop / Rap
    "kendrick lamar", "tyler the creator", "jpegmafia", "danny brown",
    "mf doom", "earl sweatshirt", "playboi carti", "death grips",
    # Regional Mexican
    "peso pluma", "natanael cano", "junior h", "fuerza regida",
    "grupo frontera", "ivan cornejo", "eslabon armado", "carin leon",
    # Latin
    "bad bunny", "rosalia", "rauw alejandro", "karol g",
    # Pop
    "charli xcx", "billie eilish", "dua lipa", "frank ocean",
    # R&B / Soul
    "sza", "daniel caesar", "steve lacy", "the weeknd",
    # Jazz / Experimental
    "kamasi washington", "thundercat", "hiatus kaiyote", "nubya garcia",
    # Metal / Punk
    "deftones", "turnstile", "show me the body", "code orange",
    # Ambient / Downtempo
    "brian eno ambient", "tycho", "bonobo", "four tet",
]


def index_bulk():
    """Index a diverse catalog across many genres."""
    print(f"Bulk indexing {len(BULK_QUERIES)} queries...")
    chunk_size = 5
    all_tracks = []
    for i in range(0, len(BULK_QUERIES), chunk_size):
        chunk = BULK_QUERIES[i:i + chunk_size]
        print(f"  Searching: {', '.join(chunk)}")
        tracks = search_tracks_with_features(chunk, limit=10)
        all_tracks.extend(tracks)

    # Deduplicate
    seen = set()
    unique = []
    for t in all_tracks:
        if t["id"] not in seen:
            seen.add(t["id"])
            unique.append(t)

    print(f"Got {len(unique)} unique tracks total")
    upsert_tracks(unique)


# ─── Discovery indexing (taste-based) ───────────────────────────────

def index_discovery(sp=None):
    """Analyze user's taste and index NEW music they'd likely enjoy but haven't heard."""
    from taste import get_taste_profile, taste_to_text, get_all_taste_artists

    print("Analyzing your taste for discovery...")
    profile = get_taste_profile(limit=10, sp=sp)
    taste_text = taste_to_text(profile)
    known_artists = get_all_taste_artists(profile)

    # Step 1: Gemini generates discovery queries
    result = generate(
        prompt=f"""User's listening data:
{taste_text}

Artists they already know: {', '.join(known_artists)}""",
        system="""You are a music discovery engine. Based on this person's taste, generate
search queries that will find music they DON'T already know but would love.

Strategy:
1. Find artists SIMILAR to their favorites but less popular / underground
2. Find adjacent genres they haven't explored (e.g., if they like electronic, try IDM or ambient techno)
3. Find older influences of their favorite artists
4. Find newer artists carrying the torch of their favorites
5. Find cross-genre bridges (e.g., jazz-electronic fusion if they like both)

IMPORTANT: Do NOT include artists they already listen to.

Return ONLY a JSON array of 20 search query strings.
Make queries specific: "experimental hip hop underground 2023" not just "hip hop".
Return only the JSON array.""",
        max_tokens=800,
    )

    try:
        clean = result.strip()
        if clean.startswith("```"):
            clean = clean.split("\n", 1)[1].rsplit("```", 1)[0]
        discovery_queries = json.loads(clean)
    except (json.JSONDecodeError, IndexError):
        discovery_queries = [
            f"artists similar to {a}" for a in known_artists[:10]
        ]

    print(f"Generated {len(discovery_queries)} discovery queries")

    # Step 2: Search Spotify for these queries
    all_tracks = []
    known_lower = {a.lower() for a in known_artists}

    chunk_size = 5
    for i in range(0, len(discovery_queries), chunk_size):
        chunk = discovery_queries[i:i + chunk_size]
        print(f"  Searching: {chunk[0][:50]}...")
        tracks = search_tracks_with_features(chunk, limit=10)
        # Filter out tracks by artists they already know
        for t in tracks:
            if t["artist"].lower() not in known_lower:
                all_tracks.append(t)

    # Deduplicate
    seen = set()
    unique = []
    for t in all_tracks:
        if t["id"] not in seen:
            seen.add(t["id"])
            unique.append(t)

    print(f"Got {len(unique)} new discovery tracks (filtered out known artists)")
    if unique:
        upsert_tracks(unique)
    else:
        print("No new tracks found. Try expanding your taste profile!")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python indexer.py <playlist_id>")
        print("  python indexer.py --search 'query1' 'query2' ...")
        print("  python indexer.py --bulk")
        print("  python indexer.py --discover    (indexes music based on your taste)")
        print("Example:")
        print("  python indexer.py --search 'crystal castles' 'aphex twin' 'machine girl'")
        sys.exit(1)

    if sys.argv[1] == "--bulk":
        index_bulk()
    elif sys.argv[1] == "--discover":
        index_discovery()
    elif sys.argv[1] == "--search":
        queries = sys.argv[2:]
        index_searches(queries)
    else:
        index_playlist(sys.argv[1])
