"""Index tracks into Pinecone for similarity search."""

import sys

from pinecone import Pinecone

from config import PINECONE_API_KEY, PINECONE_INDEX_NAME
from spotify_client import get_playlist_tracks, search_tracks_with_features


def track_to_description(track: dict) -> str:
    """Convert track metadata to a searchable text description."""
    return f"{track['name']} by {track['artist']} from the album {track['album']}"


def upsert_tracks(tracks: list[dict]):
    """Upsert tracks into Pinecone integrated index."""
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX_NAME)

    records = []
    for track in tracks:
        records.append({
            "_id": track["id"],
            "description": track_to_description(track),
            "name": track["name"],
            "artist": track["artist"],
            "album": track["album"],
            "popularity": track.get("popularity", 0),
        })

    # Upsert in batches of 96 (Pinecone integrated index limit)
    batch_size = 96
    for i in range(0, len(records), batch_size):
        batch = records[i:i + batch_size]
        index.upsert_records(namespace="tracks", records=batch)
        print(f"Upserted batch {i // batch_size + 1} ({len(batch)} tracks)")

    print(f"Done! Indexed {len(records)} tracks.")


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
    # Process in chunks to avoid rate limits
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


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python indexer.py <playlist_id>")
        print("  python indexer.py --search 'query1' 'query2' ...")
        print("  python indexer.py --bulk")
        print("Example:")
        print("  python indexer.py --search 'crystal castles' 'aphex twin' 'machine girl'")
        sys.exit(1)

    if sys.argv[1] == "--bulk":
        index_bulk()
    elif sys.argv[1] == "--search":
        queries = sys.argv[2:]
        index_searches(queries)
    else:
        index_playlist(sys.argv[1])
