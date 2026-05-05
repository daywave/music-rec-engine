# Music Recommendation Engine

## What this is
Music recommendation engine + RAG chatbot using Pinecone (vector DB), Spotify API (track data), and Claude (generation).

## Stack
- Python 3.11+
- Pinecone (serverless, 8-dim cosine index)
- Spotipy (Spotify Web API wrapper)
- Anthropic SDK (Claude for RAG)
- rich (CLI output)

## Architecture

Two modes:
1. **Recommendation engine** — vectorizes Spotify audio features (danceability, energy, valence, tempo, acousticness, instrumentalness, speechiness, liveness) into 8-dim vectors. Cosine similarity finds tracks with matching vibes.
2. **RAG chatbot** — Claude extracts mood params from natural language → query Pinecone → Claude generates response with retrieved tracks as context.

## File structure
- `config.py` — env vars, index name
- `spotify_client.py` — Spotify API: fetch playlist tracks, search, audio features
- `setup_index.py` — create Pinecone index (run once)
- `indexer.py` — vectorize tracks → upsert to Pinecone
- `recommend.py` — similarity search (by track ID or mood floats)
- `chatbot.py` — RAG chatbot loop (Claude + Pinecone)

## Running
```bash
source .venv/bin/activate
python setup_index.py          # create index
python indexer.py <playlist_id> # index tracks
python recommend.py <track_id>  # get recs
python chatbot.py               # interactive chatbot
```

## Key decisions
- 8 audio features as vector dimensions (not text embeddings) — keeps it simple and explainable
- Cosine metric for similarity
- Pinecone serverless on AWS us-east-1
- Client Credentials flow (no user auth needed)
- Claude Sonnet for mood extraction + response generation

## Next steps
- Add more playlists/genres for richer index
- Add genre/artist metadata filtering
- Hybrid search (audio features + text embeddings for lyrics/descriptions)
- Web UI with Vue
