"""Recommendation engine — find similar tracks via Pinecone with reranking."""

from pinecone import Pinecone

from config import PINECONE_API_KEY, PINECONE_INDEX_NAME


def search_tracks(
    query: str,
    top_k: int = 5,
    mood_filter: str = None,
    energy_filter: str = None,
    genre_filter: str = None,
    rerank: bool = True,
) -> list[dict]:
    """Search for tracks with optional filtering and reranking."""
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX_NAME)

    # Build filter
    filter_dict = {}
    if mood_filter:
        filter_dict["moods"] = {"$in": [mood_filter]}
    if energy_filter:
        filter_dict["energy"] = {"$eq": energy_filter}
    if genre_filter:
        filter_dict["genres"] = {"$in": [genre_filter]}

    # Fetch more candidates for reranking
    fetch_k = top_k * 4 if rerank else top_k

    search_params = {
        "namespace": "tracks",
        "top_k": fetch_k,
        "inputs": {"text": query},
    }

    if filter_dict:
        search_params["filter"] = filter_dict

    # Add reranking
    if rerank:
        search_params["rerank"] = {
            "model": "pinecone-rerank-v0",
            "rank_fields": ["description"],
            "top_n": top_k,
        }

    results = index.search_records(**search_params)

    recommendations = []
    for hit in results["result"]["hits"]:
        fields = hit["fields"]
        recommendations.append({
            "id": hit.id,
            "score": hit.score,
            "name": fields.get("name", "Unknown"),
            "artist": fields.get("artist", "Unknown"),
            "album": fields.get("album", "Unknown"),
            "genres": fields.get("genres", []),
            "moods": fields.get("moods", []),
            "energy": fields.get("energy", ""),
        })

    return recommendations


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python recommend.py 'your query'")
        print("Example: python recommend.py 'aggressive electronic music'")
        sys.exit(1)

    query = " ".join(sys.argv[1:])
    recs = search_tracks(query)
    print(f"\nResults for: {query}\n")
    for i, rec in enumerate(recs, 1):
        tags = ", ".join(rec.get("genres", [])[:3])
        print(f"  {i}. {rec['name']} — {rec['artist']} (score: {rec['score']:.3f}) [{tags}]")
