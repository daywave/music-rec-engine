"""Recommendation engine — find similar tracks via Pinecone."""

from pinecone import Pinecone

from config import PINECONE_API_KEY, PINECONE_INDEX_NAME


def search_tracks(query: str, top_k: int = 10) -> list[dict]:
    """Search for tracks matching a natural language query."""
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX_NAME)

    results = index.search_records(
        namespace="tracks",
        top_k=top_k,
        inputs={"text": query},
    )

    recommendations = []
    for hit in results["result"]["hits"]:
        fields = hit["fields"]
        recommendations.append({
            "id": hit.id,
            "score": hit.score,
            "name": fields.get("name", "Unknown"),
            "artist": fields.get("artist", "Unknown"),
            "album": fields.get("album", "Unknown"),
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
        print(f"  {i}. {rec['name']} — {rec['artist']} (score: {rec['score']:.3f})")
