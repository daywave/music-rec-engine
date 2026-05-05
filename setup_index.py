"""Create the Pinecone integrated index for music recommendations."""

from pinecone import Pinecone

from config import PINECONE_API_KEY, PINECONE_INDEX_NAME


def create_index():
    pc = Pinecone(api_key=PINECONE_API_KEY)

    existing = [idx.name for idx in pc.list_indexes()]
    if PINECONE_INDEX_NAME in existing:
        print(f"Index '{PINECONE_INDEX_NAME}' already exists.")
        return

    pc.create_index_for_model(
        name=PINECONE_INDEX_NAME,
        cloud="aws",
        region="us-east-1",
        embed={
            "model": "multilingual-e5-large",
            "field_map": {"text": "description"},
        },
    )
    print(f"Created integrated index '{PINECONE_INDEX_NAME}' (multilingual-e5-large)")


if __name__ == "__main__":
    create_index()
