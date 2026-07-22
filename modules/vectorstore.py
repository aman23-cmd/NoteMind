"""
vectorstore.py — ChromaDB operations for NoteMind.

Handles storing video transcript chunks (with embeddings and metadata)
in ChromaDB and querying them by semantic similarity.

Key design decisions:
  - Uses a single collection "video_transcripts" with video_id in metadata
    (so we can filter queries to a specific video using ChromaDB's where clause).
  - Uses `upsert` (not `add`) so re-inserting the same video is idempotent
    and doesn't crash — important since free-tier hosting has ephemeral disk.
  - Chunk IDs are formatted as "{video_id}_{chunk_id}" to be globally unique.
  - PersistentClient for production, EphemeralClient for tests.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import Config

import chromadb

# ── Module-level client and collection cache ──
_client = None
_collection = None

# Collection name (from design.md)
COLLECTION_NAME = "video_transcripts"


def get_client(persistent=True, path=None):
    """
    Get or create a ChromaDB client.

    Args:
        persistent (bool): If True, use PersistentClient (data saved to disk).
                           If False, use EphemeralClient (in-memory, for tests).
        path (str):        Path for persistent storage.
                           Defaults to Config.CHROMADB_PATH ("chroma_db/").

    Returns:
        chromadb.ClientAPI: The ChromaDB client instance.
    """
    global _client, _collection

    if _client is not None:
        # Verify the cached client is still usable (Rust bindings can go stale
        # when Flask's debug reloader restarts the process)
        try:
            _client.heartbeat()
            return _client
        except Exception:
            _client = None
            _collection = None

    if persistent:
        storage_path = path or Config.CHROMADB_PATH
        os.makedirs(storage_path, exist_ok=True)
        _client = chromadb.PersistentClient(path=storage_path)
    else:
        _client = chromadb.EphemeralClient()

    return _client


def get_collection(client=None):
    """
    Get or create the video_transcripts collection.

    We explicitly pass embedding_function=None because we always supply our
    own embeddings (from Gemini API) when adding/querying. Without this,
    ChromaDB defaults to downloading its own onnx-based embedding model from
    HuggingFace Hub, which wastes memory and isn't needed here.

    Args:
        client: A ChromaDB client. If None, uses the default persistent client.

    Returns:
        chromadb.Collection: The collection for storing transcript chunks.
    """
    global _collection

    if _collection is not None and client is None:
        return _collection

    if client is None:
        client = get_client()

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},  # use cosine similarity
        embedding_function=None,
    )

    if client == _client:
        _collection = collection

    return collection


def store_chunks(video_id, chunks, embeddings, collection=None):
    """
    Store transcript chunks with their embeddings in ChromaDB.

    Uses `upsert` so calling this multiple times for the same video_id
    is safe and idempotent — existing chunks are updated, not duplicated.

    Args:
        video_id (str):     The YouTube video ID.
        chunks (list):      List of chunk dicts from chunker.py:
                            [{'chunk_id': str, 'chunk_text': str,
                              'start_time': float, 'end_time': float}, ...]
        embeddings (list):  List of embedding vectors (list of floats),
                            one per chunk, from embedder.py.
        collection:         Optional ChromaDB collection (for testing).
                            If None, uses the default collection.

    Returns:
        dict: {
            'success': True,
            'video_id': str,
            'chunks_stored': int,
        }
        OR on failure:
        dict: {
            'success': False,
            'error': str,
        }
    """
    if not chunks or not embeddings:
        return {
            'success': False,
            'error': "No chunks or embeddings provided.",
        }

    if len(chunks) != len(embeddings):
        return {
            'success': False,
            'error': (f"Mismatch: {len(chunks)} chunks but "
                      f"{len(embeddings)} embeddings."),
        }

    if collection is None:
        collection = get_collection()

    try:
        # Build the data for ChromaDB upsert
        ids = [f"{video_id}_{chunk['chunk_id']}" for chunk in chunks]
        documents = [chunk['chunk_text'] for chunk in chunks]
        metadatas = [
            {
                'video_id': video_id,
                'start_time': chunk['start_time'],
                'end_time': chunk['end_time'],
                'chunk_id': chunk['chunk_id'],
            }
            for chunk in chunks
        ]

        # Upsert (insert or update) — safe for re-processing
        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

        return {
            'success': True,
            'video_id': video_id,
            'chunks_stored': len(chunks),
        }

    except Exception as e:
        return {
            'success': False,
            'error': f"Failed to store chunks in ChromaDB: {str(e)}",
        }


def query_chunks(query_embedding, video_id, top_k=None, collection=None):
    """
    Search ChromaDB for the most similar chunks to a query embedding.

    Filters results to only the specified video_id.

    Args:
        query_embedding (list): The embedding vector of the user's query
                                (list of 384 floats).
        video_id (str):         The YouTube video ID to search within.
        top_k (int):            Number of results to return.
                                Defaults to Config.TOP_K_RESULTS (5).
        collection:             Optional ChromaDB collection (for testing).

    Returns:
        dict: {
            'success': True,
            'video_id': str,
            'results': [
                {
                    'chunk_text': str,
                    'start_time': float,
                    'end_time': float,
                    'chunk_id': str,
                    'distance': float,   (lower = more similar for cosine)
                },
                ...
            ]
        }
        OR on failure / no results:
        dict: {
            'success': True,
            'video_id': str,
            'results': [],
        }
    """
    if top_k is None:
        top_k = Config.TOP_K_RESULTS

    if collection is None:
        collection = get_collection()

    try:
        # Check if collection has any data for this video
        count = collection.count()
        if count == 0:
            return {
                'success': True,
                'video_id': video_id,
                'results': [],
            }

        # Limit top_k to the number of items available
        actual_top_k = min(top_k, count)

        result = collection.query(
            query_embeddings=[query_embedding],
            n_results=actual_top_k,
            where={"video_id": video_id},
            include=["documents", "metadatas", "distances"],
        )

        # Parse ChromaDB's nested response format
        results = []
        if result['ids'] and result['ids'][0]:
            for i in range(len(result['ids'][0])):
                results.append({
                    'chunk_text': result['documents'][0][i],
                    'start_time': result['metadatas'][0][i]['start_time'],
                    'end_time': result['metadatas'][0][i]['end_time'],
                    'chunk_id': result['metadatas'][0][i]['chunk_id'],
                    'distance': result['distances'][0][i],
                })

        return {
            'success': True,
            'video_id': video_id,
            'results': results,
        }

    except Exception as e:
        return {
            'success': True,
            'video_id': video_id,
            'results': [],
        }


def delete_video_chunks(video_id, collection=None):
    """
    Delete all chunks for a given video_id from ChromaDB.

    Useful when re-processing a video or cleaning up.

    Args:
        video_id (str):  The YouTube video ID.
        collection:      Optional ChromaDB collection (for testing).

    Returns:
        dict: {'success': True, 'deleted': int}
    """
    if collection is None:
        collection = get_collection()

    try:
        # Find all chunk IDs for this video
        existing = collection.get(
            where={"video_id": video_id},
            include=[],
        )

        if existing['ids']:
            collection.delete(ids=existing['ids'])
            return {'success': True, 'deleted': len(existing['ids'])}

        return {'success': True, 'deleted': 0}

    except Exception as e:
        return {'success': False, 'error': str(e)}


def has_video_chunks(video_id, collection=None):
    """
    Check if ChromaDB already has chunks stored for a given video_id.

    Args:
        video_id (str):  The YouTube video ID.
        collection:      Optional ChromaDB collection (for testing).

    Returns:
        bool: True if chunks exist for this video_id.
    """
    if collection is None:
        collection = get_collection()

    try:
        existing = collection.get(
            where={"video_id": video_id},
            include=[],
        )
        return len(existing['ids']) > 0
    except Exception:
        return False


def reset_module():
    """Reset module-level caches. Used in tests to start fresh."""
    global _client, _collection
    _client = None
    _collection = None
