"""
search.py — RAG search module for NoteMind.

Takes a user's text query + video_id, retrieves the most relevant transcript
chunks from ChromaDB, then sends them to an LLM (Groq primary, Gemini fallback)
to generate a contextual answer with the relevant timestamp.

Flow:
  1. Embed the user's query using the same embedder (Phase 2).
  2. Query ChromaDB for the top-k most similar chunks (Phase 3).
  3. Build a RAG prompt with the retrieved chunks as context.
  4. Call the LLM to generate an answer based ONLY on the context.
  5. Return {answer, timestamp, source_chunks}.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import Config

from modules.embedder import generate_embeddings
from modules.vectorstore import query_chunks
from modules.transcript import format_timestamp


# ── Groq model preferences (in order of priority) ──
GROQ_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "llama3-8b-8192",
    "gemma2-9b-it",
    "mixtral-8x7b-32768",
]

# ── Gemini model ──
GEMINI_MODEL = "gemini-2.0-flash"


def _build_rag_prompt(query, chunks_with_timestamps):
    """
    Build the RAG prompt that instructs the LLM to answer using ONLY
    the provided video transcript chunks.

    Args:
        query (str): The user's question.
        chunks_with_timestamps (list): List of dicts with chunk_text,
                                       start_time, end_time.

    Returns:
        str: The full prompt string.
    """
    context_parts = []
    for i, chunk in enumerate(chunks_with_timestamps):
        ts_start = format_timestamp(chunk['start_time'])
        ts_end = format_timestamp(chunk['end_time'])
        context_parts.append(
            f"[Segment {i + 1}] ({ts_start} - {ts_end}):\n{chunk['chunk_text']}"
        )

    context_text = "\n\n".join(context_parts)

    prompt = f"""You are a helpful assistant that answers questions about a video based ONLY on the provided transcript segments. Do NOT make up information that is not in the transcript.

TRANSCRIPT SEGMENTS:
{context_text}

USER QUESTION: {query}

INSTRUCTIONS:
1. Answer the question using ONLY the information from the transcript segments above.
2. Mention the most relevant timestamp (e.g. "At 00:02:15, the speaker says...").
3. If the transcript segments do not contain enough information to answer the question, say "The video transcript does not contain enough information to answer this question."
4. Keep your answer concise but informative (2-4 sentences).

ANSWER:"""

    return prompt


def _call_groq(prompt):
    """
    Call the Groq API to generate a response.

    Tries multiple models in order of preference (some may be deprecated
    or unavailable on the free tier).

    Args:
        prompt (str): The full RAG prompt.

    Returns:
        str: The LLM's response text.

    Raises:
        Exception: If all Groq models fail.
    """
    if not Config.GROQ_API_KEY or Config.GROQ_API_KEY in ('your_groq_key_here', ''):
        raise ValueError("Groq API key is not configured.")

    import groq

    client = groq.Groq(api_key=Config.GROQ_API_KEY)
    last_error = None

    for model in GROQ_MODELS:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1024,
            )
            return response.choices[0].message.content.strip()
        except groq.NotFoundError:
            # Model not available, try the next one
            last_error = f"Model '{model}' not found on Groq."
            continue
        except groq.AuthenticationError as e:
            raise ValueError(f"Groq API key is invalid: {e}")
        except groq.RateLimitError as e:
            raise RuntimeError(
                f"Groq rate limit reached. Free tier allows ~30 requests/minute. "
                f"Please wait a moment and try again. Details: {e}"
            )
        except Exception as e:
            last_error = str(e)
            continue

    raise RuntimeError(f"All Groq models failed. Last error: {last_error}")


def _call_gemini(prompt):
    """
    Call the Gemini API as a fallback LLM.

    Uses the new google-genai SDK (not the deprecated google.generativeai).

    Args:
        prompt (str): The full RAG prompt.

    Returns:
        str: The LLM's response text.

    Raises:
        Exception: If the Gemini call fails.
    """
    if not Config.GEMINI_API_KEY or Config.GEMINI_API_KEY in ('your_gemini_key_here', ''):
        raise ValueError("Gemini API key is not configured.")

    from google import genai

    client = genai.Client(api_key=Config.GEMINI_API_KEY)

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
    )

    return response.text.strip()


def _call_llm(prompt):
    """
    Call the LLM with fallback logic: try Groq first, then Gemini.

    Args:
        prompt (str): The full RAG prompt.

    Returns:
        dict: {'success': True, 'answer': str, 'provider': str}
              OR {'success': False, 'error': str}
    """
    errors = []

    # ── Try Groq first (faster inference) ──
    try:
        answer = _call_groq(prompt)
        return {'success': True, 'answer': answer, 'provider': 'groq'}
    except ValueError as e:
        # Key not configured — skip to fallback
        errors.append(f"Groq: {e}")
    except RuntimeError as e:
        # Rate limit or all models failed
        errors.append(f"Groq: {e}")
    except Exception as e:
        errors.append(f"Groq: Unexpected error — {e}")

    # ── Try Gemini as fallback ──
    try:
        answer = _call_gemini(prompt)
        return {'success': True, 'answer': answer, 'provider': 'gemini'}
    except ValueError as e:
        errors.append(f"Gemini: {e}")
    except Exception as e:
        errors.append(f"Gemini: {e}")

    # ── Both failed ──
    return {
        'success': False,
        'error': (
            "Both LLM providers failed. Errors:\n"
            + "\n".join(f"  - {e}" for e in errors)
            + "\n\nPlease check your API keys in the .env file."
        ),
    }


def search_video(query, video_id, top_k=None, collection=None):
    """
    Main RAG search function: query a video and get an LLM-generated answer.

    This is the function that app.py and other modules should call.

    Args:
        query (str):     The user's search question.
        video_id (str):  The YouTube video ID to search within.
        top_k (int):     Number of chunks to retrieve. Defaults to 3.
        collection:      Optional ChromaDB collection (for testing).

    Returns:
        dict: {
            'success': True,
            'answer': str,           (LLM-generated answer)
            'provider': str,         (which LLM was used: 'groq' or 'gemini')
            'timestamp': str,        (formatted timestamp of most relevant chunk)
            'source_chunks': [       (the retrieved chunks used as context)
                {
                    'chunk_text': str,
                    'start_time': float,
                    'end_time': float,
                    'timestamp': str,  (formatted "HH:MM:SS")
                    'distance': float,
                },
                ...
            ],
            'timing': { ... }
        }
        OR on failure:
        dict: {
            'success': False,
            'error': str,
        }
    """
    import time as _time

    if top_k is None:
        top_k = 3  # Use fewer chunks for RAG (not all 5) to stay focused

    # ── Validate inputs ──
    if not query or not query.strip():
        return {'success': False, 'error': "Query cannot be empty."}

    if not video_id or not video_id.strip():
        return {'success': False, 'error': "Video ID cannot be empty."}

    query = query.strip()
    video_id = video_id.strip()

    timing = {}

    # ── Step 1: Embed the query ──
    t0 = _time.perf_counter()
    try:
        query_embedding = generate_embeddings(query)[0]
    except Exception as e:
        return {
            'success': False,
            'error': f"Failed to generate query embedding: {e}",
        }
    timing['1_query_embedding'] = round(_time.perf_counter() - t0, 3)

    # ── Step 2: Retrieve relevant chunks from ChromaDB ──
    t0 = _time.perf_counter()
    search_result = query_chunks(
        query_embedding=query_embedding,
        video_id=video_id,
        top_k=top_k,
        collection=collection,
    )
    timing['2_vector_query'] = round(_time.perf_counter() - t0, 3)

    if not search_result['results']:
        return {
            'success': False,
            'error': (
                f"No transcript chunks found for video '{video_id}'. "
                "Make sure the video has been processed first "
                "(transcript extracted, chunked, and stored)."
            ),
            'timing': timing,
        }

    chunks = search_result['results']

    # ── Step 3: Build RAG prompt ──
    prompt = _build_rag_prompt(query, chunks)

    # ── Step 4: Call the LLM ──
    t0 = _time.perf_counter()
    llm_result = _call_llm(prompt)
    timing['3_llm_call'] = round(_time.perf_counter() - t0, 3)

    if not llm_result['success']:
        llm_result['timing'] = timing
        return llm_result

    # ── Step 5: Format and return results ──
    # The most relevant chunk is the first one (lowest distance)
    best_chunk = chunks[0]
    best_timestamp = format_timestamp(best_chunk['start_time'])

    source_chunks = []
    for chunk in chunks:
        source_chunks.append({
            'chunk_text': chunk['chunk_text'],
            'start_time': chunk['start_time'],
            'end_time': chunk['end_time'],
            'timestamp': format_timestamp(chunk['start_time']),
            'distance': chunk.get('distance', 0.0),
        })

    return {
        'success': True,
        'answer': llm_result['answer'],
        'provider': llm_result['provider'],
        'timestamp': best_timestamp,
        'source_chunks': source_chunks,
        'timing': timing,
    }
