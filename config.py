"""
config.py — Central configuration for the NoteMind application.

Loads API keys from the .env file using python-dotenv.
Raises a clear error at startup if required keys are missing or still
set to their placeholder values, so you know immediately what to fix.
"""

import os
import sys
from dotenv import load_dotenv

# Load .env from the project root (same directory as this file)
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env'))

# ── Base directory (project root) ──
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class Config:
    """Holds all configuration values for the app."""

    # ── API Keys (read from environment, never hardcode) ──
    GROQ_API_KEY = os.environ.get('GROQ_API_KEY', '')
    GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', '')

    # ── Database Paths ──
    SQLITE_DB_PATH = os.path.join(BASE_DIR, 'database', 'notemind.db')
    CHROMADB_PATH = os.path.join(BASE_DIR, 'chroma_db')

    # ── Embedding Model ──
    EMBEDDING_MODEL = 'all-MiniLM-L6-v2'

    # ── Chunk Settings (from rules.md: 300-500 words, ~50 word overlap) ──
    CHUNK_SIZE = 400        # target words per chunk
    CHUNK_OVERLAP = 50      # overlap words between chunks

    # ── RAG Settings ──
    TOP_K_RESULTS = 5       # number of chunks to retrieve per query

    @classmethod
    def validate_keys(cls):
        """
        Check that at least one LLM API key is set and not a placeholder.
        Called at startup so you get an immediate, readable error
        instead of a cryptic failure later during an LLM call.
        """
        placeholder_values = {'', 'your_groq_key_here', 'your_gemini_key_here'}

        groq_ok = cls.GROQ_API_KEY not in placeholder_values
        gemini_ok = cls.GEMINI_API_KEY not in placeholder_values

        if not groq_ok and not gemini_ok:
            print("=" * 60)
            print("ERROR: No valid LLM API key found!")
            print("")
            print("Open the .env file in the project root and replace")
            print("the placeholder values with your real API keys:")
            print("")
            print("  GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxx")
            print("  GEMINI_API_KEY=AIzaXXXXXXXXXXXX")
            print("")
            print("You need at least ONE of these keys (Groq or Gemini).")
            print("Both are free — sign up at:")
            print("  Groq  : https://console.groq.com/keys")
            print("  Gemini: https://aistudio.google.com/app/apikey")
            print("=" * 60)
            # Don't crash the app — Phase 0 just needs Flask running.
            # Later phases that call LLMs will fail-fast with a clear message.
            return False

        if groq_ok:
            print("[OK] Groq API key loaded successfully.")
        if gemini_ok:
            print("[OK] Gemini API key loaded successfully.")

        return True


# Run validation as soon as this module is imported
Config.validate_keys()
