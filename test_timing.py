"""
test_timing.py — End-to-end timing breakdown for the NoteMind pipeline.

Hits each endpoint and prints the server-side timing data returned in JSON.
Run while Flask is serving on http://127.0.0.1:5000
"""

import time
import requests
import json

BASE = "http://127.0.0.1:5000"
# Short, well-known video (~3.5 min) that definitely has captions
TEST_URL = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
TEST_VIDEO_ID = "dQw4w9WgXcQ"

def sep(title):
    print(f"\n{'=' * 65}")
    print(f"  {title}")
    print(f"{'=' * 65}")

def print_timing(timing_dict):
    """Pretty-print a timing dict from the server response."""
    if not timing_dict:
        print("      (no timing data)")
        return
    for k, v in timing_dict.items():
        if isinstance(v, (int, float)):
            print(f"      {k:<30s}  {v:>8.3f}s" if isinstance(v, float) else f"      {k:<30s}  {v:>8}")
        else:
            print(f"      {k:<30s}  {v}")

sep("NoteMind Pipeline - Full Timing Breakdown")

# ── 1. Process Video ──
sep("1/3  POST /process-video")
print(f"  URL: {TEST_URL}")
print(f"  (includes first-time model load + transcript + chunk + embed + store)")
print()

t0 = time.perf_counter()
try:
    r = requests.post(f"{BASE}/process-video", json={"url": TEST_URL}, timeout=300)
    client_elapsed = time.perf_counter() - t0
    print(f"  Client wall-clock: {client_elapsed:.3f}s")
    print(f"  HTTP status: {r.status_code}")

    pv = r.json()
    if pv.get("success"):
        print(f"  Chunks stored: {pv.get('chunks_stored')}")
        print(f"\n  SERVER-SIDE TIMING:")
        print_timing(pv.get('timing', {}))
    else:
        print(f"  ERROR: {pv.get('error')}")
        print_timing(pv.get('timing', {}))
except Exception as e:
    print(f"  REQUEST FAILED: {e}")

# ── 2. Search ──
sep("2/3  POST /search")
print(f"  Query: 'What is this video about?'")
print()

t0 = time.perf_counter()
try:
    r = requests.post(f"{BASE}/search", json={
        "video_id": TEST_VIDEO_ID,
        "query": "What is this video about?"
    }, timeout=120)
    client_elapsed = time.perf_counter() - t0
    print(f"  Client wall-clock: {client_elapsed:.3f}s")
    print(f"  HTTP status: {r.status_code}")

    sr = r.json()
    if sr.get("success"):
        print(f"  Provider: {sr.get('provider')}")
        print(f"  Answer (first 150 chars): {sr.get('answer','')[:150]}...")
    else:
        print(f"  ERROR: {sr.get('error','')[:200]}")

    # Print both inner (search_video) and outer timing
    inner_timing = sr.get('timing', {})
    print(f"\n  SERVER-SIDE TIMING:")
    print_timing(inner_timing)

except Exception as e:
    print(f"  REQUEST FAILED: {e}")

# ── 3. Generate Notes (force=True to bypass cache) ──
sep("3/3  POST /generate-notes (force=True)")
print()

t0 = time.perf_counter()
try:
    r = requests.post(f"{BASE}/generate-notes", json={
        "video_id": TEST_VIDEO_ID,
        "force": True
    }, timeout=300)
    client_elapsed = time.perf_counter() - t0
    print(f"  Client wall-clock: {client_elapsed:.3f}s")
    print(f"  HTTP status: {r.status_code}")

    gn = r.json()
    if gn.get("success"):
        print(f"  Provider: {gn.get('provider')}")
        print(f"  Method: {gn.get('method')}")
        print(f"  Notes length: {len(gn.get('notes',''))} chars")
    else:
        print(f"  ERROR: {gn.get('error','')[:200]}")

    print(f"\n  SERVER-SIDE TIMING:")
    print_timing(gn.get('timing', {}))

except Exception as e:
    print(f"  REQUEST FAILED: {e}")

# ── 4. Generate Notes again (cached) ──
sep("BONUS  POST /generate-notes (cached)")
print()

t0 = time.perf_counter()
try:
    r = requests.post(f"{BASE}/generate-notes", json={
        "video_id": TEST_VIDEO_ID,
    }, timeout=30)
    client_elapsed = time.perf_counter() - t0
    print(f"  Client wall-clock: {client_elapsed:.3f}s (should be near-instant)")
    print(f"  HTTP status: {r.status_code}")
    gn = r.json()
    print(f"  Cached: {gn.get('cached')}")
except Exception as e:
    print(f"  REQUEST FAILED: {e}")

sep("DONE")
