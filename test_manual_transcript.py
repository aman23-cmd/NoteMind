"""
test_manual_transcript.py — Quick manual test script for transcript extraction.

Run this to test with a real YouTube video URL:
    python test_manual_transcript.py

This will hit the real YouTube API (not mocked), so use it sparingly.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.transcript import get_transcript, format_timestamp


def main():
    # You can change this URL to any YouTube video you want to test
    test_urls = [
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",  # Rick Astley — short, has captions
    ]

    for url in test_urls:
        print("=" * 60)
        print(f"Testing URL: {url}")
        print("=" * 60)

        result = get_transcript(url)

        if result['success']:
            transcript = result['transcript']
            print(f"✅ Success! Got {len(transcript)} transcript snippets.")
            print(f"   Video ID: {result['video_id']}")
            print()

            # Show first 5 snippets as a preview
            print("   First 5 snippets:")
            for i, snippet in enumerate(transcript[:5]):
                ts = format_timestamp(snippet['start_time'])
                print(f"   [{ts}] {snippet['text']}")

            print()

            # Show last snippet
            last = transcript[-1]
            ts = format_timestamp(last['start_time'])
            print(f"   Last snippet: [{ts}] {last['text']}")
            print(f"   Video ends at: {format_timestamp(last['end_time'])}")
        else:
            print(f"❌ Failed: {result['error']}")

        print()


if __name__ == '__main__':
    main()
