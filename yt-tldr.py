#!/usr/bin/env python3
"""
YouTube TL;DR - Get a summary of any YouTube video to decide if it's worth watching.

Uses YouTube's transcript if available, otherwise falls back to Whisper transcription.
Whisper fallback requires: pip install openai-whisper yt-dlp ffmpeg
"""

import argparse
import os
import re
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path
from typing import Optional

# Suppress known warnings (fallback for older Python versions)
warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL")
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU")

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound
import anthropic


def extract_video_id(url: str) -> Optional[str]:
    """Extract the video ID from various YouTube URL formats."""
    patterns = [
        r'(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/)([a-zA-Z0-9_-]{11})',
        r'^([a-zA-Z0-9_-]{11})$'  # Just the ID itself
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


def get_transcript(video_id: str) -> str:
    """Fetch the transcript for a YouTube video."""
    try:
        api = YouTubeTranscriptApi()
        transcript = api.fetch(video_id, languages=['en', 'en-US', 'en-GB'])
        return " ".join([entry.text for entry in transcript])
    except TranscriptsDisabled:
        raise Exception("Transcripts are disabled for this video")
    except NoTranscriptFound:
        raise Exception("No transcript found for this video")


def transcribe_with_whisper(video_url: str) -> str:
    """Download audio and transcribe with Whisper as fallback."""
    try:
        import whisper
    except ImportError:
        raise Exception(
            "Whisper not installed. Install with: pip install openai-whisper\n"
            "You'll also need: pip install yt-dlp && brew install ffmpeg"
        )

    # Check for yt-dlp
    try:
        subprocess.run(["yt-dlp", "--version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        raise Exception("yt-dlp not installed. Install with: pip install yt-dlp")

    # Check for ffmpeg
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        raise Exception("ffmpeg not installed. Install with: brew install ffmpeg")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Download audio
        print("Downloading audio...")
        audio_template = os.path.join(tmpdir, "audio.%(ext)s")
        result = subprocess.run(
            [
                "yt-dlp",
                "--quiet",
                "--no-warnings",
                "-f", "worst[ext=mp4]",
                "-x",
                "--audio-format", "mp3",
                "-o", audio_template,
                video_url
            ],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            # Extract just the ERROR line from stderr
            error_lines = [
                line for line in result.stderr.split('\n')
                if line.strip().startswith('ERROR')
            ]
            error_msg = error_lines[0] if error_lines else "Unknown download error"
            raise Exception(f"Failed to download audio: {error_msg}")

        # Find the downloaded audio file
        audio_file = None
        for f in os.listdir(tmpdir):
            if f.startswith("audio.") and not f.endswith(".part"):
                audio_file = os.path.join(tmpdir, f)
                break

        if not audio_file or not os.path.exists(audio_file):
            raise Exception("Audio file not found after download")

        # Transcribe with Whisper
        # Use MPS (Metal) on Apple Silicon for faster transcription
        import torch
        if torch.backends.mps.is_available():
            device = "mps"
            print("Transcribing with Whisper on Metal GPU...")
        else:
            device = "cpu"
            print("Transcribing with Whisper on CPU (this may take a while)...")

        model = whisper.load_model("base", device=device)
        result = model.transcribe(audio_file, verbose=False, fp16=False)

        return result["text"]


def summarize_with_claude(transcript: str, video_url: str) -> str:
    """Send transcript to Claude API for summarization."""
    client = anthropic.Anthropic()

    # Truncate very long transcripts to stay within token limits
    max_chars = 100000
    if len(transcript) > max_chars:
        transcript = transcript[:max_chars] + "... [transcript truncated]"

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": f"""Here's a transcript from a YouTube video. Please provide:

1. **TL;DR** (2-3 sentences max) - What's this video about?
2. **Key Points** (3-5 bullet points) - Main takeaways
3. **Worth Watching?** - Quick verdict: who would find this valuable and who can skip it

Transcript:
{transcript}"""
            }
        ]
    )

    return message.content[0].text


def save_transcript(video_id: str, video_url: str, transcript: str, source: str):
    """Save the transcript to ~/.yt-tldr/ for later reference."""
    cache_dir = Path.home() / ".yt-tldr"
    cache_dir.mkdir(exist_ok=True)

    transcript_file = cache_dir / "last_transcript.txt"
    transcript_file.write_text(
        f"Video ID: {video_id}\n"
        f"URL: {video_url}\n"
        f"Source: {source}\n"
        f"{'=' * 60}\n\n"
        f"{transcript}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Get a TL;DR summary of a YouTube video"
    )
    parser.add_argument("url", help="YouTube video URL or video ID")
    args = parser.parse_args()

    # Extract video ID
    video_id = extract_video_id(args.url)
    if not video_id:
        print(f"Error: Could not parse YouTube URL: {args.url}", file=sys.stderr)
        sys.exit(1)

    print(f"Fetching transcript for video: {video_id}...")

    try:
        transcript = get_transcript(video_id)
        source = "YouTube"
        print(f"Got YouTube transcript ({len(transcript)} chars).")
    except Exception as e:
        print(f"YouTube transcript not available: {e}")
        print("Falling back to Whisper transcription...")
        try:
            transcript = transcribe_with_whisper(args.url)
            source = "Whisper"
            print(f"Got Whisper transcript ({len(transcript)} chars).")
        except Exception as whisper_error:
            print(f"Error: {whisper_error}", file=sys.stderr)
            sys.exit(1)

    # Save transcript for later reference
    save_transcript(video_id, args.url, transcript, source)

    print("Summarizing with Claude...\n")

    try:
        summary = summarize_with_claude(transcript, args.url)
        print("=" * 60)
        print(summary)
        print("=" * 60)
    except anthropic.AuthenticationError:
        print("Error: Missing or invalid ANTHROPIC_API_KEY", file=sys.stderr)
        print("Set it with: export ANTHROPIC_API_KEY='your-key-here'", file=sys.stderr)
        sys.exit(1)
    except anthropic.APIError as e:
        print(f"Error calling Claude API: {e}", file=sys.stderr)
        sys.exit(1)
    except TypeError as e:
        if "api_key" in str(e) or "authentication" in str(e).lower():
            print("Error: Missing ANTHROPIC_API_KEY environment variable", file=sys.stderr)
            print("Set it with: export ANTHROPIC_API_KEY='your-key-here'", file=sys.stderr)
            sys.exit(1)
        raise


if __name__ == "__main__":
    main()
