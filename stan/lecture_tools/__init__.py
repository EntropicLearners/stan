"""
Lecture processing tools for UDCapture/Kaltura recordings.

Pipeline stages:
    acquire     - Discover and download lectures from Canvas/Kaltura
    audio       - Extract audio from video recordings (ffmpeg)
    transcribe  - Speech-to-text transcription (Whisper)
    segment     - Break transcripts into searchable chunks
    analyze     - Extract questions, confusion points, summaries
"""

from . import acquire
from . import audio

# segment requires optional [lecture] dependencies (llama-index, etc.)
# Import explicitly: from stan.lecture import segment
