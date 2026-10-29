"""
Lecture processing tools for UDCapture/Kaltura recordings.

Pipeline stages:
    acquire     - Discover and download lectures from Canvas/Kaltura
    audio       - Extract audio from video recordings (ffmpeg)
    recordings  - Track recordings: save dates, durations, duplicates, gaps
    voicememos  - Import recordings from Apple Voice Memos with real dates (Mac)
    recap       - Student-facing lecture recap (Markdown) for Canvas
    transcribe  - Speech-to-text transcription (Whisper)
    segment     - Break transcripts into searchable chunks
    analyze     - Extract questions, confusion points, summaries
"""

from . import acquire
from . import audio

# segment requires optional [lecture] dependencies (llama-index, etc.)
# Import explicitly: from stan.lecture_tools import segment
