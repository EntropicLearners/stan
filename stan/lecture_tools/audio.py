"""
Extract audio from lecture video recordings using ffmpeg.

Converts MP4 (or other video) files to WAV for Whisper transcription.
WAV is preferred over MP3 for transcription accuracy (lossless).

Usage:
    python -m stan.lecture.audio                              # dry run
    python -m stan.lecture.audio --go                         # extract all
    python -m stan.lecture.audio --go --format mp3            # smaller files
    python -m stan.lecture.audio --go --filter CHEG231-010    # only section 010

Requires: ffmpeg (sudo apt install ffmpeg)
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

# Default paths relative to repo root
DEFAULT_VIDEO_DIR = Path(__file__).resolve().parents[1] / "data" / "lectures" / "video"
DEFAULT_AUDIO_DIR = Path(__file__).resolve().parents[1] / "data" / "lectures" / "audio"

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".webm", ".avi", ".mov"}

# Audio format presets
FORMATS = {
    "wav": {
        "ext": ".wav",
        "args": ["-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1"],
        "desc": "16kHz mono WAV (best for Whisper, ~58 MB/hr)",
    },
    "mp3": {
        "ext": ".mp3",
        "args": ["-vn", "-acodec", "libmp3lame", "-ar", "16000", "-ac", "1", "-b:a", "64k"],
        "desc": "16kHz mono MP3 (smaller, ~29 MB/hr)",
    },
}


def find_videos(video_dir: Path, filter_str: str | None = None) -> list[Path]:
    """Find all video files in the directory."""
    videos = sorted(
        p for p in video_dir.iterdir()
        if p.suffix.lower() in VIDEO_EXTENSIONS
    )
    if filter_str:
        videos = [v for v in videos if filter_str in v.name]
    return videos


def extract_audio(
    video_path: Path,
    audio_dir: Path,
    fmt: str = "wav",
    dry_run: bool = True,
    overwrite: bool = False,
) -> Path | None:
    """Extract audio from a single video file."""
    preset = FORMATS[fmt]
    out_path = audio_dir / (video_path.stem + preset["ext"])

    if dry_run:
        print(f"  [DRY RUN] {video_path.name} -> {out_path.name}")
        return out_path

    if out_path.exists() and not overwrite:
        print(f"  [SKIP]    {out_path.name} (already exists)")
        return out_path

    print(f"  [EXTRACT] {video_path.name} -> {out_path.name}")
    result = subprocess.run(
        ["ffmpeg", "-i", str(video_path), *preset["args"], "-y", str(out_path)],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(f"  [FAIL]    ffmpeg returned {result.returncode}")
        # Show last few lines of stderr for diagnosis
        for line in result.stderr.strip().splitlines()[-3:]:
            print(f"            {line}")
        return None

    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"  [OK]      {size_mb:.1f} MB")
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Extract audio from lecture video recordings"
    )
    parser.add_argument(
        "--video-dir", type=Path, default=DEFAULT_VIDEO_DIR,
        help="Directory containing video files"
    )
    parser.add_argument(
        "--audio-dir", type=Path, default=DEFAULT_AUDIO_DIR,
        help="Output directory for audio files"
    )
    parser.add_argument(
        "--format", choices=FORMATS.keys(), default="wav",
        help="Audio format (default: wav)"
    )
    parser.add_argument(
        "--go", action="store_true",
        help="Actually extract (default is dry run)"
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite existing audio files"
    )
    parser.add_argument(
        "--filter", type=str, default=None,
        help="Only process videos whose filename contains this string"
    )
    args = parser.parse_args()

    # Check ffmpeg
    if args.go and not shutil.which("ffmpeg"):
        print("Error: ffmpeg not found. Install with: sudo apt install ffmpeg",
              file=sys.stderr)
        sys.exit(1)

    if not args.video_dir.exists():
        print(f"Error: {args.video_dir} not found", file=sys.stderr)
        sys.exit(1)

    videos = find_videos(args.video_dir, args.filter)
    if not videos:
        print("No video files found.", file=sys.stderr)
        sys.exit(1)

    preset = FORMATS[args.format]
    print(f"Found {len(videos)} video(s)")
    print(f"Format: {preset['desc']}")
    print()

    if not args.go:
        print("Dry run (pass --go to extract):\n")

    args.audio_dir.mkdir(parents=True, exist_ok=True)

    success = 0
    for video in videos:
        result = extract_audio(
            video, args.audio_dir, args.format,
            dry_run=not args.go, overwrite=args.overwrite,
        )
        if result is not None:
            success += 1

    print(f"\n{success}/{len(videos)} completed")

    if not args.go:
        print(f"\nRerun with --go to extract audio to {args.audio_dir}")


if __name__ == "__main__":
    main()
