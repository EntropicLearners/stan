"""
Catalog and validate manually downloaded lecture recordings.

Since Canvas API access is not available, lecture recordings are downloaded
manually from UDCapture/Kaltura via the browser. This module provides tools
to catalog those files and prepare them for the processing pipeline.

Expected directory layout:
    stan/data/lectures/
    ├── lecture_01.mp4
    ├── lecture_02.mp4
    └── ...

Usage:
    from stan.lecture.acquire import LectureCatalog

    catalog = LectureCatalog("stan/data/lectures")
    catalog.summary()

    # As CLI
    python -m stan.lecture.acquire stan/data/lectures
"""

import os
from pathlib import Path
from dataclasses import dataclass, field

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".webm", ".avi", ".mov"}
AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}
MEDIA_EXTENSIONS = VIDEO_EXTENSIONS | AUDIO_EXTENSIONS


@dataclass
class LectureFile:
    """Metadata for a single lecture recording."""
    path: Path
    size_bytes: int
    media_type: str  # "video" or "audio"

    @property
    def size_mb(self):
        return self.size_bytes / (1024 * 1024)

    @property
    def name(self):
        return self.path.name


class LectureCatalog:
    """Catalog of lecture recordings in a directory.

    Scans a directory for video/audio files and provides summary
    information to verify downloads before processing.
    """

    def __init__(self, lectures_dir):
        self.lectures_dir = Path(lectures_dir)
        self.files = self._scan()

    def _scan(self):
        """Scan the lectures directory for media files."""
        if not self.lectures_dir.exists():
            print(f"Directory not found: {self.lectures_dir}")
            return []

        files = []
        for p in sorted(self.lectures_dir.iterdir()):
            if p.suffix.lower() in MEDIA_EXTENSIONS:
                media_type = (
                    "video" if p.suffix.lower() in VIDEO_EXTENSIONS else "audio"
                )
                files.append(LectureFile(
                    path=p,
                    size_bytes=p.stat().st_size,
                    media_type=media_type,
                ))
        return files

    @property
    def video_files(self):
        return [f for f in self.files if f.media_type == "video"]

    @property
    def audio_files(self):
        return [f for f in self.files if f.media_type == "audio"]

    def summary(self):
        """Print a summary of available lecture files."""
        if not self.files:
            print(f"No media files found in {self.lectures_dir}")
            return

        total_mb = sum(f.size_mb for f in self.files)
        print(f"Lecture catalog: {self.lectures_dir}")
        print(f"  {len(self.video_files)} video files, "
              f"{len(self.audio_files)} audio files")
        print(f"  Total size: {total_mb:.1f} MB")
        print()
        for f in self.files:
            print(f"  {f.media_type:5s}  {f.size_mb:8.1f} MB  {f.name}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Catalog lecture recordings in a directory"
    )
    parser.add_argument(
        "lectures_dir", nargs="?", default="stan/data/lectures",
        help="Path to lectures directory (default: stan/data/lectures)"
    )
    args = parser.parse_args()

    catalog = LectureCatalog(args.lectures_dir)
    catalog.summary()


if __name__ == "__main__":
    main()
