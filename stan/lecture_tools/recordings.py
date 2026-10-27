"""
Track lecture recordings: what each audio file is, when it was saved, and
whether it is what its name says.

For every file in the audio directory this records the section and lecture
number from the filename, the date that lecture met (from
course_schedule.json), the save time embedded in the file (the container's
``creation_time`` tag, which survives copying, unlike the filesystem date),
the duration, and a fingerprint of the decoded audio.  It then flags:

  DUPLICATE      same decoded audio as another file.  A re-saved copy has new
                 metadata and so a different file hash; only the decoded audio
                 shows it is the same recording.
  SAVED EARLY    saved before the lecture its name claims met -- the file
                 cannot be that lecture.
  MISSING        a lecture that met (per the schedule) with no file; meeting
                 days listed under "no_recording" (exams) are not reported.

The save time is when the file was written, which may be days after the
lecture (several lectures are often saved in one sitting), so a late save is
normal and not flagged.

The report is written as Markdown (for reading) and CSV (for tools) next to
the audio directory.

Usage:
    python -m stan.lecture_tools.recordings                 # report to stdout
    python -m stan.lecture_tools.recordings --write         # also write recordings.md/.csv
    python -m stan.lecture_tools.recordings --audio-dir ~/Desktop/recorded-lectures/audio
    python -m stan.lecture_tools.recordings --write --strict   # exit 1 if any file is flagged

Requires ffprobe and ffmpeg on the PATH.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from .analyze import LectureID, extract_lecture_id, lecture_instructor

LECTURES_DIR = Path(__file__).resolve().parents[1] / "data" / "lectures"
DEFAULT_AUDIO_DIR = LECTURES_DIR / "audio"
SCHEDULE_FILE = LECTURES_DIR / "course_schedule.json"

AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a"}

# Seconds of audio hashed for the fingerprint.  Two minutes of a lecture is
# effectively unique, and decoding it takes about a second per file.
FINGERPRINT_SECONDS = 120

_WEEKDAYS = {"Mon": 0, "Tue": 1, "Wed": 2, "Thu": 3, "Fri": 4, "Sat": 5, "Sun": 6}


# ── Schedule ─────────────────────────────────────────────────────────

@dataclass
class Schedule:
    first_day: date
    meeting_days: set[int]
    no_class: dict[date, str]
    tz: ZoneInfo
    no_recording: dict[date, str] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path = SCHEDULE_FILE) -> "Schedule":
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Copy course_schedule.example.json (same "
                f"directory) to {path.name} and edit it for your course.")
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            first_day=date.fromisoformat(data["first_day"]),
            meeting_days={_WEEKDAYS[d] for d in data["meeting_days"]},
            no_class={date.fromisoformat(k): v for k, v in data.get("no_class", {}).items()},
            tz=ZoneInfo(data.get("timezone", "America/New_York")),
            no_recording={date.fromisoformat(k): v
                          for k, v in data.get("no_recording", {}).items()},
        )

    def lecture_dates(self, through: date) -> list[date]:
        """Meeting dates from first_day through ``through``; index 0 is lecture 1."""
        dates, day = [], self.first_day
        while day <= through:
            if day.weekday() in self.meeting_days and day not in self.no_class:
                dates.append(day)
            day += timedelta(days=1)
        return dates


# ── Probing ──────────────────────────────────────────────────────────

@dataclass
class Recording:
    path: Path
    lid: LectureID | None
    saved: datetime | None
    duration_s: float | None
    size: int
    fingerprint: str
    lecture_date: date | None = None
    flags: list[str] = field(default_factory=list)


def _probe(path: Path) -> tuple[datetime | None, float | None]:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries",
         "format=duration:format_tags=creation_time", "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    )
    fmt = json.loads(out.stdout).get("format", {})
    created = fmt.get("tags", {}).get("creation_time")
    saved = datetime.fromisoformat(created.replace("Z", "+00:00")) if created else None
    duration = float(fmt["duration"]) if "duration" in fmt else None
    return saved, duration


def _fingerprint(path: Path) -> str:
    pcm = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-t", str(FINGERPRINT_SECONDS),
         "-f", "s16le", "-ac", "1", "-ar", "16000", "-"],
        capture_output=True, check=True,
    ).stdout
    return hashlib.md5(pcm).hexdigest()[:12]


def scan(audio_dir: Path, schedule: Schedule, today: date) -> tuple[list[Recording], list[str]]:
    """Probe every recording and flag problems.  Returns (recordings, missing)."""
    files = sorted(p for p in audio_dir.iterdir() if p.suffix.lower() in AUDIO_EXTENSIONS)
    dates = schedule.lecture_dates(today)
    recs = []
    for path in files:
        saved, duration = _probe(path)
        rec = Recording(path, extract_lecture_id(path.name), saved, duration,
                        path.stat().st_size, _fingerprint(path))
        if rec.lid and rec.lid.section is not None and 1 <= rec.lid.number <= len(dates):
            rec.lecture_date = dates[rec.lid.number - 1]
        recs.append(rec)

    by_print: dict[str, list[Recording]] = {}
    for rec in recs:
        by_print.setdefault(rec.fingerprint, []).append(rec)
    for group in by_print.values():
        if len(group) > 1:
            for rec in group:
                others = ", ".join(r.path.name for r in group if r is not rec)
                rec.flags.append(f"DUPLICATE of {others}")

    for rec in recs:
        if rec.lid and rec.lid.section is not None:
            if rec.lid.number > len(dates):
                rec.flags.append(f"NOT IN SCHEDULE (only {len(dates)} lectures through {today})")
            elif rec.saved and rec.lecture_date:
                saved_local = rec.saved.astimezone(schedule.tz).date()
                if saved_local < rec.lecture_date:
                    rec.flags.append(f"SAVED EARLY (before {rec.lecture_date:%a %b %-d})")

    sections = sorted({r.lid.section for r in recs if r.lid and r.lid.section})
    have = {(r.lid.section, r.lid.number) for r in recs if r.lid and r.lid.section}
    missing = [f"{s}-lecture-{n:02d} ({d:%a %b %-d})"
               for s in sections for n, d in enumerate(dates, 1)
               if (s, n) not in have and d not in schedule.no_recording]
    return recs, missing


# ── Report ───────────────────────────────────────────────────────────

def _fmt_duration(seconds: float | None) -> str:
    if seconds is None:
        return ""
    m, s = divmod(round(seconds), 60)
    return f"{m // 60}:{m % 60:02d}:{s:02d}"


def _rows(recs: list[Recording], tz: ZoneInfo) -> list[dict]:
    def key(r: Recording):
        return (0, r.lid.sort_key) if r.lid else (1, ("", 0), r.path.name)
    rows = []
    for r in sorted(recs, key=key):
        rows.append({
            "file": r.path.name,
            "section": r.lid.section if r.lid and r.lid.section else "",
            "lecture": r.lid.number if r.lid else "",
            "lecture_date": f"{r.lecture_date:%Y-%m-%d %a}" if r.lecture_date else "",
            "instructor": (lecture_instructor(r.lid) or "") if r.lid else "",
            "saved": f"{r.saved.astimezone(tz):%Y-%m-%d %a %H:%M}" if r.saved else "",
            "duration": _fmt_duration(r.duration_s),
            "size_mb": f"{r.size / 1e6:.1f}",
            "audio_fingerprint": r.fingerprint,
            "flags": "; ".join(r.flags),
        })
    return rows


def render_markdown(rows: list[dict], missing: list[str], audio_dir: Path,
                    today: date, schedule_file: Path) -> str:
    flagged = [r for r in rows if r["flags"]]
    lines = [
        "# Lecture recordings",
        "",
        f"Generated {today:%Y-%m-%d} by `python -m stan.lecture_tools.recordings` "
        f"from `{audio_dir}`. Lecture dates come from `{schedule_file.name}`; "
        "**Saved** is the file's embedded `creation_time` (when the recording "
        "was written, often days after the lecture). The audio fingerprint "
        "hashes the decoded audio, so a re-saved copy matches its original.",
        "",
        f"**{len(rows)} files, {len(flagged)} flagged, {len(missing)} lectures missing.**",
        "",
    ]
    if flagged or missing:
        lines += ["## Problems", ""]
        lines += [f"- `{r['file']}`: {r['flags']}" for r in flagged]
        lines += [f"- MISSING: {m}" for m in missing]
        lines += [""]
    lines += [
        "## All recordings",
        "",
        "| File | Section | Lecture | Lecture date | Instructor | Saved | Duration | MB | Audio fingerprint | Flags |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    lines += [f"| `{r['file']}` | {r['section']} | {r['lecture']} | {r['lecture_date']} | {r['instructor']} | "
              f"{r['saved']} | {r['duration']} | {r['size_mb']} | `{r['audio_fingerprint']}` | "
              f"{r['flags']} |" for r in rows]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Track lecture recordings and flag mix-ups")
    parser.add_argument("--audio-dir", type=Path, default=DEFAULT_AUDIO_DIR)
    parser.add_argument("--schedule", type=Path, default=SCHEDULE_FILE)
    parser.add_argument("--write", action="store_true",
                        help="Write recordings.md and recordings.csv beside the audio directory")
    parser.add_argument("--strict", action="store_true",
                        help="Exit with status 1 if any file is flagged (missing lectures "
                             "alone do not fail), so a script can stop before transcribing")
    args = parser.parse_args()

    audio_dir = args.audio_dir.expanduser()
    if not audio_dir.is_dir():
        print(f"Error: {audio_dir} is not a directory", file=sys.stderr)
        sys.exit(1)
    try:
        schedule = Schedule.load(args.schedule)
    except FileNotFoundError as e:
        sys.exit(f"Error: {e}")
    today = datetime.now(schedule.tz).date()

    recs, missing = scan(audio_dir, schedule, today)
    rows = _rows(recs, schedule.tz)
    md = render_markdown(rows, missing, audio_dir, today, args.schedule)
    print(md)

    if args.write:
        out_dir = audio_dir.parent
        (out_dir / "recordings.md").write_text(md, encoding="utf-8")
        with open(out_dir / "recordings.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["file"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {out_dir / 'recordings.md'} and {out_dir / 'recordings.csv'}")

    if args.strict and any(r["flags"] for r in rows):
        sys.exit(1)


if __name__ == "__main__":
    main()
