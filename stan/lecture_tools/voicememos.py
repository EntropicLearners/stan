"""
Import lecture recordings from Apple Voice Memos with their real dates.

Sharing a recording out of Voice Memos (AirDrop, Taildrop, Save to Files)
writes a new copy stamped with the time of sharing, so the recording date is
lost.  With Voice Memos syncing through iCloud, the Mac holds the ORIGINAL
files, named by recording start time ("20261002 113057-29C02ADB.m4a"), plus a
database (CloudRecordings.db) with each recording's start, duration, title,
and folder.  This reads one Voice Memos folder (default "Class") and copies
its recordings into the audio directory under the names the pipeline expects:

  section   from the start time, matched to a slot in course_schedule.json
            (11:30 -> 010, 12:40 -> 011, 15:00 Fridays -> honors)
  lecture   from the date: the Nth meeting day in course_schedule.json
            (numbered slots -> 010-lecture-15.m4a; others -> honors-2026-10-02.m4a)

Each copy keeps the original bytes and gets its file date set to the
recording start.  A recording is skipped when it is shorter than
--min-minutes (false starts), when the audio directory already holds a file
of the same duration saved no earlier than the recording started (already
imported or shared earlier, possibly under another name -- which is
reported).  Duration alone is not enough: with dozens of 50-60 minute
lectures, two land within a second of each other (Oct 5 matched Aug 28).  A
file saved before a recording began cannot be a copy of it; originals carry
their recording start as creation_time, shared copies their sharing time, or when it does not fall in a slot on a
class day (reported, never guessed).

This script uses only the standard library (plus ffprobe, if present, to read
durations of files already in the audio directory) so it runs on a Mac
without the stan environment.  Reading the Voice Memos folder needs Full Disk
Access for the app running it (System Settings > Privacy & Security).

Usage:
    python3 stan/lecture_tools/voicememos.py --audio-dir ~/lectures/audio          # dry run
    python3 stan/lecture_tools/voicememos.py --audio-dir ~/lectures/audio --go     # copy
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

VOICE_MEMOS_DIR = (Path.home() / "Library" / "Group Containers"
                   / "group.com.apple.VoiceMemos.shared" / "Recordings")
SCHEDULE_FILE = Path(__file__).resolve().parents[1] / "data" / "lectures" / "course_schedule.json"
COCOA_EPOCH = datetime(2001, 1, 1, tzinfo=timezone.utc)  # Core Data timestamps count from here
DURATION_TOLERANCE_S = 1.5
# A copy cannot be saved before its recording started; allow for clock skew.
SAVED_BEFORE_GRACE = timedelta(minutes=2)
WEEKDAYS = {"Mon": 0, "Tue": 1, "Wed": 2, "Thu": 3, "Fri": 4, "Sat": 5, "Sun": 6}


@dataclass
class Memo:
    start: datetime          # local time
    duration: float          # seconds
    title: str
    path: Path


@dataclass
class Plan:
    memo: Memo
    target: str | None       # file name to create, or None if skipped
    note: str


# ── Inputs ───────────────────────────────────────────────────────────

def read_memos(vm_dir: Path, folder: str, tz: ZoneInfo) -> list[Memo]:
    """Recordings in one Voice Memos folder, oldest first.

    The database is copied (with its -wal/-shm journal) to a temporary
    directory first, so Voice Memos is never locked or modified.
    """
    with tempfile.TemporaryDirectory() as tmp:
        for name in ("CloudRecordings.db", "CloudRecordings.db-wal", "CloudRecordings.db-shm"):
            src = vm_dir / name
            if src.exists():
                shutil.copy2(src, Path(tmp) / name)
        db = sqlite3.connect(Path(tmp) / "CloudRecordings.db")
        row = db.execute("SELECT Z_PK FROM ZFOLDER WHERE ZENCRYPTEDNAME = ?", (folder,)).fetchone()
        if row is None:
            names = [r[0] for r in db.execute("SELECT ZENCRYPTEDNAME FROM ZFOLDER")]
            raise SystemExit(f"No Voice Memos folder named {folder!r} (found: {names})")
        rows = db.execute(
            "SELECT ZDATE, ZDURATION, ZENCRYPTEDTITLE, ZPATH FROM ZCLOUDRECORDING "
            "WHERE ZFOLDER = ? ORDER BY ZDATE", (row[0],)
        ).fetchall()
        db.close()
    return [Memo((COCOA_EPOCH + timedelta(seconds=d)).astimezone(tz), dur or 0.0,
                 title or "", vm_dir / path) for d, dur, title, path in rows]


def lecture_dates(schedule: dict, through: date) -> list[date]:
    """Meeting dates; index 0 is lecture 1 (same rule as recordings.py)."""
    first = date.fromisoformat(schedule["first_day"])
    days = {WEEKDAYS[d] for d in schedule["meeting_days"]}
    skip = {date.fromisoformat(d) for d in schedule.get("no_class", {})}
    out, day = [], first
    while day <= through:
        if day.weekday() in days and day not in skip:
            out.append(day)
        day += timedelta(days=1)
    return out


@dataclass
class Existing:
    duration: float
    saved: datetime           # embedded creation_time, else file mtime (UTC)


def existing_audio(audio_dir: Path) -> dict[str, Existing]:
    """Duration and save time of audio already in the directory (needs ffprobe)."""
    if shutil.which("ffprobe") is None:
        return {}
    out = {}
    for p in sorted(audio_dir.glob("*")):
        if p.suffix.lower() not in (".m4a", ".wav", ".mp3"):
            continue
        r = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                            "format=duration:format_tags=creation_time", "-of", "json", str(p)],
                           capture_output=True, text=True)
        try:
            fmt = json.loads(r.stdout)["format"]
            duration = float(fmt["duration"])
        except (ValueError, KeyError):
            continue
        tag = fmt.get("tags", {}).get("creation_time")
        saved = (datetime.fromisoformat(tag.replace("Z", "+00:00")) if tag
                 else datetime.fromtimestamp(p.stat().st_mtime, timezone.utc))
        out[p.name] = Existing(duration, saved)
    return out


# ── Planning ─────────────────────────────────────────────────────────

def plan(memos: list[Memo], schedule: dict, audio_dir: Path, min_minutes: float) -> list[Plan]:
    tz = ZoneInfo(schedule.get("timezone", "America/New_York"))
    today = datetime.now(tz).date()
    dates = lecture_dates(schedule, today)
    tolerance = timedelta(minutes=schedule.get("slot_tolerance_minutes", 40))
    have = existing_audio(audio_dir)
    plans = []
    for m in memos:
        if m.duration < min_minutes * 60:
            plans.append(Plan(m, None, f"skip: {m.duration / 60:.1f} min (false start?)"))
            continue
        slot = None
        for s in schedule["slots"]:
            hh, mm = map(int, s["start"].split(":"))
            nominal = m.start.replace(hour=hh, minute=mm, second=0, microsecond=0)
            if m.start.strftime("%a") in s["days"] and abs(m.start - nominal) <= tolerance:
                slot = s
                break
        if slot is None:
            plans.append(Plan(m, None, "skip: start time matches no slot in course_schedule.json"))
            continue
        if slot.get("numbered"):
            if m.start.date() not in dates:
                plans.append(Plan(m, None, f"skip: {m.start:%a %b %-d} is not a class day "
                                           "in course_schedule.json"))
                continue
            target = f"{slot['name']}-lecture-{dates.index(m.start.date()) + 1:02d}.m4a"
        else:
            target = f"{slot['name']}-{m.start:%Y-%m-%d}.m4a"

        same = [n for n, e in have.items()
                if abs(e.duration - m.duration) <= DURATION_TOLERANCE_S
                and e.saved >= m.start - SAVED_BEFORE_GRACE]
        if same:
            if target in same:
                plans.append(Plan(m, None, f"present: {target}"))
            elif not slot.get("numbered"):
                # Date-named slots have no number to get wrong; an older name is fine.
                plans.append(Plan(m, None, f"present: {', '.join(same)}"))
            else:
                plans.append(Plan(m, None, f"MISMATCH: this is {target} by date and slot, but the "
                                           f"same audio is already here as {', '.join(same)}"))
            continue
        if (audio_dir / target).exists():
            plans.append(Plan(m, None, f"CONFLICT: {target} exists with a different duration"))
            continue
        if not m.path.exists():
            plans.append(Plan(m, None, f"skip: audio not downloaded yet ({m.path.name}); "
                                       "open it in Voice Memos on the Mac"))
            continue
        plans.append(Plan(m, target, "import"))
    return plans


def main():
    parser = argparse.ArgumentParser(description="Import lecture recordings from Voice Memos")
    parser.add_argument("--audio-dir", type=Path, required=True,
                        help="Where lecture audio lives, e.g. ~/lectures/audio")
    parser.add_argument("--folder", default="Class", help="Voice Memos folder (default: Class)")
    parser.add_argument("--schedule", type=Path, default=SCHEDULE_FILE)
    parser.add_argument("--voice-memos-dir", type=Path, default=VOICE_MEMOS_DIR)
    parser.add_argument("--min-minutes", type=float, default=5.0,
                        help="Skip recordings shorter than this (default 5)")
    parser.add_argument("--go", action="store_true", help="Copy files (default is dry run)")
    args = parser.parse_args()

    audio_dir = args.audio_dir.expanduser()
    if not args.schedule.exists():
        raise SystemExit(f"{args.schedule} not found. Copy course_schedule.example.json "
                         f"(same directory) to {args.schedule.name} and edit it for your course.")
    schedule = json.loads(args.schedule.read_text(encoding="utf-8"))
    tz = ZoneInfo(schedule.get("timezone", "America/New_York"))
    try:
        memos = read_memos(args.voice_memos_dir, args.folder, tz)
    except PermissionError:
        raise SystemExit("Cannot read the Voice Memos folder: give this app Full Disk Access "
                         "(System Settings > Privacy & Security > Full Disk Access).")

    plans = plan(memos, schedule, audio_dir, args.min_minutes)
    for p in plans:
        m = p.memo
        arrow = f"-> {p.target}" if p.target else ""
        print(f"{m.start:%Y-%m-%d %a %H:%M}  {m.duration / 60:5.1f} min  {m.title[:22]:22s}  "
              f"{p.note:8s} {arrow}")

    todo = [p for p in plans if p.target]
    problems = [p for p in plans if p.note.startswith(("MISMATCH", "CONFLICT"))]
    print(f"\n{len(memos)} recordings in {args.folder!r}: {len(todo)} to import, "
          f"{len(problems)} problem(s)")
    if not args.go:
        if todo:
            print("Dry run; rerun with --go to copy.")
        sys.exit(1 if problems else 0)

    audio_dir.mkdir(parents=True, exist_ok=True)
    for p in todo:
        dest = audio_dir / p.target
        shutil.copy2(p.memo.path, dest)
        ts = p.memo.start.timestamp()
        os.utime(dest, (ts, ts))      # file date = recording start
        print(f"copied {p.memo.path.name} -> {dest}")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
