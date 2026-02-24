"""
Download Kaltura lecture recordings from the export email.

Parses the Kaltura "Your videos are ready" .eml file, reconstructs
download URLs (undoing quoted-printable line wrapping), and fetches
each recording with curl.

Usage (on remote machine):
    python -m stan.lecture.download                          # dry run
    python -m stan.lecture.download --go                     # download all
    python -m stan.lecture.download --go --filter CHEG231-010  # only section 010

Requires: curl (standard on Linux/macOS)
"""
from __future__ import annotations

import argparse
import email
import quopri
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

# Default paths relative to repo root
DEFAULT_EML = Path(__file__).resolve().parent / "Fwd_ Your Kaltura videos are ready.eml"
DEFAULT_OUT = Path(__file__).resolve().parents[1] / "data" / "lectures" / "video"


@dataclass
class KalturaEntry:
    """A single downloadable Kaltura recording."""
    title: str
    entry_id: str
    url: str
    index: int  # sequential within its title group


def parse_eml(eml_path: Path) -> list[KalturaEntry]:
    """Parse the Kaltura export email and extract download entries."""
    raw = eml_path.read_bytes()
    msg = email.message_from_bytes(raw)

    # Get the plain-text body, decode quoted-printable
    body = None
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == "text/plain":
                payload = part.get_payload(decode=True)
                if payload:
                    body = payload.decode("utf-8", errors="replace")
                    break
    else:
        payload = msg.get_payload(decode=True)
        if payload:
            body = payload.decode("utf-8", errors="replace")

    if body is None:
        # Fallback: read raw and decode quoted-printable ourselves
        raw_text = eml_path.read_text(errors="replace")
        body = quopri.decodestring(raw_text.encode()).decode("utf-8", errors="replace")

    # Pattern: title line followed by a URL in angle brackets (possibly across lines)
    # After QP decoding the URLs should be clean, but handle both cases
    entries = []
    lines = body.splitlines()
    title_counts: dict[str, int] = {}

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.endswith(" - Download"):
            title = line.replace(" - Download", "").strip()

            # Next non-empty line(s) should contain the URL in < >
            url_parts = []
            i += 1
            while i < len(lines):
                l = lines[i].strip()
                if not l:
                    i += 1
                    continue
                if l.startswith("<") or url_parts:
                    url_parts.append(l.strip("<>"))
                    if ">" in lines[i]:
                        break
                else:
                    break
                i += 1

            url = "".join(url_parts)
            if url:
                # Extract entry_id from URL
                m = re.search(r"entry_id/([^/]+)", url)
                entry_id = m.group(1) if m else "unknown"

                title_counts[title] = title_counts.get(title, 0) + 1
                entries.append(KalturaEntry(
                    title=title,
                    entry_id=entry_id,
                    url=url,
                    index=title_counts[title],
                ))
        i += 1

    return entries


def download(entry: KalturaEntry, out_dir: Path, dry_run: bool = True) -> Path:
    """Download a single entry. Returns the output file path."""
    # Build filename: title_NNN_entryid.mp4
    safe_title = re.sub(r"[^\w\-]", "_", entry.title)
    filename = f"{safe_title}_{entry.index:03d}_{entry.entry_id}.mp4"
    out_path = out_dir / filename

    if dry_run:
        print(f"  [DRY RUN] {filename}")
        print(f"            {entry.url[:80]}...")
        return out_path

    if out_path.exists():
        print(f"  [SKIP] {filename} (already exists)")
        return out_path

    print(f"  [GET]  {filename}")
    result = subprocess.run(
        ["curl", "-L", "-o", str(out_path), "-#", entry.url],
        check=False,
    )
    if result.returncode != 0:
        print(f"  [FAIL] curl returned {result.returncode}")
    elif out_path.stat().st_size < 1024:
        print(f"  [WARN] File is only {out_path.stat().st_size} bytes — "
              "link may have expired")
    else:
        size_mb = out_path.stat().st_size / (1024 * 1024)
        print(f"  [OK]   {size_mb:.1f} MB")

    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Download Kaltura lecture recordings from export email"
    )
    parser.add_argument(
        "--eml", type=Path, default=DEFAULT_EML,
        help="Path to the .eml file with download links"
    )
    parser.add_argument(
        "--out", type=Path, default=DEFAULT_OUT,
        help="Output directory for downloaded videos"
    )
    parser.add_argument(
        "--go", action="store_true",
        help="Actually download (default is dry run)"
    )
    parser.add_argument(
        "--filter", type=str, default=None,
        help="Only download entries whose title contains this string"
    )
    args = parser.parse_args()

    if not args.eml.exists():
        print(f"Error: {args.eml} not found", file=sys.stderr)
        sys.exit(1)

    entries = parse_eml(args.eml)
    if not entries:
        print("No download entries found in the email.", file=sys.stderr)
        sys.exit(1)

    if args.filter:
        entries = [e for e in entries if args.filter in e.title]

    # Summary
    from collections import Counter
    counts = Counter(e.title for e in entries)
    print(f"Found {len(entries)} recordings:")
    for title, count in counts.items():
        print(f"  {title}: {count}")
    print()

    if not args.go:
        print("Dry run (pass --go to download):\n")

    args.out.mkdir(parents=True, exist_ok=True)

    for entry in entries:
        download(entry, args.out, dry_run=not args.go)

    if not args.go:
        print(f"\nRerun with --go to download to {args.out}")


if __name__ == "__main__":
    main()
