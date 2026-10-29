"""
Refresh the notation primers in stan/data/grounding/ from cbethermo.org.

cbethermo.org is the source of truth for the primers; the copies in this repo
are a snapshot so prompts are reproducible and work offline.  Re-run this when
the site's primers change, review the diff, and commit it.

Usage:
    python -m stan.notation.fetch              # dry run: report what would change
    python -m stan.notation.fetch --go         # download and overwrite
"""
from __future__ import annotations

import argparse
import sys
import urllib.error
import urllib.request

from .primer import GROUNDING_DIR, primer_version

BASE_URL = "https://cbethermo.org/grounding"
CHAPTERS = range(1, 16)


def fetch(chapter: int) -> bytes:
    url = f"{BASE_URL}/ch{chapter:02d}-notation.md"
    with urllib.request.urlopen(url, timeout=30) as resp:
        return resp.read()


def main():
    parser = argparse.ArgumentParser(description="Refresh notation primers from cbethermo.org")
    parser.add_argument("--go", action="store_true", help="Write the files (default is dry run)")
    args = parser.parse_args()

    GROUNDING_DIR.mkdir(parents=True, exist_ok=True)
    changed = 0
    for chapter in CHAPTERS:
        path = GROUNDING_DIR / f"ch{chapter:02d}-notation.md"
        try:
            new = fetch(chapter)
        except urllib.error.URLError as e:
            print(f"  [FAIL]      ch{chapter:02d}: {e}", file=sys.stderr)
            continue
        old = path.read_bytes() if path.exists() else None
        if new == old:
            print(f"  [SAME]      {path.name}")
            continue
        changed += 1
        status = "NEW" if old is None else "CHANGED"
        print(f"  [{status}]{' ' * (10 - len(status))}{path.name}")
        if args.go:
            path.write_bytes(new)

    if not args.go and changed:
        print("\nRerun with --go to write the changes.")
    elif args.go:
        print(f"\nPrimers now {primer_version()}.")


if __name__ == "__main__":
    main()
