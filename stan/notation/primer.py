"""
Load the textbook notation primers and build prompt grounding from them.

The primers (``stan/data/grounding/chNN-notation.md``) are the chapter notation
files published on cbethermo.org.  Each one is cumulative: chapter N repeats
every convention and symbol from chapters 1..N-1 and appends its own.  This
module reads all of them and attributes each convention and symbol to the
chapter that introduced it, so a prompt can carry the core conventions plus
only what the relevant chapter added, instead of a whole cumulative primer
(up to ~13k words for chapter 15).

The primers are written against the 6th edition, but the notation itself is
shared with the 5th edition and the chapter numbering and titles match.  What
differs between editions is equation, section, table, figure, and illustration
numbering, so ``edition_neutral()`` rewrites those references to chapter
references (or drops them) before the text goes into a prompt.

Usage:
    from stan.notation import grounding_block
    system += grounding_block(chapters=[6], text=query)

    python -m stan.notation.primer 6            # print the block for chapter 6
    python -m stan.notation.primer 6 --text "departure functions for enthalpy"
"""
from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

GROUNDING_DIR = Path(__file__).resolve().parents[1] / "data" / "grounding"

# Chapters 1-4 hold the conventions every answer depends on: underbar/caret/
# overbar, the sign convention for Q and W, and S_gen >= 0 as the second law.
# They are always included, whatever chapter a prompt is about.
CORE_CHAPTERS = (1, 2, 3, 4)

BOOK = ('Sandler, "Chemical, Biochemical, and Engineering Thermodynamics" '
        '(5th edition; 6th edition with Furst)')


@dataclass(frozen=True)
class Convention:
    chapter: int
    text: str


@dataclass(frozen=True)
class Symbol:
    chapter: int
    latex: str
    meaning: str
    units: str


# ── Parsing ──────────────────────────────────────────────────────────

def _section(text: str, name: str) -> str:
    match = re.search(rf"^## {re.escape(name)}\n(.*?)(?=^## |\Z)", text, re.S | re.M)
    return match.group(1) if match else ""


def _bullets(section: str) -> list[str]:
    items = re.split(r"\n(?=- )", section.strip())
    return [" ".join(b[2:].split()) for b in items if b.startswith("- ")]


def _symbol_rows(section: str) -> list[tuple[str, str, str]]:
    rows = []
    for line in section.splitlines():
        if not line.startswith("| `"):
            continue
        # | `latex` | meaning | units |  -- meanings may contain "|" only inside
        # LaTeX, which the primers do not do, so a plain split is safe.
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) >= 3:
            rows.append((cells[0].strip("`"), cells[1], cells[2]))
    return rows


@lru_cache(maxsize=None)
def load(grounding_dir: Path = GROUNDING_DIR) -> tuple[tuple[Convention, ...], tuple[Symbol, ...]]:
    """Parse every chapter primer, attributing items to their first chapter."""
    files = sorted(grounding_dir.glob("ch[0-9][0-9]-notation.md"))
    if not files:
        raise FileNotFoundError(
            f"No notation primers in {grounding_dir}; "
            f"run python -m stan.notation.fetch --go"
        )
    conventions: list[Convention] = []
    symbols: list[Symbol] = []
    seen_c: set[str] = set()
    seen_s: set[tuple[str, str]] = set()
    for path in files:
        chapter = int(path.name[2:4])
        text = path.read_text(encoding="utf-8")
        for bullet in _bullets(_section(text, "Conventions")):
            if bullet not in seen_c:
                seen_c.add(bullet)
                conventions.append(Convention(chapter, bullet))
        for latex, meaning, units in _symbol_rows(_section(text, "Symbols")):
            if (latex, meaning) not in seen_s:
                seen_s.add((latex, meaning))
                symbols.append(Symbol(chapter, latex, meaning, units))
    return tuple(conventions), tuple(symbols)


# ── Edition-neutral text ─────────────────────────────────────────────

_REF_WORDS = (r"Eqs?|Secs?|Sections?|Illustrations?|Tables?|Figs?|Figures?"
              r"|App|Appendix|Problems?")
_NUM = r"A?\d+(?:\.\d+)*(?:-\d+)?[a-z]?"
_NUM_LIST = rf"{_NUM}(?:\s*(?:,|and|to|&)\s*{_NUM})*"

# A parenthetical made only of references, e.g. "(Eq. 4.2-6)" or
# "(Illustrations 4.1-1, 4.1-2)" or "(Sec. 9.1, App. A9.2)".
_PAREN_REFS = re.compile(
    rf"\s*\((?:see\s+)?(?:(?:{_REF_WORDS})\.?\s*{_NUM_LIST}(?:\s*[,;]\s*|\s+and\s+)?)+\)"
)
_INLINE = [
    (re.compile(rf"\b(?:Illustrations?)\s+(\d+)\.{_NUM_LIST}"), r"a Ch. \1 illustration"),
    (re.compile(rf"\b(?:Tables?)\s+(\d+)\.{_NUM_LIST}"), r"a Ch. \1 table"),
    (re.compile(rf"\b(?:Figs?\.|Figures?)\s*(\d+)\.{_NUM_LIST}"), r"a Ch. \1 figure"),
    (re.compile(rf"\b(?:Eqs?\.)\s*(\d+)\.{_NUM_LIST}"), r"a Ch. \1 equation"),
    (re.compile(rf"\b(?:Secs?\.|Sections?)\s*(\d+)\.{_NUM_LIST}"), r"Ch. \1"),
    (re.compile(rf"\b(?:App\.|Appendix)\s*{_NUM_LIST}"), "the appendix"),
    (re.compile(r"\bCh\.\s*(\d+)"), r"Ch. \1"),
]


def edition_neutral(text: str) -> str:
    """Rewrite 6th-edition-specific references so the text holds for 5e too.

    Chapter references are kept (chapter numbering matches between editions);
    equation, section, table, figure, and illustration numbers are dropped
    when they stand alone in parentheses and reduced to their chapter
    otherwise.
    """
    text = _PAREN_REFS.sub("", text)
    for pattern, repl in _INLINE:
        text = pattern.sub(repl, text)
    return text


# ── Selection ────────────────────────────────────────────────────────

_STOPWORDS = {
    "about", "after", "also", "because", "being", "between", "could", "does",
    "each", "from", "have", "help", "into", "mean", "means", "more", "much",
    "need", "only", "other", "same", "should", "some", "than", "that", "their",
    "them", "then", "there", "these", "they", "this", "those", "through",
    "understand", "what", "when", "where", "which", "while", "with", "would",
    "your", "read", "page", "pages", "chapter", "section", "topic", "topics",
    "textbook", "index", "relevant", "system", "process", "rate", "total",
}


def _content_words(text: str) -> set[str]:
    words = {w.lower() for w in re.findall(r"[A-Za-z][A-Za-z\-]{3,}", text)}
    stems = set()
    for w in words - _STOPWORDS:
        stems.add(w)
        stems.add(w.rstrip("s"))
    return stems


def chapters_in_scope(chapters) -> list[int]:
    """Core chapters plus the requested ones, sorted and de-duplicated."""
    return sorted(set(CORE_CHAPTERS) | {int(c) for c in chapters or ()})


def conventions_for(chapters=()) -> list[Convention]:
    scope = set(chapters_in_scope(chapters))
    conventions, _ = load()
    return [c for c in conventions if c.chapter in scope]


def symbols_for(chapters=(), text: str | None = None,
                limit: int | None = None) -> list[Symbol]:
    """Symbols from the chapters in scope, optionally filtered by ``text``.

    With ``text``, keep only symbols whose meaning shares a content word with
    it ("entropy" keeps S, its molar/specific forms, and S_gen).  The match is
    lexical on purpose: it is cheap, deterministic, and easy to audit.
    """
    scope = set(chapters_in_scope(chapters))
    _, symbols = load()
    selected = [s for s in symbols if s.chapter in scope]
    if text is not None:
        words = _content_words(text)
        selected = [s for s in selected if words & _content_words(s.meaning)]
    if limit is not None:
        selected = selected[:limit]
    return selected


def grounding_block(chapters=(), text: str | None = None, *,
                    symbol_limit: int | None = 40) -> str:
    """Notation instructions for a system prompt.

    Args:
        chapters: Chapters the prompt is about; the core chapters are always
            added.  Pass nothing for core-only grounding.
        text: If given, restrict the symbol table to symbols relevant to it.
        symbol_limit: Cap on symbol rows (None for no cap).
    """
    scope = chapters_in_scope(chapters)
    lines = [
        f"NOTATION AND TERMINOLOGY — follow the textbook, {BOOK}.",
        "Write every symbol, equation, and technical term the way this book "
        "does, not the way other textbooks or your training data do. Write "
        "symbols in LaTeX between $...$, exactly as in the symbol list. Use the "
        "book's names for quantities (for example \"Gibbs energy\", not "
        "\"Gibbs free energy\"). If a source or the speaker uses different "
        "notation, translate it into the book's notation.",
        "",
        f"Conventions (chapters {', '.join(map(str, scope))}):",
    ]
    lines += [f"- {edition_neutral(c.text)}" for c in conventions_for(chapters)]
    symbols = symbols_for(chapters, text, symbol_limit)
    if symbols:
        lines += ["", "Symbols (LaTeX — meaning, units):"]
        lines += [f"- ${s.latex}$ — {edition_neutral(s.meaning)}"
                  + (f", {s.units}" if s.units not in ("", "-") else "")
                  for s in symbols]
    return "\n".join(lines)


def primer_version(grounding_dir: Path = GROUNDING_DIR) -> str:
    """Short content hash of the primer set, recorded with each output."""
    import hashlib
    digest = hashlib.sha256()
    for path in sorted(grounding_dir.glob("ch[0-9][0-9]-notation.md")):
        digest.update(path.read_bytes())
    return digest.hexdigest()[:12]


def main():
    parser = argparse.ArgumentParser(description="Print notation grounding for a prompt")
    parser.add_argument("chapters", nargs="*", type=int, help="Chapters beyond the core 1-4")
    parser.add_argument("--text", default=None, help="Filter symbols to those relevant to this text")
    parser.add_argument("--symbol-limit", type=int, default=40)
    args = parser.parse_args()
    block = grounding_block(args.chapters, args.text, symbol_limit=args.symbol_limit)
    print(block)
    print(f"\n[{len(block.split())} words, primers {primer_version()}]")


if __name__ == "__main__":
    main()
