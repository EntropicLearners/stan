"""
Turn lecture summaries into a student-facing recap in Markdown for Canvas.

The per-lecture summaries (analyze.py) are written for the instructor and
exist once per section.  A recap merges the sections of each lecture and
addresses the class.  The work is split so the model never touches anything
that must be exact:

  code   lecture number and date (course_schedule.json), layout, and the key
         equations -- copied verbatim from the summaries, de-duplicated across
         sections
  model  a title and a 3-5 sentence recap per lecture, prompted with the
         textbook notation primer and told to keep the summaries' notation
         and introduce none of its own; it also picks which of the numbered
         candidate equations to show, by number, so it never retypes one
  check  every recap must pass stan.notation.check, and every math span in it
         must already appear in that lecture's summaries ("no new math");
         otherwise it is retried, and if it still fails the lecture falls back
         to its summary topics, with a warning

The Markdown (with $...$ LaTeX) is meant for md_canvas.py, which turns it into
a Canvas HTML fragment.

Usage:
    python -m stan.lecture_tools.recap 13 14 15            # lectures 13-15, both sections
    python -m stan.lecture_tools.recap 13-15 -o recap.md
    python -m stan.lecture_tools.recap 13-15 --section 010
    python -m stan.lecture_tools.recap 13-15 --no-llm      # code only, no model
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

from stan.notation import primer
from stan.notation.check import check_text, _math_spans
from .analyze import (
    DEFAULT_ANALYSIS_DIR, MODEL, OllamaClient, LectureID, lecture_chapters,
)
from .recordings import Schedule

RECAP_TEMPERATURE = 0.3
RECAP_RETRIES = 2
RECAP_TEMPERATURE_STEP = 0.2
# Thinking (~1-1.5k tokens) plus a title and one paragraph.
RECAP_NUM_PREDICT = 4096

RECAP_SYSTEM = """\
You write short lecture recaps for the students of CHEG 231 (Chemical \
Engineering Thermodynamics), to be posted on the course's Canvas site. You are \
given the instructor-facing summaries of ONE lecture; the course has two \
sections, so there may be two summaries of the same lecture, which can differ \
in emphasis.

Write:
- First line: "Title: " followed by a short title for the lecture (at most 8 words).
- Then a blank line and ONE paragraph of 3-5 sentences recapping what we \
covered, in the past tense, addressing the class as "we" (e.g. "We derived..."). \
Merge the sections into one account; do not mention sections or the instructor.

Rules:
- Use only content from the summaries. Do not add topics, examples, or results.
- NOTATION: the summaries already use the textbook's notation. Keep every \
symbol exactly as the summaries write it, in $...$. Do NOT introduce any symbol, \
equation, or notation that is not in the summaries, and do not rewrite one in \
another form. If you are unsure, describe the idea in words instead.
- Do not write out the key equations in the paragraph; they are added separately.
- Last line: "Equations: " followed by the numbers of the CANDIDATE KEY EQUATIONS \
worth showing students, most important first, at most 4, comma-separated. Skip \
any that says the same thing as one already chosen (e.g. the same balance with \
the terms rearranged).
- Use the textbook's names for quantities (see below).
- Output only the title line, the paragraph, and the Equations line: no \
headings, lists, or notes."""

MAX_EQUATIONS = 4


@dataclass
class Lecture:
    number: int
    date: date | None
    summaries: list[dict]          # analysis envelopes, one per section


@dataclass
class Recap:
    title: str
    paragraph: str | None          # None when the model output failed the checks
    equations: list[str]
    warnings: list[str]


# ── Loading ──────────────────────────────────────────────────────────

def parse_lectures(specs: list[str]) -> list[int]:
    out = []
    for spec in specs:
        for part in spec.split(","):
            if "-" in part:
                a, b = part.split("-")
                out.extend(range(int(a), int(b) + 1))
            elif part:
                out.append(int(part))
    return sorted(set(out))


def load_lectures(numbers: list[int], analysis_dir: Path, sections: list[str] | None,
                  schedule: Schedule | None) -> list[Lecture]:
    summaries_dir = analysis_dir / "summaries"
    dates = schedule.lecture_dates(datetime.now(schedule.tz).date()) if schedule else []
    lectures = []
    for n in numbers:
        found = []
        for path in sorted(summaries_dir.glob(f"*-lecture-{n:02d}_summary.json")):
            section = path.name[:3]
            if sections and section not in sections:
                continue
            found.append(json.loads(path.read_text(encoding="utf-8")))
        if not found:
            print(f"Warning: no summaries for lecture {n}", file=sys.stderr)
            continue
        lectures.append(Lecture(n, dates[n - 1] if n <= len(dates) else None, found))
    return lectures


# ── Deterministic parts ──────────────────────────────────────────────

def _norm(math: str) -> str:
    return re.sub(r"\s+", "", math.strip("$"))


def _eq_key(math: str) -> str:
    """Normalized equation, with the sides of a single '=' in a fixed order,
    so "0 = X" and "X = 0" count as the same equation."""
    n = _norm(math)
    sides = n.split("=")
    return "=".join(sorted(sides)) if len(sides) == 2 else n


def key_equations(lecture: Lecture) -> list[str]:
    """Key equations from all sections, verbatim, without repeats."""
    seen, out = set(), []
    for env in lecture.summaries:
        for eq in env["result"].get("key_equations", []):
            if _eq_key(eq) not in seen:
                seen.add(_eq_key(eq))
                out.append(eq)
    return out


def source_text(lecture: Lecture) -> str:
    """The summaries as plain text, for the prompt and for the math check."""
    blocks = []
    for i, env in enumerate(lecture.summaries, 1):
        r = env["result"]
        lines = [f"SUMMARY {i}", f"Title: {r.get('title', '')}", "Topics:"]
        lines += [f"- {t.get('name', '')}: {t.get('description', '')}" for t in r.get("topics", [])]
        lines += [f"Key concepts: {'; '.join(r.get('key_concepts', []))}",
                  "Key equations:"]
        lines += [f"- {e}" for e in r.get("key_equations", [])]
        lines += [f"Narrative: {r.get('summary', '')}"]
        blocks.append("\n".join(lines))
    candidates = ["CANDIDATE KEY EQUATIONS"]
    candidates += [f"{i}. {eq}" for i, eq in enumerate(key_equations(lecture), 1)]
    return "\n\n".join(blocks + ["\n".join(candidates)])


def new_math(text: str, source: str) -> list[str]:
    """Math spans in ``text`` that do not occur in any math span of ``source``."""
    known = [_norm(s) for s in _math_spans(source)]
    return [s for s in _math_spans(text) if not any(_norm(s) in k for k in known)]


def fallback_paragraph(lecture: Lecture) -> str:
    topics = lecture.summaries[0]["result"].get("topics", [])
    return "Topics covered: " + "; ".join(t.get("name", "") for t in topics) + "."


# ── Model ────────────────────────────────────────────────────────────

def write_recap(lecture: Lecture, client: OllamaClient | None) -> Recap:
    equations = key_equations(lecture)
    default_title = lecture.summaries[0]["result"].get("title", f"Lecture {lecture.number}")
    if client is None:
        return Recap(default_title, None, equations, [])

    source = source_text(lecture)
    chapters = sorted({c for env in lecture.summaries
                       for c in lecture_chapters(LectureID(env["section"], lecture.number))})
    system = RECAP_SYSTEM + "\n\n" + primer.grounding_block(chapters, text=source, symbol_limit=30)
    prompt = f"Summaries of lecture {lecture.number}:\n\n{source}"

    warnings = []
    for attempt in range(1 + RECAP_RETRIES):
        temp = RECAP_TEMPERATURE + RECAP_TEMPERATURE_STEP * attempt
        raw = client.generate(prompt, system, temp, num_predict=RECAP_NUM_PREDICT).strip()
        match = re.match(r"Title:\s*(.+?)\s*\n\s*\n(.+?)\n\s*Equations:\s*([\d,\s]*)$", raw, re.S)
        if not match:
            warnings = [f"attempt {attempt + 1}: output not in Title/paragraph/Equations form"]
            continue
        title, paragraph = match.group(1).strip(), " ".join(match.group(2).split())
        picks = [int(x) for x in re.findall(r"\d+", match.group(3))]
        chosen = [equations[i - 1] for i in dict.fromkeys(picks) if 1 <= i <= len(equations)]
        both = f"{title}\n{paragraph}"
        findings = [f"{f.rule}: {f.match}" for f in check_text(both)]
        novel = new_math(both, source)
        if not findings and not novel:
            return Recap(title, paragraph, chosen[:MAX_EQUATIONS] or equations[:MAX_EQUATIONS], [])
        warnings = [f"attempt {attempt + 1}: " + "; ".join(
            findings + [f"new math {s!r}" for s in novel])]
    return Recap(default_title, None, equations,
                 warnings + ["fell back to the summary topics"])


# ── Markdown ─────────────────────────────────────────────────────────

def render(lectures: list[Lecture], recaps: list[Recap]) -> str:
    nums = [l.number for l in lectures]
    span = f"Lecture {nums[0]}" if len(nums) == 1 else f"Lectures {nums[0]}–{nums[-1]}"
    dated = [l.date for l in lectures if l.date]
    when = ""
    if dated:
        when = f" ({dated[0]:%b %-d})" if len(dated) == 1 else \
               f" ({dated[0]:%b %-d} – {dated[-1]:%b %-d})"
    lines = [f"# Lecture recap: {span}{when}", ""]
    for lec, rec in zip(lectures, recaps):
        day = f" · {lec.date:%a, %b %-d}" if lec.date else ""
        taught = sorted({env["instructor"] for env in lec.summaries if env.get("instructor")})
        by = f" (taught by {', '.join(taught)})" if taught else ""
        lines += [f"## Lecture {lec.number}{day}: {rec.title}{by}", ""]
        lines += [rec.paragraph or fallback_paragraph(lec), ""]
        if rec.equations:
            lines += ["**Key equations**", ""]
            lines += [f"$${eq.strip().strip('$')}$$" + "\n" for eq in rec.equations]
    return "\n".join(lines).rstrip() + "\n"


def main():
    parser = argparse.ArgumentParser(description="Student-facing lecture recap (Markdown) for Canvas")
    parser.add_argument("lectures", nargs="+", help="Lecture numbers: 13 14 15, 13-15, or 13,15")
    parser.add_argument("--section", action="append", default=None,
                        help="Use only this section's summaries (repeatable; default: all)")
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS_DIR)
    parser.add_argument("--model", default=MODEL, help=f"Ollama model (default: {MODEL})")
    parser.add_argument("--no-llm", action="store_true",
                        help="No model: titles and topics from the summaries")
    parser.add_argument("-o", "--output", type=Path, help="Write Markdown here instead of stdout")
    args = parser.parse_args()

    try:
        schedule = Schedule.load()
    except FileNotFoundError:
        schedule = None
    lectures = load_lectures(parse_lectures(args.lectures), args.analysis_dir,
                             args.section, schedule)
    if not lectures:
        sys.exit("No summaries found.")
    client = None if args.no_llm else OllamaClient(model=args.model)

    recaps = []
    for lec in lectures:
        print(f"  lecture {lec.number} ({len(lec.summaries)} summaries)...", file=sys.stderr, flush=True)
        rec = write_recap(lec, client)
        for w in rec.warnings:
            print(f"    WARNING lecture {lec.number}: {w}", file=sys.stderr)
        recaps.append(rec)

    md = render(lectures, recaps)
    if args.output:
        args.output.write_text(md, encoding="utf-8")
        print(f"Wrote {args.output}", file=sys.stderr)
    else:
        sys.stdout.write(md)


if __name__ == "__main__":
    main()
