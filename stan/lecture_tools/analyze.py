"""
Analyze lecture transcripts using a local LLM (Ollama).

Provides instructor-facing analysis tools for lecture transcripts:
    summary   - Key topics and concepts covered per lecture
    questions - Questions asked during lectures (student + rhetorical)
    confusion - Moments of student confusion or repeated explanations
    anecdotes - Stories, analogies, jokes, and real-world connections

Each analysis runs per-lecture, producing structured JSON output.
Cross-lecture aggregation finds recurring patterns across the semester.

Usage:
    # Per-lecture analysis (dry run / execute)
    python -m stan.lecture.analyze summary batch
    python -m stan.lecture.analyze summary batch --go
    python -m stan.lecture.analyze summary one transcript.txt

    # Cross-lecture aggregation
    python -m stan.lecture.analyze aggregate questions --go

    # All per-lecture analyses at once
    python -m stan.lecture.analyze all batch --go

    # Semester overview report
    python -m stan.lecture.analyze report summary
    python -m stan.lecture.analyze report questions

Requires: Ollama running locally (ollama pull llama3.1:8B)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────

DEFAULT_TRANSCRIPT_DIR = (
    Path(__file__).resolve().parents[1] / "data" / "lectures" / "transcripts"
)
DEFAULT_ANALYSIS_DIR = (
    Path(__file__).resolve().parents[1] / "data" / "lectures" / "analysis"
)
LECTURE_ORDER_FILE = (
    Path(__file__).resolve().parents[1] / "data" / "lectures" / "lecture_order.json"
)

# ── Constants ────────────────────────────────────────────────────────

# llama3.1:8B reliably follows structured JSON schemas for transcript analysis.
# command-r7b was tested (2026-02-18) and produced hallucinated structures
# (fake Q&A datasets, fabricated exam problems), but that test ran without
# num_ctx so the model only saw ~2048 tokens of context.  The failures may
# have been caused by truncation, not model capability.  Worth re-testing
# command-r7b with full context.  Override with --model if experimenting.
MODEL = "llama3.1:8B"
OLLAMA_URL = "http://localhost:11434"
OLLAMA_TIMEOUT = 600  # seconds — complex prompts (questions, confusion) with full context need time
NUM_CTX = 16384  # context window tokens — transcripts are ~13k tokens

MIN_WORDS_THRESHOLD = 100  # skip transcripts shorter than this (exams, empty)

ANALYSIS_TYPES = ("summary", "questions", "confusion", "anecdotes")

# Map analysis type -> output subdirectory name
OUTPUT_SUBDIRS = {
    "summary": "summaries",
    "questions": "questions",
    "confusion": "confusion",
    "anecdotes": "anecdotes",
}

# ── Prompts ──────────────────────────────────────────────────────────

SUMMARY_SYSTEM = """\
You are an expert teaching assistant analyzing a university lecture transcript \
from a Chemical Engineering Thermodynamics course (CHEG 231, using Sandler's \
textbook). Your task is to produce a structured summary of the lecture content.

You MUST respond with valid JSON only, no other text. Use this exact schema:

{
  "title": "Short descriptive title for this lecture (max 10 words)",
  "topics": [
    {
      "name": "Topic name",
      "description": "1-2 sentence description of what was covered"
    }
  ],
  "key_concepts": ["concept1", "concept2"],
  "key_equations": ["equation or relationship mentioned"],
  "summary": "3-5 sentence narrative summary of the entire lecture",
  "lecture_type": "one of: new_material, review, problem_solving, demonstration, mixed"
}

Note: The input is an automated speech-to-text transcription and may contain \
minor errors or awkward phrasing. Interpret generously."""

QUESTIONS_EXTRACT_SYSTEM = """\
You are an expert teaching assistant analyzing a university lecture transcript \
from a Chemical Engineering Thermodynamics course. Your ONLY task is to find \
every genuine question asked during the lecture and extract it VERBATIM.

Transcript segments are provided with timestamps in [H:MM:SS] format. \
For each question, copy the timestamp from the nearest [H:MM:SS] marker.

EXCLUDE these — they are NOT real questions:
- Filler: "right?", "okay?", "yes?", "no?", "you know?"
- Classroom management: "Any questions?", "Can you see that?", "Any volunteers?"
- Rhetorical filler: "Does that make sense?", "Are you with me?"
- Statements that are not actually questions

For each real question, extract:
- The EXACT words from the transcript (verbatim quote — do NOT paraphrase)
- The timestamp from the nearest [H:MM:SS] segment
- Whether it was a student or the instructor speaking

You MUST respond with valid JSON only, no other text. Use this exact schema:

{
  "candidates": [
    {
      "timestamp": "0:12:34",
      "speaker": "student" or "instructor",
      "verbatim": "the exact words from the transcript"
    }
  ],
  "candidate_count": 12
}

Guidelines:
- Copy the question EXACTLY as it appears — broken grammar, "um"s, and all.
- Student questions are often preceded by silence or "um" and may be incomplete.
- Include instructor Socratic questions ("What would happen if...?") but NOT filler.
- Questions may lack "?" in spoken language — look for interrogative phrasing.
- It is better to include a borderline question than to miss a real one.
- Note: input is automated transcription and may contain errors."""

QUESTIONS_FILTER_SYSTEM = """\
You are an expert teaching assistant reviewing questions extracted from a \
Chemical Engineering Thermodynamics lecture. You will receive a list of \
candidate questions with verbatim quotes and timestamps.

Your task is to CLASSIFY and RANK these questions by pedagogical significance. \
Keep 5–15 of the most important questions. Drop duplicates and low-value items.

Significance levels:
- HIGH: Student conceptual questions revealing misunderstanding or deep thinking; \
  Instructor Socratic questions that guide discovery ("What would happen if...?")
- MEDIUM: Student clarification questions about specific steps or definitions; \
  Instructor questions that test understanding ("How would you calculate...?")
- LOW: Procedural questions about homework, exams, logistics (keep only if notable)

You MUST respond with valid JSON only. Use this exact schema:

{
  "questions": [
    {
      "timestamp": "0:12:34",
      "speaker": "student" or "instructor",
      "type": "conceptual" or "clarification" or "procedural" or "socratic",
      "question_text": "the verbatim question text",
      "context": "Brief description of what was being discussed",
      "topic": "The thermodynamic topic",
      "significance": "high" or "medium" or "low"
    }
  ],
  "question_count": 8,
  "student_question_count": 5,
  "most_questioned_topics": ["topic1", "topic2"]
}

Guidelines:
- Preserve the verbatim text — do NOT rewrite the questions.
- Keep 5–15 questions total. Fewer is fine if the lecture had few real questions.
- Prefer student questions over instructor questions when they overlap.
- Group related back-and-forth into a single entry if they address the same point."""

CONFUSION_SYSTEM = """\
You are an expert teaching assistant analyzing a university lecture transcript \
from a Chemical Engineering Thermodynamics course. Your task is to identify \
moments where students appear confused or where the instructor re-explains \
material.

Transcript segments are provided with timestamps in [H:MM:SS] format. \
You MUST extract the actual timestamp from the nearest segment for each \
confusion point. Do NOT use placeholder text like "H:MM:SS" — use the \
real time from the transcript (e.g., "0:15:42").

You MUST respond with valid JSON only, no other text. Use this exact schema:

{
  "confusion_points": [
    {
      "timestamp": "0:15:42",
      "topic": "The thermodynamic concept causing difficulty",
      "evidence": "What signals confusion (e.g., 're-explanation', 'student question')",
      "description": "Brief description of what happened",
      "severity": "minor" or "moderate" or "significant"
    }
  ],
  "confusion_count": 5,
  "most_confusing_topics": ["topic1", "topic2"]
}

Signals of confusion to look for:
- Instructor says "let me say that again", "in other words", "does that make sense?"
- Instructor restarts an explanation from a different angle
- Student asks a clarifying question about something just explained
- Instructor acknowledges difficulty ("this is the tricky part", "people often get confused")
- Instructor corrects themselves mid-explanation
- Multiple attempts at the same derivation or concept
- Note: input is automated transcription and may contain minor errors."""

ANECDOTES_SYSTEM = """\
You are an expert teaching assistant analyzing a university lecture transcript \
from a Chemical Engineering Thermodynamics course. Your task is to catalog \
non-technical teaching moments: stories, real-world analogies, humor, \
demonstrations, and personal anecdotes.

You MUST respond with valid JSON only, no other text. Use this exact schema:

{
  "items": [
    {
      "type": "anecdote" or "analogy" or "joke" or "real_world_example" or "historical_note",
      "description": "Brief description of the moment",
      "quote": "Key phrase or sentence from the transcript (verbatim if possible)",
      "related_topic": "The technical concept this connects to, if any",
      "pedagogical_purpose": "Why the instructor used this"
    }
  ],
  "item_count": 7
}

Look for:
- Stories about scientists (Maxwell, Carnot, van der Waals, Gibbs, etc.)
- Real-world examples (power plants, refrigerators, car engines, cooking)
- Analogies to everyday life for abstract concepts
- Humor, jokes, or lighthearted comments
- References to current events, campus life, or student experiences
- Historical context about the development of thermodynamics
- Note: input is automated transcription and may contain minor errors."""

# ── Aggregation prompts ──────────────────────────────────────────────

AGGREGATE_QUESTIONS_SYSTEM = """\
You are analyzing questions collected from an entire semester of a Chemical \
Engineering Thermodynamics course (35 lectures). Your task is to find patterns: \
recurring question topics, frequently confused areas, and how student engagement \
evolved over the semester.

You MUST respond with valid JSON only. Use this schema:

{
  "recurring_topics": [
    {
      "topic": "Topic name",
      "frequency": 12,
      "lectures_appeared_in": [3, 5, 7, 12],
      "representative_questions": ["example question 1", "example question 2"],
      "trend": "increasing" or "decreasing" or "stable" or "concentrated"
    }
  ],
  "engagement_trend": "Description of how student questioning changed over the semester",
  "total_questions": 245,
  "total_student_questions": 89
}"""

AGGREGATE_CONFUSION_SYSTEM = """\
You are analyzing confusion points collected from an entire semester of a \
Chemical Engineering Thermodynamics course. Your task is to identify \
persistently difficult topics and how confusion patterns evolved.

You MUST respond with valid JSON only. Use this schema:

{
  "persistent_difficulties": [
    {
      "topic": "Topic name",
      "total_occurrences": 8,
      "lectures": [5, 7, 12, 15],
      "avg_severity": "moderate",
      "description": "Summary of what students struggle with",
      "suggestion": "Potential pedagogical intervention"
    }
  ],
  "semester_trend": "Description of how confusion evolved",
  "most_confusing_topics": ["topic1", "topic2", "topic3"]
}"""

AGGREGATE_ANECDOTES_SYSTEM = """\
You are analyzing anecdotes, analogies, jokes, and teaching moments collected \
from an entire semester of a Chemical Engineering Thermodynamics course. \
Organize them into a catalog grouped by type and topic.

You MUST respond with valid JSON only. Use this schema:

{
  "by_type": {
    "anecdote": 12,
    "analogy": 23,
    "joke": 8,
    "real_world_example": 15,
    "historical_note": 7
  },
  "highlights": [
    {
      "type": "analogy",
      "description": "...",
      "related_topic": "entropy",
      "lecture": 5
    }
  ],
  "total_items": 68
}"""

# Prompt config: maps analysis type -> (system_prompt, input_format, temperature)
# Note: "questions" is handled separately via two-pass (extract → filter).
ANALYSIS_CONFIG = {
    "summary": (SUMMARY_SYSTEM, "text", 0.2),
    "confusion": (CONFUSION_SYSTEM, "segments", 0.2),
    "anecdotes": (ANECDOTES_SYSTEM, "text", 0.4),
}

AGGREGATE_PROMPTS = {
    "questions": AGGREGATE_QUESTIONS_SYSTEM,
    "confusion": AGGREGATE_CONFUSION_SYSTEM,
    "anecdotes": AGGREGATE_ANECDOTES_SYSTEM,
}


# ── OllamaClient ────────────────────────────────────────────────────

class OllamaClient:
    """HTTP client for Ollama API using stdlib urllib.

    Adapted from the pattern in stan/run/query_rewrite_llm.py.
    Uses longer timeouts for large transcript analysis.
    """

    def __init__(self, base_url: str = OLLAMA_URL, model: str = MODEL):
        self.base_url = base_url
        self.model = model

    def generate(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.2,
        timeout: int = OLLAMA_TIMEOUT,
        json_mode: bool = False,
    ) -> str:
        """Generate text completion. Returns raw response string."""
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature, "num_ctx": NUM_CTX},
        }
        if system:
            payload["system"] = system
        if json_mode:
            payload["format"] = "json"

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url, data=data, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                return result["response"]
        except urllib.error.URLError as e:
            raise RuntimeError(f"Ollama API error: {e}")

    def generate_json(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.2,
        timeout: int = OLLAMA_TIMEOUT,
        retries: int = 1,
    ) -> dict:
        """Generate a response and parse it as JSON.

        Strips markdown code fences and retries on parse failure.
        """
        for attempt in range(1 + retries):
            raw = self.generate(prompt, system, temperature, timeout, json_mode=True)
            cleaned = raw.strip()
            # Strip markdown code fences (```json ... ```)
            if cleaned.startswith("```"):
                lines = cleaned.split("\n")
                lines = [l for l in lines if not l.strip().startswith("```")]
                cleaned = "\n".join(lines)
            try:
                return json.loads(cleaned)
            except json.JSONDecodeError:
                if attempt < retries:
                    print("    [RETRY]  JSON parse failed, retrying...")
                    continue
                # Last resort: extract first JSON object from response
                match = re.search(r"\{.*\}", cleaned, re.DOTALL)
                if match:
                    try:
                        return json.loads(match.group())
                    except json.JSONDecodeError:
                        pass
                raise ValueError(
                    f"Could not parse JSON from LLM response:\n{raw[:500]}"
                )


# ── Helpers ──────────────────────────────────────────────────────────

def extract_lecture_number(filename: str) -> int | None:
    """Extract the lecture number (e.g. 5) from a transcript filename.

    Pattern: CHEG231-010_Chemical_Engineering_Thermodynamics_NNN_1_HASH.{txt,json}
    """
    match = re.search(r"_(\d{3})_1_", filename)
    if match:
        return int(match.group(1))
    return None


def find_transcripts(
    transcript_dir: Path,
    filter_str: str | None = None,
) -> list[tuple[int, Path, Path]]:
    """Find transcript file pairs and return sorted (lecture_num, txt, json).

    Only returns pairs where both .txt and .json exist.
    """
    txt_files = sorted(transcript_dir.glob("*.txt"))
    results = []
    for txt_path in txt_files:
        json_path = txt_path.with_suffix(".json")
        if not json_path.exists():
            continue
        if filter_str and filter_str not in txt_path.name:
            continue
        num = extract_lecture_number(txt_path.name)
        if num is not None:
            results.append((num, txt_path, json_path))
    results.sort(key=lambda x: x[0])
    return results


def load_transcript_text(txt_path: Path) -> str:
    """Load a plain text transcript."""
    return txt_path.read_text(encoding="utf-8")


def load_transcript_segments(json_path: Path) -> list[dict]:
    """Load segments from a JSON transcript file."""
    data = json.loads(json_path.read_text(encoding="utf-8"))
    return data.get("segments", [])


def format_segments_with_timestamps(segments: list[dict]) -> str:
    """Format segments as timestamped lines for LLM consumption.

    Produces compact lines like:
        [0:00:32] So if we just wait, then heat is going to flow out...
    """
    lines = []
    for seg in segments:
        t = seg["start"]
        h = int(t // 3600)
        m = int((t % 3600) // 60)
        s = int(t % 60)
        lines.append(f"[{h}:{m:02d}:{s:02d}] {seg['text']}")
    return "\n".join(lines)


def is_valid_lecture(txt_path: Path) -> tuple[bool, int]:
    """Check if a transcript is a real lecture (not an exam or empty).

    Returns (is_valid, word_count).
    """
    text = txt_path.read_text(encoding="utf-8")
    word_count = len(text.split())
    return word_count >= MIN_WORDS_THRESHOLD, word_count


def output_path(
    analysis_dir: Path, analysis_type: str, lecture_num: int
) -> Path:
    """Compute output path for a per-lecture analysis result."""
    subdir = OUTPUT_SUBDIRS[analysis_type]
    return analysis_dir / subdir / f"lecture_{lecture_num:03d}_{analysis_type}.json"


def make_envelope(
    lecture_num: int,
    source_file: str,
    analysis_type: str,
    model: str,
    analysis_seconds: float,
    result: dict,
) -> dict:
    """Wrap an LLM result in a consistent metadata envelope."""
    return {
        "lecture_number": lecture_num,
        "source_file": source_file,
        "analysis_type": analysis_type,
        "model": model,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "analysis_seconds": round(analysis_seconds, 2),
        "result": result,
    }


# ── Core analysis ────────────────────────────────────────────────────

def analyze_one(
    analysis_type: str,
    lecture_num: int,
    txt_path: Path,
    json_path: Path,
    client: OllamaClient,
    analysis_dir: Path,
) -> Path:
    """Run a single analysis type on one lecture. Returns the output path."""
    # Questions uses a two-pass pipeline (extract → filter)
    if analysis_type == "questions":
        return _analyze_questions_two_pass(
            lecture_num, txt_path, json_path, client, analysis_dir
        )

    system_prompt, input_format, temperature = ANALYSIS_CONFIG[analysis_type]

    # Build the user prompt with the appropriate input
    if input_format == "text":
        content = load_transcript_text(txt_path)
    else:
        segments = load_transcript_segments(json_path)
        content = format_segments_with_timestamps(segments)

    user_prompt = (
        f"Analyze this lecture transcript (Lecture {lecture_num} of 39) "
        f"and produce a structured JSON response.\n\n"
        f"TRANSCRIPT:\n{content}"
    )

    print(f"  [{analysis_type.upper():10s}] Lecture {lecture_num:03d}...", end="", flush=True)
    t0 = time.time()
    result = client.generate_json(
        user_prompt, system=system_prompt, temperature=temperature
    )
    elapsed = time.time() - t0
    print(f" {elapsed:.1f}s")

    # Wrap in envelope and write
    out = output_path(analysis_dir, analysis_type, lecture_num)
    out.parent.mkdir(parents=True, exist_ok=True)
    envelope = make_envelope(
        lecture_num, txt_path.name, analysis_type, client.model, elapsed, result
    )
    out.write_text(json.dumps(envelope, indent=2, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    return out


def _analyze_questions_two_pass(
    lecture_num: int,
    txt_path: Path,
    json_path: Path,
    client: OllamaClient,
    analysis_dir: Path,
) -> Path:
    """Two-pass question analysis: extract verbatim candidates, then filter & rank.

    Pass 1 (extract): Full transcript → list of verbatim question candidates.
        Simple task: find questions, copy exact words. High recall.
    Pass 2 (filter):  Candidate list (~1-2K tokens) → classified, ranked output.
        Judgment task on small input. Produces final 5-15 questions.

    This separation addresses bimodal output from one-shot prompts, where the
    model either fabricates exactly 8 study-guide questions or dumps 100+ items.
    """
    segments = load_transcript_segments(json_path)
    content = format_segments_with_timestamps(segments)

    # ── Pass 1: Extract candidates ──
    extract_prompt = (
        f"Extract all genuine questions from this lecture transcript "
        f"(Lecture {lecture_num} of 39). Copy each question VERBATIM.\n\n"
        f"TRANSCRIPT:\n{content}"
    )

    print(f"  [QUESTIONS ] Lecture {lecture_num:03d} pass 1 (extract)...",
          end="", flush=True)
    t0 = time.time()
    candidates = client.generate_json(
        extract_prompt, system=QUESTIONS_EXTRACT_SYSTEM, temperature=0.2
    )
    t1 = time.time()
    n_candidates = len(candidates.get("candidates", []))
    print(f" {t1 - t0:.1f}s ({n_candidates} candidates)")

    # ── Pass 2: Filter & classify ──
    candidate_text = json.dumps(candidates.get("candidates", []), indent=1)
    filter_prompt = (
        f"Here are {n_candidates} candidate questions extracted from Lecture "
        f"{lecture_num}. Classify, rank by significance, and keep the 5–15 "
        f"most important.\n\nCANDIDATES:\n{candidate_text}"
    )

    print(f"  [QUESTIONS ] Lecture {lecture_num:03d} pass 2 (filter)...",
          end="", flush=True)
    result = client.generate_json(
        filter_prompt, system=QUESTIONS_FILTER_SYSTEM, temperature=0.2
    )
    t2 = time.time()
    n_final = len(result.get("questions", []))
    elapsed = t2 - t0
    print(f" {t2 - t1:.1f}s ({n_final} kept)")

    # Store pass-1 candidate count in result for diagnostics
    result["candidates_extracted"] = n_candidates

    # Wrap in envelope and write
    out = output_path(analysis_dir, "questions", lecture_num)
    out.parent.mkdir(parents=True, exist_ok=True)
    envelope = make_envelope(
        lecture_num, txt_path.name, "questions", client.model, elapsed, result
    )
    out.write_text(json.dumps(envelope, indent=2, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    return out


def analyze_batch(
    analysis_type: str,
    transcript_dir: Path,
    analysis_dir: Path,
    client: OllamaClient,
    go: bool = False,
    filter_str: str | None = None,
    overwrite: bool = False,
) -> None:
    """Batch-analyze transcripts for a single analysis type."""
    transcripts = find_transcripts(transcript_dir, filter_str)
    if not transcripts:
        print("No transcript files found.", file=sys.stderr)
        sys.exit(1)

    # Classify lectures
    valid = []
    skipped_short = []
    for num, txt, jsn in transcripts:
        ok, wc = is_valid_lecture(txt)
        if ok:
            valid.append((num, txt, jsn, wc))
        else:
            skipped_short.append((num, wc))

    print(f"Found {len(transcripts)} transcript(s) "
          f"({len(valid)} valid, {len(skipped_short)} skipped)")
    print(f"Model: {client.model}")
    print(f"Analysis: {analysis_type}")
    print()

    if not go:
        print("Dry run (pass --go to analyze):\n")
        for num, wc in skipped_short:
            print(f"  [SKIP]    Lecture {num:03d} ({wc} words)")
        for num, txt, jsn, wc in valid:
            out = output_path(analysis_dir, analysis_type, num)
            if out.exists() and not overwrite:
                print(f"  [EXISTS]  Lecture {num:03d} (already analyzed)")
            else:
                print(f"  [PENDING] Lecture {num:03d} ({wc} words)")
        print(f"\nRerun with --go to analyze to "
              f"{analysis_dir / OUTPUT_SUBDIRS[analysis_type]}")
        return

    success = 0
    skipped = 0
    failed = 0
    t0_total = time.time()

    for num, txt, jsn, wc in valid:
        out = output_path(analysis_dir, analysis_type, num)
        if out.exists() and not overwrite:
            print(f"  [SKIP]       Lecture {num:03d} (already analyzed)")
            skipped += 1
            continue
        try:
            analyze_one(analysis_type, num, txt, jsn, client, analysis_dir)
            success += 1
        except Exception as e:
            print(f"  [FAIL]       Lecture {num:03d}: {e}")
            failed += 1

    elapsed_total = time.time() - t0_total
    print(f"\n{success} analyzed, {skipped} skipped, {failed} failed "
          f"({elapsed_total:.0f}s total)")


# ── Aggregation ──────────────────────────────────────────────────────

def aggregate(
    analysis_type: str,
    analysis_dir: Path,
    client: OllamaClient,
    go: bool = False,
) -> None:
    """Load all per-lecture results for a type and find cross-lecture patterns."""
    if analysis_type not in AGGREGATE_PROMPTS:
        print(f"No aggregation defined for '{analysis_type}'.", file=sys.stderr)
        sys.exit(1)

    subdir = analysis_dir / OUTPUT_SUBDIRS[analysis_type]
    if not subdir.exists():
        print(f"No {analysis_type} results found at {subdir}.", file=sys.stderr)
        print(f"Run per-lecture analysis first: "
              f"python -m stan.lecture.analyze {analysis_type} batch --go")
        sys.exit(1)

    # Load all per-lecture results
    result_files = sorted(subdir.glob("lecture_*_*.json"))
    if not result_files:
        print(f"No result files in {subdir}.", file=sys.stderr)
        sys.exit(1)

    all_results = []
    for f in result_files:
        data = json.loads(f.read_text(encoding="utf-8"))
        all_results.append({
            "lecture_number": data["lecture_number"],
            "result": data["result"],
        })

    print(f"Aggregating {len(all_results)} {analysis_type} results")
    out_path = subdir / f"semester_{analysis_type}.json"

    if not go:
        print(f"\nDry run. Would aggregate to: {out_path}")
        print(f"Rerun with --go to execute.")
        return

    system_prompt = AGGREGATE_PROMPTS[analysis_type]
    user_prompt = (
        f"Here are the {analysis_type} results from each lecture in the semester. "
        f"Find patterns and produce an aggregated analysis.\n\n"
        + json.dumps(all_results, indent=1, ensure_ascii=False)
    )

    print(f"  [AGGREGATE]  {analysis_type}...", end="", flush=True)
    t0 = time.time()
    result = client.generate_json(
        user_prompt, system=system_prompt, temperature=0.3
    )
    elapsed = time.time() - t0
    print(f" {elapsed:.1f}s")

    envelope = make_envelope(
        lecture_num=0,
        source_file=f"{len(all_results)} lecture results",
        analysis_type=f"aggregate_{analysis_type}",
        model=client.model,
        analysis_seconds=elapsed,
        result=result,
    )
    out_path.write_text(
        json.dumps(envelope, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"  Written to {out_path}")


# ── Report ────────────────────────────────────────────────────────────

def _format_summary_report(data: dict, lecture_num: int, class_num: int | None = None) -> str:
    """Format a single summary result as readable text."""
    r = data["result"]
    if class_num is not None:
        header = f"Class {class_num:02d} (capture {lecture_num:03d}): {r.get('title', '(no title)')}"
    else:
        header = f"Lecture {lecture_num:03d}: {r.get('title', '(no title)')}"
    lines = [header]

    # Topics — handle both schema variants (name/description or topic/notes)
    topics = r.get("topics", [])
    for t in topics:
        name = t.get("name") or t.get("topic", "")
        desc = t.get("description") or t.get("notes", "")
        if name:
            lines.append(f"  - {name}")
            if desc:
                lines.append(f"    {desc}")

    # Key concepts
    concepts = r.get("key_concepts", [])
    if concepts:
        # Handle both flat strings and dicts
        names = []
        for c in concepts:
            if isinstance(c, str):
                names.append(c)
            elif isinstance(c, dict):
                names.append(c.get("concept") or c.get("name", ""))
        if names:
            lines.append(f"  Concepts: {', '.join(names)}")

    # Lecture type
    ltype = r.get("lecture_type", "")
    if ltype:
        lines.append(f"  Type: {ltype}")

    # Summary narrative
    summary = r.get("summary", "")
    if summary:
        lines.append(f"  {summary}")

    return "\n".join(lines)


def _format_questions_report(data: dict, lecture_num: int, class_num: int | None = None) -> str:
    """Format a single questions result as readable text."""
    r = data["result"]
    questions = [q for q in r.get("questions", []) if isinstance(q, dict)]
    student_q = sum(1 for q in questions if q.get("speaker") == "student")
    label = f"Class {class_num:02d}" if class_num is not None else f"Lecture {lecture_num:03d}"
    lines = [f"{label}: {len(questions)} questions "
             f"({student_q} student, {len(questions) - student_q} instructor)"]
    for q in questions:
        ts = q.get("timestamp", "")
        speaker = q.get("speaker", "?")[0].upper()  # S or I
        sig = q.get("significance", "")
        sig_tag = f" [{sig}]" if sig else ""
        qtext = q.get("question_text", q.get("text", ""))
        lines.append(f"  [{ts}] ({speaker}){sig_tag} {qtext}")
    return "\n".join(lines)


def _format_confusion_report(data: dict, lecture_num: int, class_num: int | None = None) -> str:
    """Format a single confusion result as readable text."""
    r = data["result"]
    points = r.get("confusion_points", [])
    label = f"Class {class_num:02d}" if class_num is not None else f"Lecture {lecture_num:03d}"
    lines = [f"{label}: {len(points)} confusion points"]
    for p in points:
        ts = p.get("timestamp", "")
        sev = p.get("severity", "?")
        topic = p.get("topic", "")
        desc = p.get("description", "")
        lines.append(f"  [{ts}] [{sev}] {topic}: {desc}")
    return "\n".join(lines)


def _format_anecdotes_report(data: dict, lecture_num: int, class_num: int | None = None) -> str:
    """Format a single anecdotes result as readable text."""
    r = data["result"]
    items = r.get("items", [])
    label = f"Class {class_num:02d}" if class_num is not None else f"Lecture {lecture_num:03d}"
    lines = [f"{label}: {len(items)} items"]
    for item in items:
        itype = item.get("type", "?")
        desc = item.get("description", "")
        quote = item.get("quote", "")
        lines.append(f"  [{itype}] {desc}")
        if quote:
            lines.append(f"    \"{quote}\"")
    return "\n".join(lines)


REPORT_FORMATTERS = {
    "summary": _format_summary_report,
    "questions": _format_questions_report,
    "confusion": _format_confusion_report,
    "anecdotes": _format_anecdotes_report,
}


def _load_teaching_order() -> dict[str, int]:
    """Load lecture_order.json mapping capture number → teaching order."""
    if not LECTURE_ORDER_FILE.exists():
        return {}
    data = json.loads(LECTURE_ORDER_FILE.read_text(encoding="utf-8"))
    return {str(k): int(v) for k, v in data.get("order", {}).items()}


def report(analysis_type: str, analysis_dir: Path) -> None:
    """Print a formatted semester report for an analysis type."""
    subdir = analysis_dir / OUTPUT_SUBDIRS[analysis_type]
    if not subdir.exists():
        print(f"No {analysis_type} results found at {subdir}.", file=sys.stderr)
        print(f"Run analysis first: "
              f"python -m stan.lecture.analyze {analysis_type} batch --go")
        sys.exit(1)

    result_files = sorted(subdir.glob("lecture_*_*.json"))
    if not result_files:
        print(f"No result files in {subdir}.", file=sys.stderr)
        sys.exit(1)

    # Load teaching order and sort by it (fall back to capture number)
    teaching_order = _load_teaching_order()
    entries = []
    for f in result_files:
        data = json.loads(f.read_text(encoding="utf-8"))
        capture_num = data["lecture_number"]
        class_num = teaching_order.get(f"{capture_num:03d}", capture_num)
        entries.append((class_num, capture_num, data))
    entries.sort(key=lambda e: e[0])

    formatter = REPORT_FORMATTERS[analysis_type]

    has_order = bool(teaching_order)
    print(f"\n{'=' * 65}")
    print(f"  CHEG 231 — Semester {analysis_type.title()} Report "
          f"({len(entries)} lectures)")
    if has_order:
        print(f"  (sorted by teaching order)")
    print(f"{'=' * 65}\n")

    for class_num, capture_num, data in entries:
        if has_order:
            print(formatter(data, capture_num, class_num=class_num))
        else:
            print(formatter(data, capture_num))
        print()


# ── CLI ──────────────────────────────────────────────────────────────

def add_common_args(parser):
    """Add arguments shared across analysis subcommands."""
    parser.add_argument(
        "--transcript-dir", type=str, default=str(DEFAULT_TRANSCRIPT_DIR),
        help=f"Transcript directory (default: {DEFAULT_TRANSCRIPT_DIR})",
    )
    parser.add_argument(
        "--output-dir", type=str, default=str(DEFAULT_ANALYSIS_DIR),
        help=f"Analysis output directory (default: {DEFAULT_ANALYSIS_DIR})",
    )
    parser.add_argument("--model", default=MODEL, help=f"Ollama model (default: {MODEL})")
    parser.add_argument("--go", action="store_true", help="Actually run (default is dry run)")
    parser.add_argument("--overwrite", action="store_true", help="Re-analyze existing results")
    parser.add_argument("--filter", type=str, default=None, help="Filter transcripts by string")


def cmd_analysis(args, analysis_type: str):
    """Run a per-lecture analysis type."""
    client = OllamaClient(model=args.model)

    if args.mode == "one":
        txt_path = Path(args.file)
        if not txt_path.exists():
            print(f"Error: {txt_path} not found", file=sys.stderr)
            sys.exit(1)
        json_path = txt_path.with_suffix(".json")
        if not json_path.exists():
            print(f"Error: companion JSON not found: {json_path}", file=sys.stderr)
            sys.exit(1)
        num = extract_lecture_number(txt_path.name)
        if num is None:
            print("Warning: could not extract lecture number, using 0")
            num = 0
        analysis_dir = Path(args.output_dir)
        analyze_one(analysis_type, num, txt_path, json_path, client, analysis_dir)

    elif args.mode == "batch":
        analyze_batch(
            analysis_type,
            transcript_dir=Path(args.transcript_dir),
            analysis_dir=Path(args.output_dir),
            client=client,
            go=args.go,
            filter_str=args.filter,
            overwrite=args.overwrite,
        )


def cmd_aggregate(args):
    """Run cross-lecture aggregation."""
    client = OllamaClient(model=args.model)
    aggregate(
        args.type,
        analysis_dir=Path(args.output_dir),
        client=client,
        go=args.go,
    )


def cmd_report(args):
    """Print a formatted semester report."""
    report(args.type, analysis_dir=Path(args.output_dir))


def cmd_all(args):
    """Run all four analysis types in batch."""
    client = OllamaClient(model=args.model)
    for atype in ANALYSIS_TYPES:
        print(f"\n{'=' * 60}")
        print(f"  {atype.upper()}")
        print(f"{'=' * 60}\n")
        analyze_batch(
            atype,
            transcript_dir=Path(args.transcript_dir),
            analysis_dir=Path(args.output_dir),
            client=client,
            go=args.go,
            filter_str=args.filter,
            overwrite=args.overwrite,
        )


def main():
    parser = argparse.ArgumentParser(
        description="Analyze lecture transcripts with a local LLM (Ollama)"
    )
    sub = parser.add_subparsers(dest="command")

    # Per-lecture analysis subcommands (summary, questions, confusion, anecdotes)
    for atype in ANALYSIS_TYPES:
        p = sub.add_parser(atype, help=f"Run {atype} analysis")
        mode_sub = p.add_subparsers(dest="mode")

        p_one = mode_sub.add_parser("one", help=f"Analyze a single transcript")
        p_one.add_argument("file", help="Path to transcript .txt file")
        p_one.add_argument("--output-dir", type=str, default=str(DEFAULT_ANALYSIS_DIR))
        p_one.add_argument("--model", default=MODEL)

        p_batch = mode_sub.add_parser("batch", help=f"Batch analyze all transcripts")
        add_common_args(p_batch)

        p.set_defaults(func=lambda args, at=atype: cmd_analysis(args, at))

    # report
    p_rep = sub.add_parser("report", help="Print formatted semester report")
    p_rep.add_argument("type", choices=list(ANALYSIS_TYPES),
                       help="Analysis type to report on")
    p_rep.add_argument("--output-dir", type=str, default=str(DEFAULT_ANALYSIS_DIR))
    p_rep.set_defaults(func=cmd_report)

    # aggregate
    p_agg = sub.add_parser("aggregate", help="Cross-lecture pattern analysis")
    p_agg.add_argument("type", choices=list(AGGREGATE_PROMPTS.keys()),
                       help="Analysis type to aggregate")
    p_agg.add_argument("--output-dir", type=str, default=str(DEFAULT_ANALYSIS_DIR))
    p_agg.add_argument("--model", default=MODEL)
    p_agg.add_argument("--go", action="store_true")
    p_agg.set_defaults(func=cmd_aggregate)

    # all
    p_all = sub.add_parser("all", help="Run all analysis types")
    mode_sub_all = p_all.add_subparsers(dest="mode")
    p_all_batch = mode_sub_all.add_parser("batch", help="Batch all analyses")
    add_common_args(p_all_batch)
    p_all.set_defaults(func=cmd_all)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
