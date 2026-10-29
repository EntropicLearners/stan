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
    python -m stan.lecture_tools.analyze summary batch
    python -m stan.lecture_tools.analyze summary batch --go
    python -m stan.lecture_tools.analyze summary one transcript.txt

    # Cross-lecture aggregation
    python -m stan.lecture_tools.analyze aggregate questions --go

    # All per-lecture analyses at once
    python -m stan.lecture_tools.analyze all batch --go

    # Semester overview report
    python -m stan.lecture_tools.analyze report summary
    python -m stan.lecture_tools.analyze report questions

Requires: Ollama running locally (ollama pull gemma4:31b)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.request
import urllib.error
from dataclasses import dataclass
from pathlib import Path

from stan.notation import primer

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
LECTURE_CHAPTERS_FILE = (
    Path(__file__).resolve().parents[1] / "data" / "lectures" / "lecture_chapters.json"
)
COURSE_SCHEDULE_FILE = (
    Path(__file__).resolve().parents[1] / "data" / "lectures" / "course_schedule.json"
)

# ── Constants ────────────────────────────────────────────────────────

# gemma4:31b is the default as of 2026-09-08: 19 GB of weights leaves enough
# room on a 24 GB card for a 16k context, and analysis runs offline in batch
# so the slower generation costs little.  The Spring 2026 results were
# produced with llama3.1:8B, which is still available with --model when
# reproducing them.
#
# command-r7b was tested (2026-02-18) and produced hallucinated structures
# (fake Q&A datasets, fabricated exam problems), but that test ran without
# num_ctx so the model only saw ~2048 tokens of context.  The failures may
# have been caused by truncation, not model capability.
MODEL = "gemma4:31b"
OLLAMA_URL = "http://localhost:11434"
OLLAMA_TIMEOUT = 1800  # seconds — gemma4:31b partly offloads to CPU (see MODEL note)

# Context window.  This must fit the prompt AND leave room for the answer.
# A timestamped transcript ("segments" input, used by confusion and questions)
# runs ~12.5k tokens for a 55-minute lecture, so the previous 16384 left under
# 4k for output and the models quietly truncated: on 010-lecture-02, confusion
# returned 1 point at 16384 versus 5 at 32768, and question extraction
# returned an empty list at 16384 versus 12-60 candidates at 32768.  The
# failure is silent -- schema-valid JSON with empty arrays -- so it reads as
# "nothing found in this lecture" rather than as an error.
NUM_CTX = 32768

# Output cap per analysis type, in tokens.  A schema-constrained response
# should never fail to parse, but a model caught in a repetition loop
# generates until the context is full (888 s on one summary) and the
# truncated JSON then fails.  gemma4 THINKS by default, and the hidden
# thinking (~1.2-1.4k tokens on a summary) counts toward num_predict along
# with the ~800-token answer: a 2048 cap left some runs with an empty
# response.  Summaries measured 1.7-2.0k tokens in total (2026-10-03); 6144
# leaves 3x headroom and still bounds a runaway at ~4 minutes (~27 tok/s).
# Types not listed are uncapped.
NUM_PREDICT = {"summary": 6144}

# Added to the temperature on each retry of a failed generation.  At 0.2 a
# retry often reproduces the same failure token for token.
RETRY_TEMPERATURE_STEP = 0.3

# Retries per analysis type (default 1).  About 1 in 5 grounded summary
# attempts writes single-backslash LaTeX and is rejected (Fall 2026
# regeneration, 2026-10-03), so summaries get a third attempt.
RETRIES = {"summary": 2}

MIN_WORDS_THRESHOLD = 100  # skip transcripts shorter than this (exams, empty)

ANALYSIS_TYPES = ("summary", "questions", "confusion", "anecdotes")

# Analysis types whose prompts carry the textbook notation primer (see
# stan.notation).  Summaries write equations and technical terms, so they
# drift into the model's own notation without it; the other types quote or
# paraphrase the transcript and are left unchanged for now.
NOTATION_TYPES = ("summary",)

# Cap on symbol rows in the grounding block.  Rows are filtered to those whose
# meaning appears in the transcript, so this rarely binds.
NOTATION_SYMBOL_LIMIT = 60

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

# Appended to SUMMARY_SYSTEM, ahead of the notation primer, when notation
# grounding is on.  The speaker's spoken notation ("s sub k", "m dot") is
# exactly what the summary should NOT copy.
SUMMARY_NOTATION_NOTE = """\
Write each entry of "key_equations" as LaTeX between $...$, in the textbook's \
notation below (for example $\\dot{M}_k \\hat{S}_k$, not m_dot_k * s_k). In \
"title", "topics", "key_concepts", and "summary", use the textbook's names for \
quantities and put any symbol in $...$. The transcript records how the lecture \
was spoken, not how the book writes it: translate into the book's notation, \
and correct speech-to-text misspellings of names (the transcripts write \
"Joule-Thompson"; the book writes "Joule-Thomson").

JSON escaping: inside a JSON string every LaTeX backslash is written TWICE, \
for example "$\\\\hat{H} = \\\\hat{U} + P\\\\hat{V}$". A single backslash is an \
error. Write each equation once, as the book would; if you are unsure of an \
equation, leave it out rather than revising it inside the string."""

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

# ── Output schemas ───────────────────────────────────────────────────
#
# Each prompt already describes its JSON shape in prose, which llama3.1:8B
# followed reliably.  Later models do not: gemma4:e4b and gemma4:31b both
# ignore the prose schema and answer in a shape of their own choosing (an
# extract-questions call came back as a lecture summary), because Gemma's
# chat template has no distinct system role and Ollama folds the system text
# into the user turn, where it competes with 12k tokens of transcript.
#
# Passing a real JSON Schema as Ollama's ``format`` constrains decoding, so
# the shape no longer depends on the model choosing to cooperate.  The prose
# schemas stay in the prompts: they still convey intent and field semantics,
# which the grammar cannot.
#
# Schemas deliberately omit minItems.  It is expressible, but forcing a
# minimum count invites invented content when a lecture genuinely has (say)
# no jokes in it.

_TIMESTAMP = {"type": "string", "description": "Timestamp as H:MM:SS"}
_SPEAKER = {"type": "string", "enum": ["student", "instructor"]}

SUMMARY_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "topics": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "description": {"type": "string"},
                },
                "required": ["name", "description"],
            },
        },
        "key_concepts": {"type": "array", "items": {"type": "string"}},
        "key_equations": {"type": "array", "items": {"type": "string"}},
        "summary": {"type": "string"},
        "lecture_type": {
            "type": "string",
            "enum": ["new_material", "review", "problem_solving",
                     "demonstration", "mixed"],
        },
    },
    "required": ["title", "topics", "key_concepts", "key_equations",
                 "summary", "lecture_type"],
}

QUESTIONS_EXTRACT_SCHEMA = {
    "type": "object",
    "properties": {
        "candidates": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "timestamp": _TIMESTAMP,
                    "speaker": _SPEAKER,
                    "verbatim": {"type": "string"},
                },
                "required": ["timestamp", "speaker", "verbatim"],
            },
        },
        "candidate_count": {"type": "integer"},
    },
    "required": ["candidates", "candidate_count"],
}

QUESTIONS_FILTER_SCHEMA = {
    "type": "object",
    "properties": {
        "questions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "timestamp": _TIMESTAMP,
                    "speaker": _SPEAKER,
                    "type": {
                        "type": "string",
                        "enum": ["conceptual", "clarification",
                                 "procedural", "socratic"],
                    },
                    "question_text": {"type": "string"},
                    "context": {"type": "string"},
                    "topic": {"type": "string"},
                    "significance": {
                        "type": "string",
                        "enum": ["high", "medium", "low"],
                    },
                },
                "required": ["timestamp", "speaker", "type", "question_text",
                             "context", "topic", "significance"],
            },
        },
        "question_count": {"type": "integer"},
        "student_question_count": {"type": "integer"},
        "most_questioned_topics": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["questions", "question_count", "student_question_count",
                 "most_questioned_topics"],
}

CONFUSION_SCHEMA = {
    "type": "object",
    "properties": {
        "confusion_points": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "timestamp": _TIMESTAMP,
                    "topic": {"type": "string"},
                    "evidence": {"type": "string"},
                    "description": {"type": "string"},
                    "severity": {
                        "type": "string",
                        "enum": ["minor", "moderate", "significant"],
                    },
                },
                "required": ["timestamp", "topic", "evidence",
                             "description", "severity"],
            },
        },
        "confusion_count": {"type": "integer"},
        "most_confusing_topics": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["confusion_points", "confusion_count", "most_confusing_topics"],
}

ANECDOTES_SCHEMA = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {
                        "type": "string",
                        "enum": ["anecdote", "analogy", "joke",
                                 "real_world_example", "historical_note"],
                    },
                    "description": {"type": "string"},
                    "quote": {"type": "string"},
                    "related_topic": {"type": "string"},
                    "pedagogical_purpose": {"type": "string"},
                },
                "required": ["type", "description", "quote",
                             "related_topic", "pedagogical_purpose"],
            },
        },
        "item_count": {"type": "integer"},
    },
    "required": ["items", "item_count"],
}

AGGREGATE_QUESTIONS_SCHEMA = {
    "type": "object",
    "properties": {
        "recurring_topics": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string"},
                    "frequency": {"type": "integer"},
                    "lectures_appeared_in": {
                        "type": "array", "items": {"type": "integer"},
                    },
                    "representative_questions": {
                        "type": "array", "items": {"type": "string"},
                    },
                    "trend": {
                        "type": "string",
                        "enum": ["increasing", "decreasing",
                                 "stable", "concentrated"],
                    },
                },
                "required": ["topic", "frequency", "lectures_appeared_in",
                             "representative_questions", "trend"],
            },
        },
        "engagement_trend": {"type": "string"},
        "total_questions": {"type": "integer"},
        "total_student_questions": {"type": "integer"},
    },
    "required": ["recurring_topics", "engagement_trend",
                 "total_questions", "total_student_questions"],
}

AGGREGATE_CONFUSION_SCHEMA = {
    "type": "object",
    "properties": {
        "persistent_difficulties": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string"},
                    "total_occurrences": {"type": "integer"},
                    "lectures": {"type": "array", "items": {"type": "integer"}},
                    "avg_severity": {
                        "type": "string",
                        "enum": ["minor", "moderate", "significant"],
                    },
                    "description": {"type": "string"},
                    "suggestion": {"type": "string"},
                },
                "required": ["topic", "total_occurrences", "lectures",
                             "avg_severity", "description", "suggestion"],
            },
        },
        "semester_trend": {"type": "string"},
        "most_confusing_topics": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["persistent_difficulties", "semester_trend",
                 "most_confusing_topics"],
}

AGGREGATE_ANECDOTES_SCHEMA = {
    "type": "object",
    "properties": {
        "by_type": {
            "type": "object",
            "properties": {
                "anecdote": {"type": "integer"},
                "analogy": {"type": "integer"},
                "joke": {"type": "integer"},
                "real_world_example": {"type": "integer"},
                "historical_note": {"type": "integer"},
            },
            "required": ["anecdote", "analogy", "joke",
                         "real_world_example", "historical_note"],
        },
        "highlights": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {"type": "string"},
                    "description": {"type": "string"},
                    "related_topic": {"type": "string"},
                    "lecture": {"type": "integer"},
                },
                "required": ["type", "description", "related_topic", "lecture"],
            },
        },
        "total_items": {"type": "integer"},
    },
    "required": ["by_type", "highlights", "total_items"],
}

AGGREGATE_SCHEMAS = {
    "questions": AGGREGATE_QUESTIONS_SCHEMA,
    "confusion": AGGREGATE_CONFUSION_SCHEMA,
    "anecdotes": AGGREGATE_ANECDOTES_SCHEMA,
}


# Prompt config: analysis type -> (system_prompt, input_format, temperature, schema)
# Note: "questions" is handled separately via two-pass (extract → filter).
ANALYSIS_CONFIG = {
    "summary": (SUMMARY_SYSTEM, "text", 0.2, SUMMARY_SCHEMA),
    "confusion": (CONFUSION_SYSTEM, "segments", 0.2, CONFUSION_SCHEMA),
    "anecdotes": (ANECDOTES_SYSTEM, "text", 0.4, ANECDOTES_SCHEMA),
}

AGGREGATE_PROMPTS = {
    "questions": AGGREGATE_QUESTIONS_SYSTEM,
    "confusion": AGGREGATE_CONFUSION_SYSTEM,
    "anecdotes": AGGREGATE_ANECDOTES_SYSTEM,
}


# ── OllamaClient ────────────────────────────────────────────────────

class OllamaClient:
    """HTTP client for Ollama API using stdlib urllib.

    Adapted from the pattern in stan/hybrid_query/extract_llm.py.
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
        schema: dict | None = None,
        num_predict: int | None = None,
    ) -> str:
        """Generate text completion. Returns raw response string.

        Passing ``schema`` (a JSON Schema) constrains decoding to that shape,
        which is far more reliable than asking for JSON in the prompt.
        ``json_mode`` is the weaker fallback: valid JSON, arbitrary shape.
        """
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature, "num_ctx": NUM_CTX},
        }
        if num_predict is not None:
            payload["options"]["num_predict"] = num_predict
        if system:
            payload["system"] = system
        if schema is not None:
            payload["format"] = schema
        elif json_mode:
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
        schema: dict | None = None,
        num_predict: int | None = None,
    ) -> dict:
        """Generate a response and parse it as JSON.

        Strips markdown code fences and retries on parse failure.  When
        ``schema`` is given the response shape is grammar-constrained, so
        fence-stripping should never trigger.  Parse failures still happen
        when the model loops until ``num_predict`` cuts it off, and a parse
        can succeed with control characters from mis-escaped LaTeX; both are
        retried, at a higher temperature so the retry does not retrace the
        same path.
        """
        for attempt in range(1 + retries):
            attempt_temp = temperature + RETRY_TEMPERATURE_STEP * attempt
            raw = self.generate(prompt, system, attempt_temp, timeout,
                                json_mode=True, schema=schema,
                                num_predict=num_predict)
            cleaned = raw.strip()
            # Strip markdown code fences (```json ... ```)
            if cleaned.startswith("```"):
                lines = cleaned.split("\n")
                lines = [l for l in lines if not l.strip().startswith("```")]
                cleaned = "\n".join(lines)
            try:
                parsed = json.loads(cleaned)
            except json.JSONDecodeError:
                parsed = None
            if parsed is None:
                if attempt < retries:
                    print("    [RETRY]  JSON parse failed, retrying...")
                    continue
                # Last resort: extract first JSON object from response
                match = re.search(r"\{.*\}", cleaned, re.DOTALL)
                if match:
                    try:
                        parsed = json.loads(match.group())
                    except json.JSONDecodeError:
                        pass
                if parsed is None:
                    raise ValueError(
                        f"Could not parse JSON from LLM response:\n{raw[:500]}"
                    )
            if not _has_control_chars(parsed):
                return parsed
            if attempt < retries:
                print("    [RETRY]  Control characters in response "
                      "(single-backslash LaTeX), retrying...")
                continue
            raise ValueError(
                "Response contains control characters (single-backslash "
                f"LaTeX):\n{raw[:500]}"
            )


def _has_control_chars(node) -> bool:
    """True if any string in a parsed response holds a tab, form feed, etc.

    These are the signature of LaTeX written with single backslashes inside
    JSON: "\\frac" and "\\text" decode as form feed + "rac" and tab + "ext".
    The JSON grammar Ollama enforces allows only those escapes, so a model in
    that mode cannot write "\\underline" or "\\hat" at all and tends to loop on
    the few commands it can (observed on 010-lecture-03, 2026-10-03).
    """
    if isinstance(node, str):
        return any(c in node for c in "\t\f\b\r\v")
    if isinstance(node, dict):
        return any(_has_control_chars(v) for v in node.values())
    if isinstance(node, list):
        return any(_has_control_chars(v) for v in node)
    return False


# ── Helpers ──────────────────────────────────────────────────────────

@dataclass(frozen=True)
class LectureID:
    """Identifies one lecture recording, with or without a section.

    Two naming conventions are in play.  The Spring 2026 corpus came from
    Panopto exports with a single global capture number and no usable section
    marker (``section=None``).  Recordings from Fall 2026 onward are captured
    per section, so the same lecture is delivered twice and the section is
    part of the identity — section 010 lecture 2 and section 011 lecture 2
    are different recordings of the same material and must not overwrite
    each other.
    """

    section: str | None
    number: int

    @property
    def key(self) -> str:
        """Filename-safe identifier used for analysis output paths."""
        if self.section is None:
            return f"lecture_{self.number:03d}"
        return f"{self.section}-lecture-{self.number:02d}"

    @property
    def label(self) -> str:
        """Short human-readable label for console output and reports."""
        if self.section is None:
            return f"Lecture {self.number:03d}"
        return f"Sec {self.section} Lecture {self.number:02d}"

    @property
    def name(self) -> str:
        """Bare identifier for interpolation mid-sentence in prompts."""
        if self.section is None:
            return str(self.number)
        return f"{self.number}, section {self.section}"

    @property
    def sort_key(self) -> tuple[str, int]:
        """Total order over both conventions: legacy corpus first, then sections.

        Defined explicitly rather than via ``order=True`` because a generated
        comparison would compare ``section`` directly and raise ``TypeError``
        as soon as a legacy id (``None``) met a sectioned one.
        """
        return (self.section or "", self.number)

    def __lt__(self, other: "LectureID") -> bool:
        if not isinstance(other, LectureID):
            return NotImplemented
        return self.sort_key < other.sort_key

    @property
    def prompt_label(self) -> str:
        """Lecture designation as presented to the LLM.

        The legacy form is kept verbatim ("Lecture N of 39") because the
        analysis prompts were tuned against it.
        """
        if self.section is None:
            return f"Lecture {self.number} of 39"
        return f"Lecture {self.number}, section {self.section}"


# Panopto export: CHEG231-010_Chemical_Engineering_Thermodynamics_NNN_1_HASH
_LEGACY_PATTERN = re.compile(r"_(\d{3})_1_")
# Per-section capture: NNN-lecture-MM (e.g. 010-lecture-02)
_SECTION_PATTERN = re.compile(r"^(\d{3})-lecture-(\d+)", re.IGNORECASE)


def extract_lecture_id(filename: str) -> LectureID | None:
    """Parse a LectureID from a transcript or analysis filename.

    Recognizes both the legacy Panopto pattern and the per-section pattern;
    returns None if neither matches.
    """
    match = _SECTION_PATTERN.match(filename)
    if match:
        return LectureID(section=match.group(1), number=int(match.group(2)))

    match = _LEGACY_PATTERN.search(filename)
    if match:
        return LectureID(section=None, number=int(match.group(1)))

    # Analysis outputs are named from LectureID.key, so re-parse that form too.
    match = re.match(r"^lecture_(\d{3})_", filename)
    if match:
        return LectureID(section=None, number=int(match.group(1)))

    return None


def find_transcripts(
    transcript_dir: Path,
    filter_str: str | None = None,
) -> list[tuple[LectureID, Path, Path]]:
    """Find transcript file pairs and return sorted (lecture_id, txt, json).

    Only returns pairs where both .txt and .json exist.  Sections sort
    together, with the legacy unsectioned corpus first.
    """
    txt_files = sorted(transcript_dir.glob("*.txt"))
    results = []
    for txt_path in txt_files:
        json_path = txt_path.with_suffix(".json")
        if not json_path.exists():
            continue
        if filter_str and filter_str not in txt_path.name:
            continue
        lid = extract_lecture_id(txt_path.name)
        if lid is not None:
            results.append((lid, txt_path, json_path))
    results.sort(key=lambda x: x[0].sort_key)
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
    analysis_dir: Path, analysis_type: str, lid: LectureID
) -> Path:
    """Compute output path for a per-lecture analysis result."""
    subdir = OUTPUT_SUBDIRS[analysis_type]
    return analysis_dir / subdir / f"{lid.key}_{analysis_type}.json"


def make_envelope(
    lid: LectureID,
    source_file: str,
    analysis_type: str,
    model: str,
    analysis_seconds: float,
    result: dict,
    notation: dict | None = None,
    instructor: str | None = None,
) -> dict:
    """Wrap an LLM result in a consistent metadata envelope.

    ``lecture_number`` is retained unchanged so the Spring 2026 result files
    stay readable; ``lecture_id`` and ``section`` are the fields that
    disambiguate multi-section semesters.
    """
    envelope = {
        "lecture_id": lid.key,
        "section": lid.section,
        "lecture_number": lid.number,
        "source_file": source_file,
        "analysis_type": analysis_type,
        "model": model,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "analysis_seconds": round(analysis_seconds, 2),
    }
    if instructor is not None:
        envelope["instructor"] = instructor
    if notation is not None:
        envelope["notation"] = notation
    envelope["result"] = result
    return envelope


# ── Notation grounding ───────────────────────────────────────────────

def lecture_instructor(lid: LectureID) -> str | None:
    """Who taught a lecture, if not the course's usual instructor.

    From course_schedule.json "instructors": "lectures" keyed by lecture id
    (010-lecture-20) or by number (both sections).  Returns None for the
    default instructor and for the legacy corpus.
    """
    if lid.section is None or not COURSE_SCHEDULE_FILE.exists():
        return None
    data = json.loads(COURSE_SCHEDULE_FILE.read_text(encoding="utf-8"))
    info = data.get("instructors", {})
    lectures = info.get("lectures", {})
    name = lectures.get(lid.key) or lectures.get(str(lid.number))
    return name if name and name != info.get("default") else None


def instructor_note(lid: LectureID) -> str:
    """Prompt sentence for a lecture taught by someone else ('' otherwise).

    The system prompts say "the instructor"; without this, a summary of a
    co-instructor's lecture would read as the usual instructor's.
    """
    name = lecture_instructor(lid)
    if not name:
        return ""
    return (f"Note: this lecture was taught by {name}, a co-instructor, not the "
            f"course's usual instructor. Refer to the lecturer as {name}.\n\n")


def lecture_chapters(lid: LectureID) -> list[int]:
    """Textbook chapters a lecture covers, from lecture_chapters.json.

    Looks up the full lecture id first (an override for one section), then the
    lecture number (shared by sections taught in parallel), then "default".
    The core chapters (1-4) are always added by stan.notation, so an unmapped
    lecture still gets core grounding.
    """
    if not LECTURE_CHAPTERS_FILE.exists():
        return []
    data = json.loads(LECTURE_CHAPTERS_FILE.read_text(encoding="utf-8"))
    overrides = data.get("overrides", {})
    if lid.key in overrides:
        return [int(c) for c in overrides[lid.key]]
    if lid.section is not None:
        lectures = data.get("lectures", {})
        if str(lid.number) in lectures:
            return [int(c) for c in lectures[str(lid.number)]]
    return [int(c) for c in data.get("default", [])]


def notation_grounding(
    lid: LectureID, transcript: str, chapters: list[int] | None = None
) -> tuple[str, dict]:
    """System-prompt addition and provenance record for one lecture."""
    chapters = lecture_chapters(lid) if chapters is None else chapters
    block = primer.grounding_block(
        chapters, text=transcript, symbol_limit=NOTATION_SYMBOL_LIMIT
    )
    record = {
        "chapters": primer.chapters_in_scope(chapters),
        "primers": primer.primer_version(),
    }
    return block, record


# ── Core analysis ────────────────────────────────────────────────────

def analyze_one(
    analysis_type: str,
    lid: LectureID,
    txt_path: Path,
    json_path: Path,
    client: OllamaClient,
    analysis_dir: Path,
    notation: bool = True,
    chapters: list[int] | None = None,
) -> Path:
    """Run a single analysis type on one lecture. Returns the output path.

    With ``notation`` (the default), analysis types in NOTATION_TYPES get the
    textbook notation primer for the lecture's chapters appended to their
    system prompt; ``chapters`` overrides lecture_chapters.json.
    """
    # Questions uses a two-pass pipeline (extract → filter)
    if analysis_type == "questions":
        return _analyze_questions_two_pass(
            lid, txt_path, json_path, client, analysis_dir
        )

    system_prompt, input_format, temperature, schema = ANALYSIS_CONFIG[analysis_type]

    # Build the user prompt with the appropriate input
    if input_format == "text":
        content = load_transcript_text(txt_path)
    else:
        segments = load_transcript_segments(json_path)
        content = format_segments_with_timestamps(segments)

    user_prompt = (
        f"Analyze this lecture transcript ({lid.prompt_label}) "
        f"and produce a structured JSON response.\n\n"
        f"{instructor_note(lid)}"
        f"TRANSCRIPT:\n{content}"
    )

    notation_record = None
    if notation and analysis_type in NOTATION_TYPES:
        block, notation_record = notation_grounding(lid, content, chapters)
        system_prompt = f"{system_prompt}\n\n{SUMMARY_NOTATION_NOTE}\n\n{block}"

    print(f"  [{analysis_type.upper():10s}] {lid.label:24s}...", end="", flush=True)
    t0 = time.time()
    result = client.generate_json(
        user_prompt, system=system_prompt, temperature=temperature,
        schema=schema, num_predict=NUM_PREDICT.get(analysis_type),
        retries=RETRIES.get(analysis_type, 1),
    )
    elapsed = time.time() - t0
    print(f" {elapsed:.1f}s")

    # Wrap in envelope and write
    out = output_path(analysis_dir, analysis_type, lid)
    out.parent.mkdir(parents=True, exist_ok=True)
    envelope = make_envelope(
        lid, txt_path.name, analysis_type, client.model, elapsed, result,
        notation=notation_record, instructor=lecture_instructor(lid),
    )
    out.write_text(json.dumps(envelope, indent=2, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    return out


def _analyze_questions_two_pass(
    lid: LectureID,
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
        f"({lid.prompt_label}). Copy each question VERBATIM.\n\n"
        f"{instructor_note(lid)}"
        f"TRANSCRIPT:\n{content}"
    )

    print(f"  [QUESTIONS ] {lid.label:24s} pass 1 (extract)...",
          end="", flush=True)
    t0 = time.time()
    candidates = client.generate_json(
        extract_prompt, system=QUESTIONS_EXTRACT_SYSTEM, temperature=0.2,
        schema=QUESTIONS_EXTRACT_SCHEMA,
    )
    t1 = time.time()
    n_candidates = len(candidates.get("candidates", []))
    print(f" {t1 - t0:.1f}s ({n_candidates} candidates)")

    # ── Pass 2: Filter & classify ──
    candidate_text = json.dumps(candidates.get("candidates", []), indent=1)
    filter_prompt = (
        f"Here are {n_candidates} candidate questions extracted from Lecture "
        f"{lid.name}. Classify, rank by significance, and keep the 5–15 "
        f"most important.\n\n{instructor_note(lid)}CANDIDATES:\n{candidate_text}"
    )

    print(f"  [QUESTIONS ] {lid.label:24s} pass 2 (filter)...",
          end="", flush=True)
    result = client.generate_json(
        filter_prompt, system=QUESTIONS_FILTER_SYSTEM, temperature=0.2,
        schema=QUESTIONS_FILTER_SCHEMA,
    )
    t2 = time.time()
    n_final = len(result.get("questions", []))
    elapsed = t2 - t0
    print(f" {t2 - t1:.1f}s ({n_final} kept)")

    # Store pass-1 candidate count in result for diagnostics
    result["candidates_extracted"] = n_candidates

    # Wrap in envelope and write
    out = output_path(analysis_dir, "questions", lid)
    out.parent.mkdir(parents=True, exist_ok=True)
    envelope = make_envelope(
        lid, txt_path.name, "questions", client.model, elapsed, result,
        instructor=lecture_instructor(lid),
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
    notation: bool = True,
    chapters: list[int] | None = None,
) -> None:
    """Batch-analyze transcripts for a single analysis type."""
    transcripts = find_transcripts(transcript_dir, filter_str)
    if not transcripts:
        print("No transcript files found.", file=sys.stderr)
        sys.exit(1)

    # Classify lectures
    valid = []
    skipped_short = []
    for lid, txt, jsn in transcripts:
        ok, wc = is_valid_lecture(txt)
        if ok:
            valid.append((lid, txt, jsn, wc))
        else:
            skipped_short.append((lid, wc))

    print(f"Found {len(transcripts)} transcript(s) "
          f"({len(valid)} valid, {len(skipped_short)} skipped)")
    print(f"Model: {client.model}")
    print(f"Analysis: {analysis_type}")
    if analysis_type in NOTATION_TYPES:
        print(f"Notation primer: {'on' if notation else 'off'}"
              + (f" (chapters {chapters})" if notation and chapters else ""))
    print()

    if not go:
        print("Dry run (pass --go to analyze):\n")
        for lid, wc in skipped_short:
            print(f"  [SKIP]    {lid.label:24s} ({wc} words)")
        for lid, txt, jsn, wc in valid:
            out = output_path(analysis_dir, analysis_type, lid)
            if out.exists() and not overwrite:
                print(f"  [EXISTS]  {lid.label:24s} (already analyzed)")
            else:
                print(f"  [PENDING] {lid.label:24s} ({wc} words)")
        print(f"\nRerun with --go to analyze to "
              f"{analysis_dir / OUTPUT_SUBDIRS[analysis_type]}")
        return

    success = 0
    skipped = 0
    failed = 0
    t0_total = time.time()

    for lid, txt, jsn, wc in valid:
        out = output_path(analysis_dir, analysis_type, lid)
        if out.exists() and not overwrite:
            print(f"  [SKIP]       {lid.label:24s} (already analyzed)")
            skipped += 1
            continue
        try:
            analyze_one(analysis_type, lid, txt, jsn, client, analysis_dir,
                        notation=notation, chapters=chapters)
            success += 1
        except Exception as e:
            print(f"  [FAIL]       {lid.label:24s}: {e}")
            failed += 1

    elapsed_total = time.time() - t0_total
    print(f"\n{success} analyzed, {skipped} skipped, {failed} failed "
          f"({elapsed_total:.0f}s total)")


# ── Result loading ───────────────────────────────────────────────────

def find_results(
    subdir: Path,
    analysis_type: str,
    section: str | None = None,
) -> list[tuple[LectureID, Path]]:
    """Find per-lecture result files, sorted, optionally scoped to a section.

    Matches both naming conventions and excludes the ``semester_*``
    aggregate outputs that live in the same directory.
    """
    results = []
    for f in sorted(subdir.glob(f"*_{analysis_type}.json")):
        if f.name.startswith("semester_"):
            continue
        lid = extract_lecture_id(f.name)
        if lid is None:
            continue
        if section is not None and lid.section != section:
            continue
        results.append((lid, f))
    results.sort(key=lambda x: x[0].sort_key)
    return results


# ── Aggregation ──────────────────────────────────────────────────────

def aggregate(
    analysis_type: str,
    analysis_dir: Path,
    client: OllamaClient,
    go: bool = False,
    section: str | None = None,
) -> None:
    """Load all per-lecture results for a type and find cross-lecture patterns."""
    if analysis_type not in AGGREGATE_PROMPTS:
        print(f"No aggregation defined for '{analysis_type}'.", file=sys.stderr)
        sys.exit(1)

    subdir = analysis_dir / OUTPUT_SUBDIRS[analysis_type]
    if not subdir.exists():
        print(f"No {analysis_type} results found at {subdir}.", file=sys.stderr)
        print(f"Run per-lecture analysis first: "
              f"python -m stan.lecture_tools.analyze {analysis_type} batch --go")
        sys.exit(1)

    # Load per-lecture results
    result_files = find_results(subdir, analysis_type, section=section)
    if not result_files:
        scope = f" for section {section}" if section else ""
        print(f"No {analysis_type} result files in {subdir}{scope}.", file=sys.stderr)
        sys.exit(1)

    all_results = []
    for lid, f in result_files:
        data = json.loads(f.read_text(encoding="utf-8"))
        all_results.append({
            "lecture": lid.key,
            "lecture_number": lid.number,
            "result": data["result"],
        })

    print(f"Aggregating {len(all_results)} {analysis_type} results"
          f"{f' (section {section})' if section else ''}")
    suffix = f"_{section}" if section else ""
    out_path = subdir / f"semester_{analysis_type}{suffix}.json"

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
        user_prompt, system=system_prompt, temperature=0.3,
        schema=AGGREGATE_SCHEMAS[analysis_type],
    )
    elapsed = time.time() - t0
    print(f" {elapsed:.1f}s")

    envelope = make_envelope(
        lid=LectureID(section=section, number=0),
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

def _report_label(lid: LectureID, class_num: int | None = None) -> str:
    """Build a report heading for one lecture.

    Sectioned recordings are already numbered in teaching order, so the
    capture-number cross-reference only applies to the legacy Panopto corpus,
    whose capture numbers run roughly backwards.
    """
    if lid.section is not None:
        name = lecture_instructor(lid)
        return f"Sec {lid.section} Class {lid.number:02d}" + (f" (taught by {name})" if name else "")
    if class_num is not None:
        return f"Class {class_num:02d} (capture {lid.number:03d})"
    return f"Lecture {lid.number:03d}"


def _format_summary_report(data: dict, lid: LectureID, class_num: int | None = None) -> str:
    """Format a single summary result as readable text."""
    r = data["result"]
    lines = [f"{_report_label(lid, class_num)}: {r.get('title', '(no title)')}"]

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


def _format_questions_report(data: dict, lid: LectureID, class_num: int | None = None) -> str:
    """Format a single questions result as readable text."""
    r = data["result"]
    # Drop non-dict entries: the model occasionally emits a stray schema key
    # (e.g. the bare string "context") as a list item.  Observed on 3 of the
    # 35 Spring 2026 lectures with llama3.1:8B.
    questions = [q for q in r.get("questions", []) if isinstance(q, dict)]
    student_q = sum(1 for q in questions if q.get("speaker") == "student")
    label = _report_label(lid, class_num)
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


def _format_confusion_report(data: dict, lid: LectureID, class_num: int | None = None) -> str:
    """Format a single confusion result as readable text."""
    r = data["result"]
    points = r.get("confusion_points", [])
    label = _report_label(lid, class_num)
    lines = [f"{label}: {len(points)} confusion points"]
    for p in points:
        ts = p.get("timestamp", "")
        sev = p.get("severity", "?")
        topic = p.get("topic", "")
        desc = p.get("description", "")
        lines.append(f"  [{ts}] [{sev}] {topic}: {desc}")
    return "\n".join(lines)


def _format_anecdotes_report(data: dict, lid: LectureID, class_num: int | None = None) -> str:
    """Format a single anecdotes result as readable text."""
    r = data["result"]
    items = r.get("items", [])
    label = _report_label(lid, class_num)
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


def report(analysis_type: str, analysis_dir: Path, section: str | None = None) -> None:
    """Print a formatted semester report for an analysis type."""
    subdir = analysis_dir / OUTPUT_SUBDIRS[analysis_type]
    if not subdir.exists():
        print(f"No {analysis_type} results found at {subdir}.", file=sys.stderr)
        print(f"Run analysis first: "
              f"python -m stan.lecture_tools.analyze {analysis_type} batch --go")
        sys.exit(1)

    result_files = find_results(subdir, analysis_type, section=section)
    if not result_files:
        scope = f" for section {section}" if section else ""
        print(f"No {analysis_type} result files in {subdir}{scope}.", file=sys.stderr)
        sys.exit(1)

    # Map legacy captures to teaching order; sectioned recordings are already
    # numbered in teaching order, so they pass their own number through.
    teaching_order = _load_teaching_order()
    entries = []
    for lid, f in result_files:
        data = json.loads(f.read_text(encoding="utf-8"))
        if lid.section is None:
            class_num = teaching_order.get(f"{lid.number:03d}", lid.number)
        else:
            class_num = lid.number
        entries.append((lid, class_num, data))
    entries.sort(key=lambda e: (e[0].section or "", e[1]))

    formatter = REPORT_FORMATTERS[analysis_type]

    has_order = bool(teaching_order)
    sections = sorted({lid.section for lid, _, _ in entries if lid.section})
    print(f"\n{'=' * 65}")
    plural = "lecture" if len(entries) == 1 else "lectures"
    print(f"  CHEG 231 — Semester {analysis_type.title()} Report "
          f"({len(entries)} {plural})")
    if sections:
        print(f"  Sections: {', '.join(sections)}")
    if has_order:
        print(f"  (sorted by teaching order)")
    print(f"{'=' * 65}\n")

    for lid, class_num, data in entries:
        if has_order:
            print(formatter(data, lid, class_num=class_num))
        else:
            print(formatter(data, lid))
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
    add_notation_args(parser)


def add_notation_args(parser):
    """Notation-primer options (apply to summary only; see NOTATION_TYPES)."""
    parser.add_argument(
        "--no-notation", action="store_true",
        help="Omit the textbook notation primer from the summary prompt",
    )
    parser.add_argument(
        "--chapters", type=_chapter_list, default=None,
        help="Comma-separated chapters for the notation primer, overriding "
             "lecture_chapters.json (e.g. 6 or 8,9)",
    )


def _chapter_list(value: str) -> list[int]:
    return [int(c) for c in value.split(",") if c.strip()]


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
        lid = extract_lecture_id(txt_path.name)
        if lid is None:
            print(f"Warning: could not parse a lecture id from "
                  f"{txt_path.name!r}, using lecture 0")
            lid = LectureID(section=None, number=0)
        analysis_dir = Path(args.output_dir)
        analyze_one(analysis_type, lid, txt_path, json_path, client, analysis_dir,
                    notation=not args.no_notation, chapters=args.chapters)

    elif args.mode == "batch":
        analyze_batch(
            analysis_type,
            transcript_dir=Path(args.transcript_dir),
            analysis_dir=Path(args.output_dir),
            client=client,
            go=args.go,
            filter_str=args.filter,
            overwrite=args.overwrite,
            notation=not args.no_notation,
            chapters=args.chapters,
        )


def cmd_aggregate(args):
    """Run cross-lecture aggregation."""
    client = OllamaClient(model=args.model)
    aggregate(
        args.type,
        analysis_dir=Path(args.output_dir),
        client=client,
        go=args.go,
        section=args.section,
    )


def cmd_report(args):
    """Print a formatted semester report."""
    report(args.type, analysis_dir=Path(args.output_dir), section=args.section)


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
            notation=not args.no_notation,
            chapters=args.chapters,
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
        add_notation_args(p_one)

        p_batch = mode_sub.add_parser("batch", help=f"Batch analyze all transcripts")
        add_common_args(p_batch)

        p.set_defaults(func=lambda args, at=atype: cmd_analysis(args, at))

    # report
    p_rep = sub.add_parser("report", help="Print formatted semester report")
    p_rep.add_argument("type", choices=list(ANALYSIS_TYPES),
                       help="Analysis type to report on")
    p_rep.add_argument("--output-dir", type=str, default=str(DEFAULT_ANALYSIS_DIR))
    p_rep.add_argument("--section", type=str, default=None,
                       help="Limit the report to one section (e.g. 010)")
    p_rep.set_defaults(func=cmd_report)

    # aggregate
    p_agg = sub.add_parser("aggregate", help="Cross-lecture pattern analysis")
    p_agg.add_argument("type", choices=list(AGGREGATE_PROMPTS.keys()),
                       help="Analysis type to aggregate")
    p_agg.add_argument("--output-dir", type=str, default=str(DEFAULT_ANALYSIS_DIR))
    p_agg.add_argument("--model", default=MODEL)
    p_agg.add_argument("--go", action="store_true")
    p_agg.add_argument("--section", type=str, default=None,
                       help="Aggregate one section only (e.g. 010)")
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
