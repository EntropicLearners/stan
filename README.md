# Stan

A LLM-based chemical engineering thermodynamics learning assistant and 
associated tools.

This repository houses a suite of learning tools for single and multicomponent
equilibrium thermodynamics, including a hybrid (LLM + regex)-based search over 
textbooks and lecture recordings, an equation-of-state library, and utilities for 
training and running local LLMs.

Described in E. M. Furst and V. Venkateshwaran, "Stan: An LLM-based
thermodynamics course assistant," [arXiv:2603.04657](https://arxiv.org/abs/2603.04657)
(2026). See [Citation](#citation).

**Note: This repo is under active development.**

## Installation

New to the project? See **[docs/howto.md](docs/howto.md)** for a
start-to-finish setup and lecture-processing guide.

```bash
pip install -e .               # core package
pip install -e ".[lecture]"    # + lecture search (LlamaIndex, HuggingFace embeddings)
pip install -e ".[transcribe]" # + Whisper transcription (faster-whisper, GPU)
pip install -e ".[dev]"        # + development tools (pytest, ruff)
```

## Repository Structure

```
.
├── stan/                       # Core Python package
│   ├── data/                   # Prompts and data schemas; your textbook index, primers, and lecture data (not distributed)
│   ├── eos/                    # Equation of state (not yet released)
│   ├── hybrid_query/           # Textbook search: regex + LLM extraction, TOC navigation
│   ├── lecture_tools/          # Lecture processing & search pipeline
│   │   ├── acquire.py          #   File catalog for downloaded recordings
│   │   ├── audio.py            #   Audio extraction from video (ffmpeg)
│   │   ├── transcribe.py       #   Whisper transcription (GPU, vocab prompting)
│   │   ├── segment.py          #   Chunking, embedding, vector index, search
│   │   ├── download.py         #   Bulk download from Kaltura export email
│   │   ├── analyze.py          #   Instructor-facing transcript analysis
│   │   ├── recordings.py       #   Recording tracker: dates, durations, duplicates, gaps
│   │   ├── voicememos.py       #   Import from Apple Voice Memos with real dates (Mac)
│   │   └── recap.py            #   Student-facing lecture recap for Canvas
│   ├── notation/               # Textbook notation grounding: primers into prompts, drift checker
│   ├── rag/                    # RAG over the textbook index
│   ├── unifac/                 # UNIFAC activity coefficients
│   └── utils/                  # Utility functions
├── apps/                       # CLI applications (query_student, rag_terminal)
├── docs/                       # Sphinx documentation and setup guide (docs/howto.md)
└── examples/                   # Example notebooks
```

## Lecture Pipeline

Full pipeline from video recordings to searchable transcripts:

```bash
# 1. Download recordings from Kaltura export
python -m stan.lecture_tools.download          # dry run
python -m stan.lecture_tools.download --go     # download all

# 2. Extract audio (requires ffmpeg)
python -m stan.lecture_tools.audio --go

# 3. Transcribe with Whisper (requires GPU, faster-whisper)
python -m stan.lecture_tools.transcribe batch                  # dry run
python -m stan.lecture_tools.transcribe batch --go             # large-v3-turbo, ~80x realtime
python -m stan.lecture_tools.transcribe batch --go --filter 010

# 4. Build the vector index from transcripts
python -m stan.lecture_tools.segment build

# 5. Query with LLM synthesis
python -m stan.lecture_tools.segment query "What is entropy generation?"

# 5b. Retrieve raw chunks (no LLM)
python -m stan.lecture_tools.segment query --retrieve-only "second law"
```

## Lecture Analysis (instructor-facing)

Per-lecture summaries, questions, confusion points, and anecdotes, extracted
with a local model via Ollama (default `gemma4:31b`):

```bash
# Per-lecture analysis — dry run, then run
python -m stan.lecture_tools.analyze summary batch
python -m stan.lecture_tools.analyze all batch --go        # all four analyses

# One transcript, one analysis type
python -m stan.lecture_tools.analyze questions one path/to/transcript.txt

# Formatted reports, optionally scoped to one course section
python -m stan.lecture_tools.analyze report summary
python -m stan.lecture_tools.analyze report questions --section 010

# Cross-lecture patterns
python -m stan.lecture_tools.analyze aggregate confusion --go --section 010
```

Outputs land in `stan/data/lectures/analysis/{summaries,questions,confusion,anecdotes}/`.
See `docs/howto.md` (Step 6, "Model and VRAM notes") for model selection,
context-window, and VRAM notes — all three matter more than they look.

## Textbook Search

Query the textbook index with TOC-aware retrieval. The index
(`stan/data/bindex_tab.json`) and table of contents (`stan/data/ftoc_nav_tree.json`)
are derived from the textbook and are not distributed; supply your own in the
same format (the index schema is documented in `stan/data/__init__.py`).

```bash
python apps/query_student/query_student_run.py "What is fugacity?"
```

## Notation Grounding

Left alone, the models write their own notation (lowercase $h$ and $s$,
$\dot{m}$, "Gibbs free energy", $Q - W$) instead of the textbook's.
`stan.notation` carries the chapter notation primers from
[cbethermo.org](https://cbethermo.org) into the prompts for lecture summaries
and student answers: the core conventions (chapters 1-4) plus the chapters in
scope. The primers are written for the 6th edition; equation, section, and
table numbers are rewritten to chapter references so the same text holds for
the 5th edition, whose chapters match.

```bash
python -m stan.notation.check stan/data/lectures/analysis/summaries   # measure drift
python -m stan.notation.primer 6          # print the grounding block for chapter 6
python -m stan.notation.fetch             # dry run: are the primers current?
```

Summaries take their chapters from `stan/data/lectures/lecture_chapters.json`
(or `--chapters`); `--no-notation` restores the ungrounded prompt. If
`stan/data/grounding/` is empty, run `python -m stan.notation.fetch --go`.
Example course configuration files are in `stan/data/lectures/*.example.json`;
see `docs/howto.md`.

## Citation

If you use Stan, please cite:

> E. M. Furst and V. Venkateshwaran, "Stan: An LLM-based thermodynamics course
> assistant," arXiv:2603.04657 (2026). https://doi.org/10.48550/arXiv.2603.04657

```bibtex
@misc{furst2026stan,
  title         = {Stan: An {LLM}-based thermodynamics course assistant},
  author        = {Furst, Eric M. and Venkateshwaran, Vasudevan},
  year          = {2026},
  eprint        = {2603.04657},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  doi           = {10.48550/arXiv.2603.04657},
  url           = {https://arxiv.org/abs/2603.04657}
}
```
