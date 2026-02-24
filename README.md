# Stan

A LLM-based chemical engineering thermodynamics learning assistant and 
associated tools.

This repository houses a suite of learning tools for single and multicomponent
equilibrium thermodynamics, including a hybrid (LLM + regex)-based search over 
textbooks and lecture recordings, an equation-of-state library, and utilities for 
training and running local LLMs.

**Note: This repo is under active development.**

## Installation

```bash
pip install -e .               # core package
pip install -e ".[lecture]"    # + lecture search (LlamaIndex, HuggingFace embeddings)
pip install -e ".[transcribe]" # + Whisper transcription (faster-whisper, GPU)
pip install -e ".[dev]"        # + development tools (pytest, ruff)
```

## Repository Structure

```
.
├── stan/                          # Core Python package
│   ├── data/                      # Textbook index, training data, lecture transcripts
│   ├── eos/                       # Equation of state tools
│   ├── unifac/                    # IFAC tools
│   ├── lecture_tools/             # Lecture processing & search pipeline
│   │   ├── acquire.py             # File catalog for downloaded recordings
│   │   ├── audio.py               # Audio extraction from video (ffmpeg)
│   │   ├── transcribe.py          # Whisper transcription (GPU, vocab prompting)
│   │   ├── segment.py             # Chunking, embedding, vector index, search
│   │   └── download.py            # Bulk download from Kaltura export email
│   ├── hybrid_query/              # LLM + Regex search pipeline
│   │   ├── extract_llm.py         # File catalog for downloaded recordings
│   │   ├── extract_regex.py       # Audio extraction from video (ffmpeg)
│   │   ├── toc_navigator.py       # Whisper transcription (GPU, vocab prompting)
│   │   └── compare_extractors.py  # Bulk download from Kaltura export email
│   ├── rag/                       # LLM query pipelines (student, textbook search)
│   ├── eos/                       # Equation of State tools
│   ├── utils/                     # Utility functions
│   └── visualization/             # Thermodynamic plotting tools
├── apps/                          # CLI applications
├── docs/                          # Sphinx documentation & planning docs
├── examples/                      # Example scripts and notebooks
├── paper/                         # Project paper (LaTeX)
└── scratchpad/                    # Temporary folder with code examples and experiments
```

## Lecture Pipeline

Full pipeline from video recordings to searchable transcripts:

```bash
# 1. Download recordings from Kaltura export
python -m stan.lecture.download          # dry run
python -m stan.lecture.download --go     # download all

# 2. Extract audio (requires ffmpeg)
python -m stan.lecture.audio --go

# 3. Transcribe with Whisper (requires GPU, faster-whisper)
python -m stan.lecture.transcribe batch --go --filter CHEG231-010

# 4. Build the vector index from transcripts
python -m stan.lecture.segment build

# 5. Query with LLM synthesis
python -m stan.lecture.segment query "What is entropy generation?"

# 5b. Retrieve raw chunks (no LLM)
python -m stan.lecture.segment query --retrieve-only "second law"
```

## Textbook Search

Query the textbook index with TOC-aware retrieval:

```bash
python -m stan.run.query_student "What is fugacity?"
```

