# HOWTO: Set up Stan and run the lecture pipeline

A practical, start-to-finish guide for getting this repo running on a new
machine and processing a batch of lecture recordings.

The pipeline has five stages:

```
video ──(ffmpeg)──> audio ──(Whisper)──> transcripts ──┬──(Ollama)──> analysis
                      ▲                                 └──(LlamaIndex)──> search index
                      │
              start here if you record
              your own audio
```

If you record your own lectures, you already have audio — skip straight to
[Step 4](#step-4-name-and-stage-your-recordings). Steps 1-3 are a one-time
machine setup.

---

## Step 1: System prerequisites

| Requirement | Why | Install |
|---|---|---|
| **ffmpeg** | Audio extraction from video; Whisper also uses it to decode `.m4a` | `sudo apt install ffmpeg` |
| **NVIDIA GPU + driver** | Transcription is ~80x realtime on GPU, ~50x slower on CPU | vendor driver |
| **Ollama** | Runs the local LLM for analysis and search synthesis | <https://ollama.com/download> |
| **uv** | Python environment manager used by this project | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |

Tested on a workstation with a 24 GB NVIDIA GPU running Ubuntu 22.04
(ffmpeg 4.4, Python 3.14).

A GPU is strongly recommended but not required — pass `--device cpu` to
`transcribe` if you don't have one, and expect roughly 50x longer runs.

---

## Step 2: Clone and build the environment

```bash
git clone https://github.com/EntropicLearners/stan.git
cd stan

uv venv                                  # creates .venv/
uv pip install -e ".[lecture,transcribe]"
```

That gets you the core package, LlamaIndex + HuggingFace embeddings
(`[lecture]`), and faster-whisper (`[transcribe]`).

### The CUDA library gotcha

faster-whisper runs on **ctranslate2**, not torch, and ctranslate2 needs
**cuBLAS 12** and **cuDNN 9**. Neither is a declared dependency, and a torch
built for CUDA 13 ships `.so.13` instead, so install them explicitly:

```bash
uv pip install nvidia-cublas-cu12 nvidia-cudnn-cu12
```

Without these, transcription dies part-way through the first encode with:

```
RuntimeError: Library libcublas.so.12 is not found or cannot be loaded
```

You do *not* need to set `LD_LIBRARY_PATH` — `transcribe.py` preloads these
libraries itself (`_preload_cuda12_libs()`).

### Pull the analysis model

```bash
ollama pull gemma4:31b     # default — best quality, see VRAM note below
ollama pull gemma4:e4b     # optional — ~50x faster, lower recall
```

### Course configuration and notation primers

Three small files under `stan/data/lectures/` describe your course. Start from
the examples:

```bash
cd stan/data/lectures
cp course_schedule.example.json course_schedule.json   # meeting days, no-class days, time slots, instructors
cp lecture_chapters.example.json lecture_chapters.json # which textbook chapters each lecture covers
cd -
```

`course_schedule.json` turns lecture numbers into dates; the recordings tracker
and the Voice Memos importer need it. `lecture_chapters.json` picks the
notation primer for each summary. Without it, summaries get the core primer
(chapters 1-4) only.

Summaries and student answers are grounded in the textbook's notation primers.
If `stan/data/grounding/` is empty, download them from cbethermo.org:

```bash
python -m stan.notation.fetch --go
```

---

## Step 3: Verify the setup

```bash
cd ~/LLM/stan
source .venv/bin/activate          # or prefix commands with .venv/bin/

ffmpeg -version | head -1
ollama list
.venv/bin/python -c "import ctranslate2; print('CUDA devices:', ctranslate2.get_cuda_device_count())"
python -m stan.lecture_tools.transcribe models
```

`CUDA devices: 1` means transcription will use the GPU. If it prints `0`,
transcription silently falls back to CPU.

**Note on torch:** `torch.cuda.is_available()` can be `False` when your torch
build expects a newer CUDA than your driver supports (for example, a cu130
wheel on a driver capped at CUDA 12.2). This does *not* affect
transcription, but it means the embedding model in `segment.py` runs on CPU.
Transcription checks ctranslate2 rather than torch precisely so this
mismatch can't quietly halve your throughput.

---

## Step 4: Name and stage your recordings

**This is the step that matters most.** Filenames *are* the metadata — the
pipeline parses section and lecture number out of them, and nothing
downstream can recover from a bad name.

Use the per-section convention:

```
NNN-lecture-MM.ext        e.g.  010-lecture-06.m4a
│   │       │                   │   │
│   │       └ lecture number,   │   └ teaching order (01, 02, 03...)
│   │         zero-padded       └ course section
└ 3-digit section
```

Drop your recordings into the audio directory:

```bash
cp ~/recordings/*.m4a stan/data/lectures/audio/
ls stan/data/lectures/audio/
```

Accepted audio extensions: `.wav`, `.mp3`, `.m4a`.

Why sections matter: the same lecture delivered to sections 010 and 011 is
two different recordings of the same material. Including the section in the
name keeps their transcripts and analyses from overwriting each other.

Sanity-check what's staged:

```bash
python -m stan.lecture_tools.acquire stan/data/lectures/audio
```

<details>
<summary><b>If you're starting from video instead</b></summary>

Put video files in `stan/data/lectures/video/` and extract the audio:

```bash
python -m stan.lecture_tools.audio                      # dry run
python -m stan.lecture_tools.audio --go                 # 16 kHz mono WAV
python -m stan.lecture_tools.audio --go --format mp3    # smaller, ~29 MB/hr
python -m stan.lecture_tools.audio --go --filter 010    # one section only
```

WAV at 16 kHz mono is Whisper's native input — no resampling, ~58 MB/hr.
Output filenames inherit the video stem, so name the videos correctly first.

For bulk Kaltura/UDCapture exports there's also
`python -m stan.lecture_tools.download` (dry run) / `--go`.
</details>

---

**Check the recordings before transcribing.** Copying audio replaces the
file's date, and lecture numbers are assigned by hand, so mix-ups happen (Fall
2026 had a re-saved copy of lecture 14 named as lecture 15). This lists every
file with its lecture date, embedded save time, duration, and a fingerprint of
the decoded audio, and flags duplicates, files saved before their lecture,
numbers past the schedule, and missing lectures:

```bash
python -m stan.lecture_tools.recordings --write     # writes stan/data/lectures/recordings.md
```

Lecture dates come from `stan/data/lectures/course_schedule.json`; add each
no-class day to it. `--strict` exits 1 if any file is flagged.

## Step 5: Transcribe

Every batch command is a **dry run by default**; add `--go` to execute.

```bash
# See what would run
python -m stan.lecture_tools.transcribe batch

# Transcribe everything not already done
python -m stan.lecture_tools.transcribe batch --go

# One section, or one file
python -m stan.lecture_tools.transcribe batch --go --filter 010
python -m stan.lecture_tools.transcribe one stan/data/lectures/audio/010-lecture-06.m4a
```

Useful flags: `--model` (default `large-v3-turbo`), `--device auto|cuda|cpu`,
`--overwrite`, `--no-prompt`.

Existing transcripts are **skipped**, so re-running the batch after adding new
recordings only processes the new ones.

Each run produces two files in `stan/data/lectures/transcripts/`:

- `010-lecture-06.txt` — plain text, one segment per line (input to analysis)
- `010-lecture-06.json` — timestamped segments plus metadata

Both are required downstream: `analyze` only picks up transcripts that have
**both** files.

**What to expect:** ~80x realtime on a 24 GB GPU with `large-v3-turbo`. A reference
batch of 8 recordings — 7.2 hours of audio — finished in 4.8 minutes.

**Two quality features worth knowing about:**

- A thermodynamics vocabulary prompt (`THERMO_PROMPT`) biases Whisper toward
  course terms — fugacity, Clausius-Clapeyron, Poynting, UNIFAC. If you teach a
  different subject, edit that constant in `transcribe.py`.
- Whisper's decoder can loop, repeating one phrase dozens of times during
  silence. Runs of 3+ identical segments are collapsed — the first occurrence
  is kept, the repeats are dropped, and the run is recorded in the JSON under
  `hallucination_loops` with its time span. Watch for `[LOOP]` lines in the
  console output. (The docstring in `transcribe.py` mentions a `[loop removed]`
  marker; no such marker is actually inserted.)

---

## Step 6: Analyze (instructor-facing)

Four per-lecture analyses, each run by a local LLM through Ollama:

| Type | What it extracts |
|---|---|
| `summary` | Topics, key concepts, key equations, narrative summary |
| `questions` | Questions asked during the lecture (two-pass: extract, then filter) |
| `confusion` | Moments of confusion, repeated explanations, self-corrections |
| `anecdotes` | Stories, analogies, jokes, real-world connections |

```bash
# Dry run first — always
python -m stan.lecture_tools.analyze all batch

# Run all four analyses on all transcripts
python -m stan.lecture_tools.analyze all batch --go

# One type, one section
python -m stan.lecture_tools.analyze summary batch --go --filter 010

# One transcript
python -m stan.lecture_tools.analyze questions one \
    stan/data/lectures/transcripts/010-lecture-06.txt
```

Results land in `stan/data/lectures/analysis/{summaries,questions,confusion,anecdotes}/`
as `010-lecture-06_summary.json` and so on. Existing results are skipped unless
you pass `--overwrite`. Transcripts under 100 words are skipped (exam days,
failed recordings).

**Notation.** Summaries are prompted with the textbook's notation primer for
the chapters the lecture covers, so key equations come out in the book's
LaTeX notation ($\dot{M}_k \hat{S}_k$, not `m_dot_k * s_k`). The chapters come
from `stan/data/lectures/lecture_chapters.json`: add each new lecture's
chapters there as the semester goes (chapters 1-4 are always included). Pass
`--chapters 6` to override it, or `--no-notation` to reproduce the ungrounded
prompt. To measure drift in any set of results:

```bash
python -m stan.notation.check stan/data/lectures/analysis/summaries
```

**Budget the time.** `gemma4:31b` runs 8 lectures x 4 types in about 150
minutes — questions is by far the slowest stage (~500-800 s/lecture), because
it makes two passes. Start it and walk away.

### Model and VRAM notes — read before you change `--model`

- `gemma4:31b` (the default) **does not fit a 24 GB card.** Ollama reports 31 GB
  at 32k context and offloads ~20% to CPU. It still produces the best output;
  it's just slow. The 19 GB weight figure is misleading — the KV cache is the
  rest.
- While `gemma4:31b` is resident it occupies the whole GPU, so a concurrent
  Ollama request for a *different* model fails with HTTP 500. Don't run
  analysis and a search query at the same time.
- `gemma4:e4b` fits entirely in VRAM and runs ~50x faster with lower recall —
  a good choice for iterating on prompts.
- **Context window is load-bearing.** `NUM_CTX` is 32768. At 16384 the models
  silently truncated and returned schema-valid *empty arrays* — which reads as
  "nothing found in this lecture," not as an error. Don't lower it.
- Question extraction is the most fragile stage in the pipeline. Spot-check it.
  Smaller models have fabricated timestamps and returned their own
  chain-of-thought as a question.

### Student-facing recaps for Canvas

```bash
python -m stan.lecture_tools.recap 13-15 -o recap.md
```

Merges both sections' summaries of each lecture into a short recap addressed
to the class, with the lecture date from `course_schedule.json`. The model
(gemma4, with the notation primer) writes only the title and paragraph, and
picks which key equations to show by number; the equations themselves are
copied verbatim from the summaries. A recap that fails `stan.notation.check`
or contains math not found in its summaries is retried, then replaced by the
summary topics. `--no-llm` skips the model. Convert the Markdown for Canvas
with `md_canvas.py`.

### Reports and aggregation

```bash
python -m stan.lecture_tools.analyze report summary
python -m stan.lecture_tools.analyze report questions --section 010
python -m stan.lecture_tools.analyze aggregate confusion --go --section 010
```

`analysis/` accumulates across semesters, so use `--section` to scope a report
to one delivery. Aggregation looks for semester-scale patterns and needs more
than a handful of lectures to say anything meaningful.

---

## Step 7: Build the search index (student-facing)

```bash
python -m stan.lecture_tools.segment build stan/data/lectures/transcripts/010-lecture-06.txt
python -m stan.lecture_tools.segment query "What is entropy generation?"
python -m stan.lecture_tools.segment query --retrieve-only "second law"
```

Transcripts are chunked (512 tokens, 50 overlap), embedded with
`BAAI/bge-large-en-v1.5`, and persisted to `stan/data/lectures/index/`.
`--retrieve-only` skips LLM synthesis and shows the raw matching chunks —
useful for checking what retrieval actually found before blaming the model.

**Two caveats:**

1. `build` takes **one transcript** and **overwrites** the persisted index.
   It does not accumulate across a directory. The README's `segment build`
   without an argument will fail. Multi-transcript indexing isn't built yet.
2. Embeddings run through sentence-transformers on torch, so if torch can't
   see your GPU (see Step 3) this stage runs on CPU.

---

## Where everything lives

```
stan/data/lectures/
├── video/           # source recordings          (gitignored)
├── audio/           # extracted or recorded audio (gitignored)
├── transcripts/     # .txt + .json per lecture    (gitignored)
├── analysis/        # summaries/ questions/ confusion/ anecdotes/  (gitignored)
├── index/           # persisted vector index      (gitignored)
├── lecture_chapters.json  # lecture → textbook chapters, for the notation primer  (committed)
├── course_schedule.json   # meeting days and no-class dates → lecture dates  (committed)
├── recordings.md/.csv     # recording tracker, regenerated  (gitignored)
└── lecture_order.json   # legacy Panopto capture number → teaching order  (committed)
```

Everything under `stan/data/lectures/` is gitignored except the two
hand-edited mappings: media is too large, and transcripts, analysis JSON and
the vector index are all regenerated by the pipeline.

`lecture_order.json` only matters for the legacy Spring 2026 Panopto corpus,
where capture numbers ran roughly backwards. Per-section recordings are already
numbered in teaching order and bypass it entirely.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `Library libcublas.so.12 is not found` | ctranslate2 needs cuBLAS 12 / cuDNN 9 | `uv pip install nvidia-cublas-cu12 nvidia-cudnn-cu12` |
| Transcription crawling at ~1x realtime | Running on CPU | Check `ctranslate2.get_cuda_device_count()`; force `--device cuda` |
| `No audio files found` | Wrong directory or unsupported extension | Files go in `stan/data/lectures/audio/`, as `.wav` / `.mp3` / `.m4a` |
| Transcript exists but analysis skips it | `analyze` requires both `.txt` **and** `.json` | Re-transcribe that file |
| Analysis returns empty arrays | Context truncation | Confirm `NUM_CTX = 32768` in `analyze.py` |
| Ollama HTTP 500 mid-run | A 31b model is holding the whole GPU | Don't run two models at once |
| Analysis times out | `gemma4:31b` partly on CPU | `OLLAMA_TIMEOUT` is 1800 s; raise it or use `--model gemma4:e4b` |
| Repeated phrases in transcript | Whisper hallucination loop | Runs of 3+ are auto-collapsed to the first occurrence; check `hallucination_loops` in the JSON |
| `RuntimeWarning: ... found in sys.modules` from `acquire` | `stan.lecture_tools.__init__` imports the module before `-m` runs it | Harmless; ignore |
| Analysis output overwrote another section | Filename missing the section prefix | Rename to `NNN-lecture-MM` and re-run |

---

## Known limitations

- **Lapel mics don't capture student speech.** In one 8-lecture batch the
  analysis found 3 student questions against 72 instructor questions. The
  instructor's Socratic questions come through cleanly; the room does not.
  If cataloging student questions is a goal, fix the mic setup first — this is
  a recording problem, not a software one.
- The search index holds one transcript at a time (Step 7).
- `analysis/` mixes semesters; scope reports with `--section`.

---

## Quick reference

```bash
cd ~/LLM/stan && source .venv/bin/activate

cp ~/recordings/010-lecture-06.m4a stan/data/lectures/audio/   # 1. stage
python -m stan.lecture_tools.transcribe batch --go                   # 2. transcribe
python -m stan.lecture_tools.analyze all batch --go                  # 3. analyze
python -m stan.lecture_tools.analyze report summary --section 010    # 4. read
```

Every batch command dry-runs without `--go`, and skips completed work without
`--overwrite`. Run the dry run first, every time.
