"""
Transcribe lecture audio using faster-whisper (CTranslate2).

Generates plain text transcripts and JSON with segment timestamps
from WAV audio files extracted by audio.py.  Includes post-processing
to detect and collapse Whisper hallucination loops (repeated segments).

Usage:
    # Transcribe a single file (good for testing)
    python -m stan.lecture.transcribe one path/to/audio.wav

    # Batch transcribe all audio files
    python -m stan.lecture.transcribe batch                    # dry run
    python -m stan.lecture.transcribe batch --go               # transcribe all
    python -m stan.lecture.transcribe batch --go --filter 010  # section 010 only

    # List available models
    python -m stan.lecture.transcribe models

Requires: pip install faster-whisper (or pip install -e '.[transcribe]')
Hardware: NVIDIA GPU recommended (large-v3 uses ~10GB VRAM on 4090)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

DEFAULT_AUDIO_DIR = Path(__file__).resolve().parents[1] / "data" / "lectures" / "audio"
DEFAULT_TRANSCRIPT_DIR = Path(__file__).resolve().parents[1] / "data" / "lectures" / "transcripts"

AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a"}

# Vocabulary prompt: biases Whisper toward expected technical terms.
# This is passed as initial_prompt to condition the model's decoder.
# Terms are drawn from the course textbook (Sandler) and observed
# misrecognitions in initial transcription runs.
THERMO_PROMPT = (
    "Chemical engineering thermodynamics lecture. "
    "Terms: fugacity, fugacity coefficient, enthalpy, entropy, "
    "Gibbs free energy, Helmholtz energy, "
    "Clausius-Clapeyron equation, van der Waals, "
    "Peng-Robinson equation of state, Raoult's law, "
    "Henry's law, Antoine equation, Joule-Thomson, "
    "adiabatic, isothermal, isobaric, isochoric, "
    "exothermic, endothermic, "
    "chemical potential, partial molar, activity coefficient, "
    "equation of state, ideal gas, compressibility factor, "
    "phase equilibrium, vapor-liquid equilibrium, VLE, "
    "Maxwell relations, departure functions, residual properties, "
    "excess Gibbs energy, UNIFAC, NRTL, Margules, van Laar, "
    "Carnot cycle, Rankine cycle, refrigeration cycle, "
    "Poynting correction factor, molar volume, "
    "critical point, critical temperature, critical pressure, "
    "acentric factor, vapor pressure, saturation, "
    "binodal, spinodal, subcooled, superheated, supercritical, "
    "heat capacity, latent heat, heat of vaporization, "
    "Kirkbride, Jupyter notebook."
)

# Hallucination detection parameters
REPEAT_THRESHOLD = 3  # minimum consecutive identical segments to flag as loop

# Model presets: name -> (description, approx VRAM)
MODELS = {
    "large-v3": ("Most accurate, best for technical vocabulary", "~10 GB"),
    "medium": ("Good accuracy, faster", "~5 GB"),
    "small": ("Decent accuracy, much faster", "~2 GB"),
    "base": ("Basic accuracy, very fast", "~1 GB"),
    "tiny": ("Lowest accuracy, fastest (testing only)", "~0.5 GB"),
}


def collapse_hallucination_loops(
    segments: list[dict],
    threshold: int = REPEAT_THRESHOLD,
) -> tuple[list[dict], list[dict]]:
    """Detect and collapse consecutive repeated segments (Whisper hallucinations).

    Whisper's autoregressive decoder can enter loops where it repeats the same
    short phrase (e.g., "Elizabeth.", "Okay.", "That's a lot.") dozens of times,
    especially during silence or low-signal audio.  These loops share diagnostic
    signatures: identical text in 3+ consecutive segments, often with exactly
    1.0-second durations and zero inter-segment gaps.

    This function keeps the *first* occurrence of each run and replaces the rest
    with a single ``[loop removed]`` marker that preserves the time span.

    Parameters
    ----------
    segments : list of dict
        Raw segments with keys ``start``, ``end``, ``text``.
    threshold : int
        Minimum run length to consider a hallucination (default 3).

    Returns
    -------
    cleaned : list of dict
        Segments with loops collapsed.
    removed : list of dict
        Records of each collapsed loop for diagnostics / the JSON metadata.
    """
    if not segments:
        return segments, []

    cleaned = []
    removed = []
    i = 0

    while i < len(segments):
        # Look ahead for a run of identical text
        j = i + 1
        while j < len(segments) and segments[j]["text"] == segments[i]["text"]:
            j += 1

        run_length = j - i
        if run_length >= threshold:
            # Keep first occurrence, record the loop
            cleaned.append(segments[i])
            loop_record = {
                "text": segments[i]["text"],
                "first_segment_index": i,
                "run_length": run_length,
                "removed_count": run_length - 1,
                "start": segments[i]["start"],
                "end": segments[j - 1]["end"],
                "duration": round(segments[j - 1]["end"] - segments[i]["start"], 2),
            }
            removed.append(loop_record)
            print(f"    [LOOP]     '{segments[i]['text']}' x{run_length} "
                  f"({loop_record['duration']:.1f}s) at {segments[i]['start']:.1f}s "
                  f"-> kept 1, removed {run_length - 1}")
            i = j
        else:
            cleaned.append(segments[i])
            i += 1

    return cleaned, removed


def load_model(model_size: str = "large-v3", device: str = "auto"):
    """Load a faster-whisper model."""
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        print("Error: faster-whisper not installed.", file=sys.stderr)
        print("Install with: pip install faster-whisper", file=sys.stderr)
        sys.exit(1)

    if device == "auto":
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"

    compute_type = "float16" if device == "cuda" else "int8"
    print(f"Loading {model_size} on {device} ({compute_type})...")
    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    return model


def transcribe_file(
    model,
    audio_path: Path,
    output_dir: Path,
    language: str = "en",
    prompt: str | None = None,
) -> tuple[Path, Path]:
    """
    Transcribe a single audio file.

    Returns (txt_path, json_path) for the outputs.
    """
    stem = audio_path.stem
    txt_path = output_dir / f"{stem}.txt"
    json_path = output_dir / f"{stem}.json"

    print(f"  [TRANSCRIBE] {audio_path.name}")
    t0 = time.time()

    segments_iter, info = model.transcribe(
        str(audio_path),
        language=language,
        beam_size=5,
        vad_filter=True,
        vad_parameters=dict(
            min_silence_duration_ms=500,
        ),
        initial_prompt=prompt,
        condition_on_previous_text=False,  # prevent context poisoning loops
        repetition_penalty=1.1,            # mild penalty on repeated tokens
        no_repeat_ngram_size=3,            # block exact 3-gram repetition
    )

    # Collect raw segments
    raw_segments = []
    for seg in segments_iter:
        raw_segments.append({
            "start": round(seg.start, 2),
            "end": round(seg.end, 2),
            "text": seg.text.strip(),
        })

    elapsed = time.time() - t0
    duration_min = info.duration / 60
    speed = info.duration / elapsed if elapsed > 0 else 0

    # Post-process: collapse hallucination loops
    segments, loops_removed = collapse_hallucination_loops(raw_segments)

    total_removed = sum(r["removed_count"] for r in loops_removed)
    print(f"  [OK]         {len(raw_segments)} raw -> {len(segments)} segments"
          f" ({total_removed} hallucinated removed), "
          f"{duration_min:.1f} min audio in {elapsed:.1f}s "
          f"({speed:.1f}x realtime)")

    text_lines = [seg["text"] for seg in segments]

    # Write plain text
    txt_path.write_text("\n".join(text_lines) + "\n", encoding="utf-8")

    # Write JSON with metadata
    json_data = {
        "source_audio": audio_path.name,
        "model": str(model.model_size_or_path) if hasattr(model, 'model_size_or_path') else "unknown",
        "initial_prompt": prompt,
        "language": info.language,
        "language_probability": round(info.language_probability, 3),
        "duration_seconds": round(info.duration, 2),
        "transcription_seconds": round(elapsed, 2),
        "num_segments_raw": len(raw_segments),
        "num_segments": len(segments),
        "hallucination_loops": loops_removed,
        "segments": segments,
    }
    json_path.write_text(json.dumps(json_data, indent=2, ensure_ascii=False) + "\n",
                         encoding="utf-8")

    return txt_path, json_path


def find_audio(audio_dir: Path, filter_str: str | None = None) -> list[Path]:
    """Find audio files in the directory."""
    files = sorted(
        p for p in audio_dir.iterdir()
        if p.suffix.lower() in AUDIO_EXTENSIONS
    )
    if filter_str:
        files = [f for f in files if filter_str in f.name]
    return files


def cmd_one(args):
    """Transcribe a single file."""
    audio_path = Path(args.audio_file)
    if not audio_path.exists():
        print(f"Error: {audio_path} not found", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prompt = None if args.no_prompt else THERMO_PROMPT
    model = load_model(args.model, args.device)
    txt_path, json_path = transcribe_file(model, audio_path, output_dir, prompt=prompt)

    print(f"\n  Text: {txt_path}")
    print(f"  JSON: {json_path}")


def cmd_batch(args):
    """Batch transcribe audio files."""
    audio_dir = Path(args.audio_dir)
    if not audio_dir.exists():
        print(f"Error: {audio_dir} not found", file=sys.stderr)
        sys.exit(1)

    files = find_audio(audio_dir, args.filter)
    if not files:
        print("No audio files found.", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output_dir)
    print(f"Found {len(files)} audio file(s)")
    print(f"Model: {args.model}")
    print()

    if not args.go:
        print("Dry run (pass --go to transcribe):\n")
        for f in files:
            existing_txt = output_dir / f"{f.stem}.txt"
            status = "[SKIP] exists" if existing_txt.exists() else "[PENDING]"
            print(f"  {status} {f.name}")
        print(f"\nRerun with --go to transcribe to {output_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    model = load_model(args.model, args.device)

    success = 0
    skipped = 0
    t0_total = time.time()

    for f in files:
        existing_txt = output_dir / f"{f.stem}.txt"
        if existing_txt.exists() and not args.overwrite:
            print(f"  [SKIP]       {f.name} (already transcribed)")
            skipped += 1
            continue
        try:
            prompt = None if args.no_prompt else THERMO_PROMPT
            transcribe_file(model, f, output_dir, prompt=prompt)
            success += 1
        except Exception as e:
            print(f"  [FAIL]       {f.name}: {e}")

    elapsed_total = time.time() - t0_total
    print(f"\n{success} transcribed, {skipped} skipped, "
          f"{len(files) - success - skipped} failed "
          f"({elapsed_total:.0f}s total)")


def cmd_models(args):
    """List available Whisper models."""
    print("Available faster-whisper models:\n")
    for name, (desc, vram) in MODELS.items():
        print(f"  {name:12s}  {desc} ({vram} VRAM)")
    print(f"\nDefault: large-v3")


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe lecture audio with faster-whisper"
    )
    sub = parser.add_subparsers(dest="command")

    # one
    p_one = sub.add_parser("one", help="Transcribe a single audio file")
    p_one.add_argument("audio_file", help="Path to audio file")
    p_one.add_argument("--output-dir", type=str, default=str(DEFAULT_TRANSCRIPT_DIR))
    p_one.add_argument("--model", default="large-v3", choices=MODELS.keys())
    p_one.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p_one.add_argument("--no-prompt", action="store_true",
                       help="Disable thermodynamics vocabulary prompt")
    p_one.set_defaults(func=cmd_one)

    # batch
    p_batch = sub.add_parser("batch", help="Batch transcribe audio files")
    p_batch.add_argument("--audio-dir", type=str, default=str(DEFAULT_AUDIO_DIR))
    p_batch.add_argument("--output-dir", type=str, default=str(DEFAULT_TRANSCRIPT_DIR))
    p_batch.add_argument("--model", default="large-v3", choices=MODELS.keys())
    p_batch.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p_batch.add_argument("--go", action="store_true", help="Actually transcribe")
    p_batch.add_argument("--overwrite", action="store_true")
    p_batch.add_argument("--filter", type=str, default=None)
    p_batch.add_argument("--no-prompt", action="store_true",
                         help="Disable thermodynamics vocabulary prompt")
    p_batch.set_defaults(func=cmd_batch)

    # models
    p_models = sub.add_parser("models", help="List available models")
    p_models.set_defaults(func=cmd_models)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
