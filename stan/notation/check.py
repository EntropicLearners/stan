"""
Flag LLM output that drifts from the textbook's notation and terminology.

Each rule encodes one convention from the notation primers.  The checks are
regular expressions, so they are fast, deterministic, and auditable, and they
measure the failure mode before and after a prompt change.  They are
deliberately narrow: a rule that fires on correct text is worse than a missed
case, because it trains you to ignore the report.

Two kinds of text are scanned differently:
  prose  terminology rules only ("Gibbs free energy", "Joule-Thompson")
  math   symbol rules as well -- text inside $...$, \\(...\\), or a JSON value
         under a key such as ``key_equations``

Usage:
    python -m stan.notation.check stan/data/lectures/analysis/summaries
    python -m stan.notation.check answer.txt --verbose
    python -m stan.notation.check a.json b.json --json

    from stan.notation.check import check_text
    findings = check_text(answer)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

# JSON keys whose string values are equations rather than prose.
MATH_KEYS = {"key_equations", "equations", "equation"}


@dataclass(frozen=True)
class Rule:
    id: str
    pattern: re.Pattern
    message: str
    math_only: bool = False


@dataclass(frozen=True)
class Finding:
    rule: str
    message: str
    match: str
    where: str = ""


_MATH_SPAN = re.compile(r"\$\$(.+?)\$\$|\$(.+?)\$|\\\((.+?)\\\)|\\\[(.+?)\\\]", re.S)

RULES = [
    # ── Terminology (prose and math) ──
    Rule("free-energy",
         re.compile(r"\b(?:(?:Gibbs|Helmholtz)(?:'s)?\s+)?free[\s-]+energ\w*", re.I),
         'The book says "Gibbs energy" / "Helmholtz energy", not "free energy".'),
    Rule("entropy-of-universe",
         re.compile(r"entropy of the universe", re.I),
         "The book's second law is S_gen >= 0, not \"the entropy of the universe increases\"."),
    Rule("work-by-system",
         re.compile(r"\bwork (?:is )?(?:done )?by the system\b", re.I),
         "Sign convention: W is positive when done ON the system. Check the sign is not flipped."),
    Rule("joule-thompson",
         re.compile(r"Joule[\s-]+Thompson", re.I),
         'Spelled "Joule-Thomson" (William Thomson, Lord Kelvin).'),
    Rule("cop-efficiency",
         re.compile(r"\b(?:C\.?O\.?P\.?|coefficient of performance)\b[^.]{0,40}\befficiency\b", re.I),
         "A C.O.P. is not an efficiency and may exceed 1."),

    # ── Symbols (math only) ──
    Rule("first-law-sign",
         re.compile(r"=\s*(?:\\delta\s*|δ|d|Δ|\\Delta\s*)?(?:\\dot\{Q\}|Q(?:_?dot)?)\s*[-−]\s*"
                    r"(?:\\delta\s*|δ|d|Δ|\\Delta\s*)?(?:\\dot\{W\}|W)"),
         "Q - W is the 'work done BY the system' convention; the book writes Q + W (W done ON the system).",
         math_only=True),
    Rule("overbar-molar",
         # \bar{X} or \overline{X} or X̄ NOT followed by a species subscript
         re.compile(r"(?:\\bar\{\\?[A-Za-z]+\}|\\overline\{\\?[A-Za-z]+\}|[A-Za-z]\u0304)(?!\s*_)"),
         "Overbar is only for partial molar properties (with a species subscript); molar is an underbar.",
         math_only=True),
    Rule("mass-flow-lowercase",
         re.compile(r"\\dot\{m\}|\bm_?dot(?![A-Za-z])|ṁ"),
         "Mass flow rate is \\dot{M} (capital M), not \\dot{m}.",
         math_only=True),
    Rule("moles-lowercase",
         re.compile(r"\\dot\{n\}|\bn_?dot(?![A-Za-z])|\bnRT\b|\bdn/dt\b"),
         "Moles are N (capital) and molar flow is \\dot{N}; the ideal gas law is PV = NRT.",
         math_only=True),
    Rule("lowercase-property",
         # h, u, s, g as a property symbol: alone, with Δ/d, or with a
         # single-letter/number subscript (s_k, h_in).  Two-letter subscripts
         # such as g_{ij}, u_{ij} are the book's NRTL/UNIQUAC energies; h before
         # \nu is Planck's constant.
         re.compile(r"(?<![A-Za-z\\{/_])(?:\\Delta\s*|Δ|d)?([hsug])(?:_\{?(?:[A-Za-z0-9]|in|out|gen)\}?)?(?![A-Za-z(_{])(?!\s*(?:\\nu|ν))"),
         "Lowercase property symbol: the book writes molar \\underline{H} or specific \\hat{H} "
         "(likewise U, S, G).",
         math_only=True),
    Rule("cp-lowercase",
         re.compile(r"\bC_?\{?[pv]\b"),
         "Heat capacities are C_P and C_V (capital subscripts).",
         math_only=True),
    Rule("residual-symbol",
         re.compile(r"\b[HSG]\^\{?R\}?|\^\{?\\?(?:mathrm\{)?dep\}?"),
         "The book writes departure functions as (\\underline{H} - \\underline{H}^{IG}), not H^R or a 'dep' symbol.",
         math_only=True),
]


def _math_spans(text: str) -> list[str]:
    return [next(g for g in m.groups() if g is not None) for m in _MATH_SPAN.finditer(text)]


def check_text(text: str, *, is_math: bool = False, where: str = "") -> list[Finding]:
    """Check one string.  ``is_math`` treats the whole string as an equation."""
    findings = []
    math = [text] if is_math else _math_spans(text)
    for rule in RULES:
        targets = math if rule.math_only else [text]
        for target in targets:
            for m in rule.pattern.finditer(target):
                findings.append(Finding(rule.id, rule.message, m.group(0).strip(), where))
    return findings


def _walk_json(node, path: str = "", math: bool = False):
    if isinstance(node, dict):
        for key, value in node.items():
            yield from _walk_json(value, f"{path}.{key}" if path else key,
                                  math or key in MATH_KEYS)
    elif isinstance(node, list):
        for i, value in enumerate(node):
            yield from _walk_json(value, f"{path}[{i}]", math)
    elif isinstance(node, str):
        yield path, node, math


def check_file(path: Path) -> list[Finding]:
    if path.suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        # Analysis envelopes: only the LLM's result is under test.
        data = data.get("result", data) if isinstance(data, dict) else data
        findings = []
        for where, text, math in _walk_json(data):
            findings += check_text(text, is_math=math, where=where)
        return findings
    return check_text(path.read_text(encoding="utf-8"))


def _expand(paths: list[str]) -> list[Path]:
    files = []
    for p in map(Path, paths):
        if p.is_dir():
            files += sorted(f for f in p.rglob("*") if f.suffix in (".json", ".txt", ".md"))
        else:
            files.append(p)
    return files


def main():
    parser = argparse.ArgumentParser(description="Check text for drift from the book's notation")
    parser.add_argument("paths", nargs="+", help="Files or directories (.json, .txt, .md)")
    parser.add_argument("--verbose", "-v", action="store_true", help="List every finding")
    parser.add_argument("--json", action="store_true", help="Machine-readable output")
    args = parser.parse_args()

    files = _expand(args.paths)
    results = {str(f): check_file(f) for f in files}
    by_rule = Counter(fd.rule for fds in results.values() for fd in fds)
    flagged = sum(1 for fds in results.values() if fds)

    if args.json:
        print(json.dumps({
            "files": len(files), "files_flagged": flagged, "by_rule": dict(by_rule),
            "findings": {f: [asdict(x) for x in fds] for f, fds in results.items() if fds},
        }, indent=2, ensure_ascii=False))
        return

    if args.verbose:
        for f, fds in results.items():
            for fd in fds:
                loc = f" [{fd.where}]" if fd.where else ""
                print(f"{Path(f).name}{loc}: {fd.rule}: {fd.match!r}")
        print()
    print(f"{len(files)} file(s), {flagged} with findings, {sum(by_rule.values())} finding(s)")
    width = max((len(r) for r in by_rule), default=0)
    for rule in RULES:
        if by_rule[rule.id]:
            print(f"  {rule.id:{width}s}  {by_rule[rule.id]:4d}  {rule.message}")
    sys.exit(0)


if __name__ == "__main__":
    main()
