"""
Textbook notation grounding
===========================

The LLMs in stan default to their own notation (lowercase specific properties,
"Gibbs free energy", Q - W) rather than the textbook's.  This package carries
the chapter notation primers published on cbethermo.org into prompts, and
measures drift in the output.

* `primer` - load the primers and build a grounding block for a system prompt
* `check` - flag text that departs from the book's notation (``python -m stan.notation.check``)
* `fetch` - refresh ``stan/data/grounding/`` from cbethermo.org (``python -m stan.notation.fetch``)
"""

from .primer import grounding_block, primer_version
from .check import check_text
