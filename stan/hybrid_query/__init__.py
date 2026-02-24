"""
Hybrid search pipeline for student queries with answer generation
==================================================================

.. module:: compare_extractors
   :synopsis: Core query processing pipeline using regex and LLM extraction.

This module implements the core query processing pipeline. It is designed to 
increase retrieval accuracy by combining traditional rule-based extraction 
with large language model (LLM) reasoning.

Key Features
------------

1. **Dual-Path Extraction**: Processes student queries through both regex and LLM-based term extraction.
2. **Result Merging**: Merges results using a max-score approach for robustness.
3. **Contextual Generation**: Generates student-friendly answers enriched with textbook references.

The main entry point for this logic is the ``compare_extractors.py`` script.

Pipeline Flow
-------------

.. code-block:: text

    Student Query
         │
    ┌────┴────┐
    ▼         ▼
  REGEX     LLM
  extract   extract
    │         │
    └────┬────┘
         ▼
    merge_results()
    (max-score, page-sorted)
         │
         ▼
    generate_student_answer()
    (with TOC context)
         │
         ▼
    Student Response


Temperature Settings
--------------------

* **Term extraction (LLM):** 0.2 (deterministic)
* **Answer generation:** 0.6 (natural but factual)


Submodules
----------

The hybrid_query module contains the following submodules

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   stan.hybrid_query.extract_llm
   stan.hybrid_query.extract_regex
   stan.hybrid_query.toc_navigator
   stan.hybrid_query.compare_extractors

:Author: E. M. Furst
:Date: January 2026
"""
from . import extract_llm
from . import extract_regex
from . import compare_extractors
from . import toc_navigator