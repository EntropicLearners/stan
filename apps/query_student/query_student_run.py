#!/usr/bin/env python3
# query_student.py

"""
Student query interface for thermodynamics textbook index search.

This script provides a command-line interface for students to ask questions about
thermodynamics topics and receive guidance to relevant textbook sections.

Features:
- Accepts natural language queries via command line
- Uses hybrid regex + LLM approach for optimal results
- Displays detailed trace of search process
- Generates student-friendly answers with page references

Usage:
    python query_student.py "What is entropy?"
    python query_student.py "Where can I read about van der Waals equation?"

E. M. Furst, January 2026
"""

import sys
import argparse
from pathlib import Path

# Add the stan/run directory to the path so we can import modules
run_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(run_dir))

from stan.hybrid_query.extract_regex import QueryProcessor
from stan.hybrid_query.extract_llm import QueryProcessorLLM, OllamaClient
from stan.hybrid_query.compare_extractors import merge_results, generate_student_answer
from stan.hybrid_query.toc_navigator import TOCNavigator


def process_query(query: str, model: str = None, show_trace: bool = True):
    """
    Process a student query and generate an answer.

    Args:
        query: Natural language question from student
        model: Ollama model name for answer generation (default: OllamaClient default)
        show_trace: If True, display detailed trace of search process

    Returns:
        Tuple of (student_answer, merged_matches)
    """
    # Path to index file relative to the apps directory
    stan_data_root = Path(__file__).parent.parent.parent / "stan" / "data"
    index_path = (stan_data_root / "bindex_tab.json").as_posix()
    toc_path = (stan_data_root / "ftoc_nav_tree.json").as_posix()

    # Initialize processors
    processor_regex = QueryProcessor(index_path)

    # Initialize TOC navigator for chapter/section context
    try:
        toc_navigator = TOCNavigator(toc_path)
        if show_trace:
            print(f"✓ TOC Navigator loaded ({len(toc_navigator.entries)} entries)")
    except Exception as e:
        if show_trace:
            print(f"Warning: TOC Navigator not available ({e})")
        toc_navigator = None

    try:
        if model:
            processor_llm = QueryProcessorLLM(index_path, model=model, use_fallback=True)
            ollama = OllamaClient(model=model)
        else:
            processor_llm = QueryProcessorLLM(index_path, use_fallback=True)
            ollama = OllamaClient()
        llm_available = True
        if show_trace:
            print(f"✓ Using model: {model or 'llama3.2 (default)'}")
    except RuntimeError as e:
        if show_trace:
            print(f"Warning: LLM not available ({e}). Using regex only.")
        llm_available = False
        processor_llm = None
        ollama = None

    if show_trace:
        print("\n" + "=" * 90)
        print(f"Query: {query}")
        print("=" * 90)

    # Regex extraction
    if show_trace:
        print("\n[REGEX METHOD]")
    terms_regex = processor_regex.extract_key_terms(query)
    if show_trace:
        print(f"Extracted terms: {sorted(terms_regex)}")

    matches_regex = processor_regex.search_index(query, max_results=5)
    if show_trace:
        print(f"Top {len(matches_regex)} matches:")
        for j, match in enumerate(matches_regex, 1):
            path = " > ".join(match.path) if len(match.path) > 1 else match.topic
            print(f"  {j}. {path} (score: {match.score:.1f})")
            if match.pages:
                print(f"     Pages: {match.pages}")

    # LLM extraction
    if llm_available:
        if show_trace:
            print("\n[LLM METHOD]")
        try:
            matches_llm, terms_llm = processor_llm.search_index(query, max_results=5)
            if show_trace:
                print(f"Extracted terms: {sorted(terms_llm)}")
                print(f"Top {len(matches_llm)} matches:")
                for j, match in enumerate(matches_llm, 1):
                    path = " > ".join(match.path) if len(match.path) > 1 else match.topic
                    print(f"  {j}. {path} (score: {match.score:.1f})")
                    if match.pages:
                        print(f"     Pages: {match.pages}")

            # Highlight differences
            if show_trace:
                terms_only_regex = set(terms_regex) - set(terms_llm)
                terms_only_llm = set(terms_llm) - set(terms_regex)

                if terms_only_regex or terms_only_llm:
                    print("\n[DIFFERENCES]")
                    if terms_only_regex:
                        print(f"  Only in regex: {sorted(terms_only_regex)}")
                    if terms_only_llm:
                        print(f"  Only in LLM: {sorted(terms_only_llm)}")

            # Merge results from both methods
            if show_trace:
                print("\n[MERGED RESULTS]")
            score_threshold = 20.0
            merged_matches = merge_results(matches_regex, matches_llm, score_threshold=score_threshold)

            if show_trace:
                print(f"Total matches after filtering: {len(merged_matches)}")
                print(f"\nAll {len(merged_matches)} merged results (filtered by score >= {score_threshold}, sorted by page):")
                for j, match in enumerate(merged_matches, 1):
                    path = " > ".join(match.path) if len(match.path) > 1 else match.topic
                    print(f"  {j}. {path} (score: {match.score:.1f})")
                    if match.pages:
                        print(f"     Pages: {match.pages}")

            # Generate student-facing answer using merged results
            if show_trace:
                print("\n[GENERATING STUDENT ANSWER]")
            student_answer, llm_context = generate_student_answer(
                query, merged_matches, terms_llm, ollama, toc_navigator
            )

            # Show what context was sent to the LLM
            if show_trace:
                print("\n[CONTEXT SENT TO LLM]")
                print(f"Number of results sent: {min(5, len(merged_matches))}")
                print("\nFull context string:")
                print("-" * 70)
                print(llm_context)
                print("-" * 70)

            if show_trace:
                print("\n[STUDENT ANSWER]")
            print(student_answer)

            return student_answer, merged_matches

        except Exception as e:
            if show_trace:
                print(f"\nLLM processing failed: {e}")
                print("(Using regex fallback)")
            llm_available = False

    # Fallback to regex-only if LLM not available
    if not llm_available:
        if show_trace:
            print("\n[USING REGEX-ONLY RESULTS]")

        # Filter and sort regex results by page
        filtered_matches = [m for m in matches_regex if m.pages and m.pages.strip()]

        def get_start_page(match):
            try:
                pages_str = match.pages.split(',')[0].split('-')[0].strip()
                return int(pages_str)
            except (ValueError, AttributeError):
                return float('inf')

        filtered_matches.sort(key=get_start_page)

        # Generate simple answer
        if filtered_matches:
            answer = f"I found information about your question in the textbook.\n\n"
            answer += "Relevant sections:\n"
            for i, match in enumerate(filtered_matches[:5], 1):
                path = " > ".join(match.path) if len(match.path) > 1 else match.topic
                answer += f"{i}. {path}"
                if match.pages:
                    answer += f" (pages {match.pages})"
                answer += "\n"
            print(answer)
            return answer, filtered_matches
        else:
            answer = "I couldn't find any relevant topics in the textbook index for your query."
            print(answer)
            return answer, []


def main():
    """Main entry point for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Student query interface for thermodynamics textbook index search."
    )
    parser.add_argument("query", nargs="+", help="Natural language question")
    parser.add_argument("--model", "-m", default=None,
                        help="Ollama model for answer generation (default: llama3.2)")
    args = parser.parse_args()

    query = " ".join(args.query)
    process_query(query, model=args.model, show_trace=True)


if __name__ == "__main__":
    main()
