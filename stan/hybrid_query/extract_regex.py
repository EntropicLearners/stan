"""
query_rewrite.py - Regex-based query processor for matching student queries to textbook index topics.
=====================================================================================================

This is the base implementation that uses regular expressions and pattern matching
to extract key terms from natural language queries and match them against a
hierarchical textbook index.

Typical student queries:
------------------------

- "Where can I read about the van der Waals equation of state?"
- "I don't understand the van Laar equation."
- "What is entropy?"
- "I need help with the energy balance equations."

Architecture:
-------------

1. Extract key terms from query (remove stopwords, identify multi-word phrases)
2. Search hierarchical index recursively
3. Score matches based on term overlap and position
4. Return ranked results

This module serves as the base for query_rewrite_llm.py, which extends it
with LLM-powered semantic understanding.

Data source: bindex_tab.json (hierarchical thermodynamics textbook index)

E. M. Furst, January 2026
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass


@dataclass
class IndexMatch:
    """Represents a match between a query and an index entry."""
    topic: str
    pages: str
    score: float
    path: List[str]  # Hierarchy path for subtopics

    def __repr__(self):
        path_str = " > ".join(self.path) if len(self.path) > 1 else self.path[0]
        return f"IndexMatch(topic='{path_str}', pages='{self.pages}', score={self.score:.2f})"


class QueryProcessor:
    """
    Regex-based processor for matching student queries to textbook index topics.

    This class provides the core functionality for:
    - Extracting key terms from natural language queries using regex patterns
    - Searching a hierarchical index structure recursively
    - Scoring matches based on term overlap and relevance
    - Formatting results for display to students

    This serves as the base implementation that query_rewrite_llm.QueryProcessorLLM
    extends with LLM-powered semantic understanding.
    """

    def __init__(self, index_path: str):
        """
        Initialize the query processor with a textbook index.

        Args:
            index_path: Path to the bindex_tab.json file containing the hierarchical
                       thermodynamics textbook index
        """
        self.index_path = Path(index_path)
        self.index_data = self._load_index()

    def _load_index(self) -> List[Dict[str, Any]]:
        """Load the index JSON file."""
        with open(self.index_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def extract_key_terms(self, query: str) -> List[str]:
        """
        Extract key technical terms from a natural language query using regex.

        Process:
        1. Remove common stopwords (question words, articles, prepositions)
        2. Identify multi-word technical phrases using predefined patterns
        3. Include both multi-word phrases and individual remaining words

        Example:
            Input: "Where can I read about the van der Waals equation of state?"
            Output: ['van der waals', 'equation of state', 'van', 'der', 'waals', 'equation', 'state']

        Args:
            query: Natural language query from student

        Returns:
            List of extracted key terms (both multi-word phrases and individual words)

        Note:
            This regex approach is fast but limited to predefined patterns.
            For semantic understanding, use QueryProcessorLLM instead.
        """
        # Stopwords: Common words that don't help identify technical topics
        stopwords = [
            'where', 'can', 'i', 'read', 'about', 'what', 'is', 'are',
            'how', 'do', 'does', 'the', 'a', 'an', 'in', 'on', 'at',
            'to', 'for', 'of', 'with', "don't", 'understand', 'need',
            'help', 'explain', 'tell', 'me', 'show', 'and', 'between',
            'discussed', 'find',
        ]

        # Normalize: lowercase and remove punctuation
        query_lower = query.lower()
        query_lower = re.sub(r'[?!.,;]', ' ', query_lower)
        words = query_lower.split()

        # Filter out stopwords to get candidate terms
        key_terms = [word for word in words if word not in stopwords]
        text = ' '.join(key_terms)

        terms = []

        # Multi-word patterns: Common thermodynamics phrases to recognize as units
        # These are searched first to preserve important technical phrases
        multi_word_patterns = [
            r'van der waals',       # Named equation
            r'van laar',            # Named equation
            r'energy balance',      # Fundamental concept
            r'mass balance',        # Fundamental concept
            r'equation of state',   # Fundamental concept
            r'ideal gas',           # Common phrase
            r'first law',           # Thermodynamics law
            r'second law',          # Thermodynamics law
            r'entropy generation',  # Specific entropy concept
            r'entropy change',      # Specific entropy concept
            r'phase equilibrium',   # Equilibrium type
            r'chemical potential',  # Thermodynamic property
            r'gibbs energy',        # Energy type
            r'helmholtz energy',    # Energy type
            r'internal energy',     # Energy type
        ]

        # Extract recognized multi-word phrases
        for pattern in multi_word_patterns:
            if re.search(pattern, text):
                terms.append(pattern)

        # Add all individual terms (creates redundancy but improves recall)
        terms.extend(key_terms)

        return list(set(terms))  # Remove duplicates

    def _search_topic_recursive(
        self,
        entry: Dict[str, Any],
        search_terms: List[str],
        parent_path: List[str] = None
    ) -> List[IndexMatch]:
        """
        Recursively search through topics and subtopics.

        Args:
            entry: Index entry dictionary
            search_terms: List of terms to search for
            parent_path: Path of parent topics (for hierarchy)

        Returns:
            List of IndexMatch objects
        """
        if parent_path is None:
            parent_path = []

        matches = []
        topic = entry.get('topic', '')
        pages = entry.get('pages', '')

        current_path = parent_path + [topic]

        # Calculate match score
        score = self._calculate_match_score(topic, search_terms)

        if score > 0:
            matches.append(IndexMatch(
                topic=topic,
                pages=pages,
                score=score,
                path=current_path
            ))

        # Recursively search subtopics
        if 'subtopics' in entry:
            for subtopic in entry['subtopics']:
                matches.extend(
                    self._search_topic_recursive(subtopic, search_terms, current_path)
                )

        return matches

    def _calculate_match_score(self, topic: str, search_terms: List[str]) -> float:
        """
        Calculate a match score between a topic and search terms.

        Args:
            topic: Index topic text
            search_terms: List of search terms

        Returns:
            Match score (0 = no match, higher = better match)
        """
        topic_lower = topic.lower()
        score = 0.0

        for term in search_terms:
            term_lower = term.lower()

            # Exact phrase match (highest score)
            if term_lower in topic_lower:
                # Weight by term length (longer terms are more specific)
                score += len(term_lower) * 2

                # Bonus if it's at the start of the topic
                if topic_lower.startswith(term_lower):
                    score += 5

            # Partial word matches
            else:
                term_words = term_lower.split()
                for word in term_words:
                    if len(word) > 2 and word in topic_lower:
                        score += len(word) * 0.5

        return score

    def search_index(self, query: str, max_results: int = 10) -> List[IndexMatch]:
        """
        Search the index for matches to the query.

        Args:
            query: Natural language query from student
            max_results: Maximum number of results to return

        Returns:
            List of IndexMatch objects, sorted by relevance
        """
        # Extract key terms from query
        search_terms = self.extract_key_terms(query)

        # Search all index entries
        all_matches = []
        for entry in self.index_data:
            matches = self._search_topic_recursive(entry, search_terms)
            all_matches.extend(matches)

        # Sort by score (descending)
        all_matches.sort(key=lambda x: x.score, reverse=True)

        # Return top results
        return all_matches[:max_results]

    def format_results(self, matches: List[IndexMatch]) -> str:
        """
        Format search results for display to student.

        Args:
            matches: List of IndexMatch objects

        Returns:
            Formatted string for display
        """
        if not matches:
            return "No matches found in the index."

        result = f"Found {len(matches)} relevant topics:\n\n"

        for i, match in enumerate(matches, 1):
            # Build hierarchy string
            if len(match.path) > 1:
                path_str = " > ".join(match.path)
            else:
                path_str = match.topic

            result += f"{i}. {path_str}\n"
            if match.pages:
                result += f"   Pages: {match.pages}\n"
            result += "\n"

        return result


def main():
    """Example usage of the QueryProcessor."""

    # Path to index file (adjust as needed)
    index_path = "../data/bindex_tab.json"

    # Initialize processor
    processor = QueryProcessor(index_path)

    # Example queries
    example_queries = [
        "Where can I read about the van der Waals equation of state?",
        "I don't understand the van Laar equation.",
        "What is entropy?",
        "I need help with the energy balance equations.",
        "What is the van Laar equation?"
    ]

    print("=" * 70)
    print("Student Query Processor - Examples")
    print("=" * 70)

    for query in example_queries:
        print(f"\nQuery: {query}")
        print("-" * 70)

        # Extract terms
        terms = processor.extract_key_terms(query)
        print(f"Extracted terms: {terms}")
        print()

        # Search index
        matches = processor.search_index(query, max_results=5)

        # Display results
        print(processor.format_results(matches))


if __name__ == "__main__":
    main()
