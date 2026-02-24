"""

extract_llm.py - LLM-powered query processor with semantic term extraction via Ollama.
======================================================================================

This module extends query_rewrite.QueryProcessor with LLM-based semantic understanding
for extracting search terms from student queries. It provides superior query expansion
and synonym recognition compared to the regex-based approach.

Key advantages over regex version:
----------------------------------

1. Semantic understanding: Recognizes synonyms and related concepts
2. Query expansion: Automatically suggests related terms (e.g., "entropy" →
   "entropy change", "entropy generation", "entropy balance")
3. Context awareness: Understands technical terminology in context
4. No pattern maintenance: Learns vocabulary naturally without manual patterns

Architecture:
-------------

- Uses local Ollama API for LLM inference (default model: llama3.2)
- Falls back to regex extraction if LLM unavailable (graceful degradation)
- Reuses scoring and matching logic from base QueryProcessor class
- Imports QueryProcessor and IndexMatch to avoid code duplication

Requirements:
-------------

- Ollama installed and running locally (http://localhost:11434)
- Model downloaded (e.g., ollama pull llama3.2)

E. M. Furst, January 2026
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import urllib.request
import urllib.error

from stan.hybrid_query.extract_regex import QueryProcessor, IndexMatch


class OllamaClient:
    """
    Simple HTTP client for Ollama API using stdlib urllib.

    This client provides a lightweight interface to Ollama without external dependencies.
    Uses urllib.request instead of the requests library to avoid dependency issues.
    """

    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama3.2"):
        """
        Initialize Ollama client.

        Args:
            base_url: URL of Ollama API server (default: http://localhost:11434)
            model: Model name to use for generation (default: llama3.2)

        Note on context length (num_ctx):
            Ollama defaults to num_ctx=2048 regardless of model capability.
            The query pipeline prompts (~800 tokens total) fit within this default.
            For full lecture transcripts, analyze.py sets num_ctx=16384 explicitly.
        """
        self.base_url = base_url
        self.model = model

    def generate(self, prompt: str, system: Optional[str] = None, temperature: float = 0.8) -> str:
        """
        Generate text completion using Ollama API.

        Makes a POST request to /api/generate endpoint with the specified prompt
        and optional system message. Uses non-streaming mode for simplicity.

        Args:
            prompt: The user prompt to send to the model
            system: Optional system message to set model behavior/context
            temperature: Sampling temperature (0.0-2.0). Lower = more deterministic,
                        higher = more creative. Default: 0.8

        Returns:
            Generated text response from the model

        Raises:
            RuntimeError: If Ollama API is unavailable or request fails
        """
        url = f"{self.base_url}/api/generate"

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature
            }
        }

        if system:
            payload["system"] = system

        try:
            data = json.dumps(payload).encode('utf-8')
            req = urllib.request.Request(
                url,
                data=data,
                headers={'Content-Type': 'application/json'}
            )
            with urllib.request.urlopen(req, timeout=30) as response:
                result = json.loads(response.read().decode('utf-8'))
                return result["response"]
        except urllib.error.URLError as e:
            raise RuntimeError(f"Ollama API error: {e}")
        except Exception as e:
            raise RuntimeError(f"Ollama API error: {e}")


class QueryProcessorLLM:
    """
    LLM-powered query processor with semantic term extraction.

    This class extends the functionality of QueryProcessor by using an LLM for
    intelligent term extraction instead of regex patterns. It provides:
    - Semantic understanding of queries (synonyms, related concepts)
    - Automatic query expansion (e.g., "entropy" → multiple entropy-related terms)
    - Context-aware technical term recognition
    - Graceful fallback to regex if LLM is unavailable

    The class reuses the base QueryProcessor for:
    - Hierarchical index searching
    - Match scoring
    - Result formatting
    - Fallback term extraction
    """

    def __init__(
        self,
        index_path: str,
        ollama_url: str = "http://localhost:11434",
        model: str = "llama3.2",
        use_fallback: bool = True
    ):
        """
        Initialize the LLM-powered query processor.

        Args:
            index_path: Path to the bindex_tab.json hierarchical index file
            ollama_url: Base URL for Ollama API (default: http://localhost:11434)
            model: Ollama model name to use (default: llama3.2)
            use_fallback: If True, falls back to regex extraction when LLM fails
                         (recommended for production to ensure robustness)
        """
        self.index_path = Path(index_path)
        self.index_data = self._load_index()
        self.ollama = OllamaClient(base_url=ollama_url, model=model)
        self.use_fallback = use_fallback

        # Initialize regex processor for fallback (only if fallback enabled)
        # This reuses QueryProcessor to avoid code duplication
        if use_fallback:
            self._regex_processor = QueryProcessor(index_path)

    def _load_index(self) -> List[Dict[str, Any]]:
        """Load the index JSON file."""
        with open(self.index_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def extract_key_terms_llm(self, query: str) -> List[str]:
        """
        Extract key terms using LLM with semantic understanding.

        Args:
            query: Natural language query from student

        Returns:
            List of extracted key terms and relevant synonyms
        """
        system_message = """You are an expert in thermodynamics helping students find topics in a textbook.
Your task is to extract search terms from student queries that can be used to search a textbook index.

Instructions:
1. Identify the main technical concepts in the query
2. Extract both single terms and multi-word phrases
3. Include relevant synonyms and related terms
4. Preserve exact technical terminology (e.g., "van der Waals", "van Laar")
5. Include common abbreviations if relevant (e.g., "EOS" for "equation of state")
6. Return ONLY the search terms as a comma-separated list, nothing else

Examples:
Query: "Where can I read about the van der Waals equation of state?"
Output: van der Waals equation of state, van der Waals, equation of state, EOS

Query: "What is entropy?"
Output: entropy, entropy change, entropy generation, entropy balance

Query: "I need help with energy balance"
Output: energy balance, energy balance equation, first law of thermodynamics, conservation of energy"""

        prompt = f"""Extract search terms from this student query:

Query: "{query}"

Search terms:"""

        try:
            # Use low temperature for consistent, deterministic term extraction
            response = self.ollama.generate(prompt, system=system_message, temperature=0.2)

            # Parse the response to extract terms
            # Remove any explanatory text and get just the terms
            terms_text = response.strip()

            # Handle cases where LLM might add extra text
            if '\n' in terms_text:
                terms_text = terms_text.split('\n')[0]

            # Split by commas and clean up
            terms = [term.strip().lower() for term in terms_text.split(',')]
            terms = [term for term in terms if term and len(term) > 1]

            return list(set(terms))  # Remove duplicates

        except Exception as e:
            print(f"LLM extraction failed: {e}")
            if self.use_fallback:
                print("Falling back to regex extraction...")
                return self._regex_processor.extract_key_terms(query)
            else:
                raise

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

        # Calculate match score (now includes page-based scoring)
        score = self._calculate_match_score(topic, pages, search_terms)

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

    def _calculate_match_score(self, topic: str, pages: str, search_terms: List[str]) -> float:
        """
        Calculate a match score between a topic and search terms.

        Scoring includes both term matching and page position. Earlier pages are
        preferred since fundamental concepts are typically introduced first.

        Args:
            topic: Index topic text
            pages: Page numbers (e.g., "123", "123-125", "123, 456")
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

        # Add page-based bonus: earlier pages get higher scores
        # This prioritizes fundamental concepts introduced early in the textbook
        if pages and score > 0:
            try:
                # Extract first page number from formats like "123", "123-125", "123, 456"
                first_page_str = pages.split(',')[0].split('-')[0].strip()
                first_page = int(first_page_str)

                # Calculate bonus: pages 1-200 get significant boost, diminishing after
                # Formula: max bonus at page 1, decreasing linearly to 0 at page 500
                page_bonus = max(0, (500 - first_page) / 50)
                score += page_bonus
            except (ValueError, AttributeError):
                pass  # If page parsing fails, just use the base score

        return score

    def search_index(self, query: str, max_results: int = 10) -> tuple[List[IndexMatch], List[str]]:
        """
        Search the index for matches to the query.

        Args:
            query: Natural language query from student
            max_results: Maximum number of results to return

        Returns:
            Tuple of (list of IndexMatch objects sorted by relevance, extracted terms)
        """
        # Extract key terms from query using LLM
        search_terms = self.extract_key_terms_llm(query)

        # Search all index entries
        all_matches = []
        for entry in self.index_data:
            matches = self._search_topic_recursive(entry, search_terms)
            all_matches.extend(matches)

        # Sort by score (descending)
        all_matches.sort(key=lambda x: x.score, reverse=True)

        # Return top results along with the search terms used
        return all_matches[:max_results], search_terms

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
    """Example usage comparing LLM vs regex extraction."""

    # Path to index file
    index_path = "../data/bindex_tab.json"

    print("=" * 70)
    print("LLM-Powered Query Processor")
    print("=" * 70)
    print("\nChecking Ollama connection...")

    try:
        # Initialize LLM processor
        processor_llm = QueryProcessorLLM(index_path, use_fallback=True)
        print("✓ Ollama is available\n")

        # Example queries
        example_queries = [
            "Where can I read about the van der Waals equation of state?",
            "I don't understand the van Laar equation.",
            "What is entropy?",
            "I need help with the energy balance equations.",
            "Tell me about heat capacity",
            "What's the difference between enthalpy and entropy?"
        ]

        for query in example_queries:
            print(f"\nQuery: {query}")
            print("-" * 70)

            # Search using LLM
            matches, terms = processor_llm.search_index(query, max_results=5)
            print(f"LLM extracted terms: {terms}")
            print()

            # Display results
            print(processor_llm.format_results(matches))

    except RuntimeError as e:
        print(f"✗ {e}")
        print("\nTo use this script, make sure Ollama is running:")
        print("  1. Install Ollama: https://ollama.ai/")
        print("  2. Run: ollama pull llama3.2")
        print("  3. Start Ollama service")


if __name__ == "__main__":
    main()
