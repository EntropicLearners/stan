"""
compare_extractors.py

Usage Notes
-----------

When run directly, this script provides diagnostic output comparing the two
extraction methods

Primary Functions
-----------------

.. autofunction:: stan.hybrid_query.compare_extractors.merge_results
.. autofunction:: stan.hybrid_query.compare_extractors.generate_student_answer
"""

from .extract_regex import QueryProcessor
from .extract_regex import IndexMatch
from .extract_llm import QueryProcessorLLM, OllamaClient
from .toc_navigator import TOCNavigator


def merge_results(matches_regex, matches_llm, score_threshold=10.0):
    """
    Intelligently merge and deduplicate results from regex and LLM searches.

    This function implements a hybrid approach that combines the strengths of both
    methods while avoiding duplicates. The strategy ensures that results found by 
    either method are preserved, using the maximum score from either method as the 
    representative value. Results are primarily ranked by page number to prioritize 
    fundamental concepts, with the score used as a secondary tie-breaker.

    The merging strategy follows these steps:
    1. Identify unique results using a (topic, pages) key.
    2. Assign the maximum score between the two methods to each unique result.
    3. Filter results below the `score_threshold`.
    4. Exclude matches lacking page numbers.
    5. Sort results by page number (primary) and score (secondary).

    Parameters
    ----------
    matches_regex : list of IndexMatch
        List of IndexMatch objects retrieved from the regex-based search.
    matches_llm : list of IndexMatch
        List of IndexMatch objects retrieved from the LLM-based search.
    score_threshold : float, optional
        The minimum score required for a match to be included in the final results. 
        Defaults to 10.0.

    Returns
    -------
    list of IndexMatch
        A merged and deduplicated list of IndexMatch objects, filtered by 
        `score_threshold` and sorted primarily by page number.

    Notes
    -----
    This approach is robust to LLM variability. By using a "max-score" logic rather 
    than a weighted average, a high-confidence match from one method will not be 
    "diluted" or filtered out if the other method fails to identify it. For example:

    Regex finds: "fugacity coefficient" (score: 21.0, page 315)
    LLM misses it on first run: (score: 0.0)

    Old weighted approach: 0.7×0 + 0.3×21 = 6.3 ❌ (filtered out!)
    New max approach: max(0, 21) = 21.0 ✓ (kept!)

    Examples
    --------
    >>> regex_results = [IndexMatch(topic="fugacity", score=21.0, page=315)]
    >>> llm_results = []
    >>> merge_results(regex_results, llm_results)
    [IndexMatch(topic="fugacity", score=21.0, page=315)]
    """
    # Dictionary to track best match for each unique topic+pages combination
    # Key: (topic, pages) tuple uniquely identifies a textbook section
    # Value: dict with regex_score, llm_score, source, and match object
    matches_dict = {}

    # PHASE 1: Process regex matches
    # Add all regex results to the dictionary
    for match in matches_regex:
        key = (match.topic, match.pages)
        if key not in matches_dict:
            matches_dict[key] = {
                'match': match,
                'regex_score': match.score,
                'llm_score': 0,  # Will be updated if LLM also finds this
                'source': 'regex'
            }

    # PHASE 2: Process LLM matches
    # Merge with regex results or add new ones
    for match in matches_llm:
        key = (match.topic, match.pages)
        if key in matches_dict:
            # Already found by regex - this is a confirmed match!
            # Update with LLM score and mark as found by both methods
            matches_dict[key]['llm_score'] = match.score
            matches_dict[key]['source'] = 'both'
            # Keep the match object with the higher individual score
            if match.score > matches_dict[key]['match'].score:
                matches_dict[key]['match'] = match
        else:
            # New match found only by LLM (semantic understanding caught it)
            matches_dict[key] = {
                'match': match,
                'regex_score': 0,  # Not found by regex
                'llm_score': match.score,
                'source': 'llm'
            }

    # PHASE 3: Calculate best scores and create final list
    merged_matches = []
    for key, data in matches_dict.items():
        match = data['match']

        # Take the MAXIMUM score from either method
        # This ensures results aren't penalized if one method misses them
        # If both found it, the higher score represents better matching
        best_score = max(data['llm_score'], data['regex_score'])

        # Create new IndexMatch with the best score
        merged_match = IndexMatch(
            topic=match.topic,
            pages=match.pages,
            score=best_score,  # Use the best score from either method
            path=match.path
        )
        merged_matches.append(merged_match)

    # PHASE 4: Filter by score threshold to exclude outliers
    filtered_matches = [m for m in merged_matches if m.score >= score_threshold]

    # PHASE 5: Exclude matches without page numbers
    # Only keep matches that have valid page information
    filtered_matches = [m for m in filtered_matches if m.pages and m.pages.strip()]

    # PHASE 6: Sort by page number (PRIMARY) then score (SECONDARY)
    # Earlier pages = more fundamental concepts
    # Higher scores = better relevance (used to break ties)
    def get_sort_key(match):
        try:
            # Handle formats like "123", "123-125", "123, 456"
            pages_str = match.pages.split(',')[0].split('-')[0].strip()
            page_num = int(pages_str)
            # Return tuple: (page ascending, score descending for tie-breaking)
            return (page_num, -match.score)
        except (ValueError, AttributeError):
            # If page parsing fails, sort to end with low score
            return (float('inf'), -match.score)

    filtered_matches.sort(key=get_sort_key)

    return filtered_matches


def generate_student_answer(query: str, matches, extracted_terms, ollama: OllamaClient,
                          toc_navigator: TOCNavigator = None) -> tuple[str, str]:
    """
    Generate a friendly, helpful answer to guide students to textbook sections.

    This function uses the LLM to create a natural language response that:
    - Directly addresses the student's question
    - Points to the most relevant textbook sections
    - Includes specific page numbers
    - Enriches answers with chapter/section context when available
    - Maintains an encouraging, supportive tone

    The LLM is given the top search results as context and instructed to act as
    a helpful teaching assistant. If the LLM fails, a simple fallback response
    is generated using the top match.

    Parameters
    ----------
    query : str
        The student's original natural language query
    matches : list of IndexMatch
        List of IndexMatch objects from merged search (best results)
    extracted_terms : list of str
        List of terms extracted from query (for context)
    ollama : OllamaClient
        OllamaClient instance for LLM text generation
    toc_navigator : TOCNavigator, optional
        Optional TOCNavigator for chapter/section context

    Returns
    -------
    answer : str
        Student-facing answer.
    context : str
        Context sent to LLM.

    Examples
    --------
    Query: "What is entropy?"
    Output: "Entropy is a measure of disorder... Check out 'Entropy Generation'
            on pages 958-962 in Chapter 15, Section 15.7 (Thermodynamic Analysis
            of Fermenters and Other Bioreactors) in your textbook."
    """
    if not matches:
        return "I couldn't find any relevant topics in the textbook index for your query.", ""

    # Build context from top matches, enriched with TOC information when available
    context = "Top relevant topics from the textbook index:\n"
    for i, match in enumerate(matches[:5], 1):
        path = " > ".join(match.path) if len(match.path) > 1 else match.topic
        context += f"{i}. {path}"

        if match.pages:
            context += f" (pages {match.pages})"

            # Add chapter/section context if TOC navigator is available
            if toc_navigator:
                try:
                    # Extract first page number from the match
                    first_page_str = match.pages.split(',')[0].split('-')[0].strip()
                    first_page = int(first_page_str)
                    location = toc_navigator.get_page_location_string(first_page)
                    if location:
                        context += f"\n   Location: {location}"
                except (ValueError, AttributeError):
                    pass  # If page parsing fails, just skip location info

        context += "\n"

    system_message = """You are a helpful teaching assistant for a thermodynamics course.
Your role is to guide students to the right sections of their textbook.

Given a student's question and relevant topics from the textbook index, provide a brief,
friendly answer that:
1. Directly addresses their question with a concise explanation
2. Lists ALL the relevant textbook sections provided (don't skip any)
3. Include SPECIFIC page numbers for each section mentioned
4. Include chapter/section location when provided (e.g., "Chapter 4, Section 4.2")
5. Use the exact topic names provided in the context
6. Is encouraging and supportive

CRITICAL RULES:
- ONLY use information explicitly provided in the context below
- DO NOT make up chapter numbers, section numbers, or other structural information
- DO NOT infer or assume any information not in the context
- ONLY reference page numbers and locations that are explicitly given
- Use the topic names and hierarchies exactly as provided
- If a "Location:" is provided for a topic, include it naturally in your answer

Format references clearly:
- "pages 102-105 in Chapter 4, Section 4.1 (Entropy: A New Concept)"
- "page 958 in Chapter 15, Section 15.7"

Keep your response concise but comprehensive (3-5 sentences)."""

    prompt = f"""Student Question: "{query}"

{context}

Provide a helpful response to guide the student to the right textbook sections.

After your response, include a "References:" section that lists all the topics and page numbers
from the context above in a clean, easy-to-read format."""

    try:
        # Use low temperature to stay grounded in provided context and reduce hallucination
        response = ollama.generate(prompt, system=system_message, temperature=0.4)
        return response.strip(), context
    except Exception as e:
        # Fallback to simple formatted response
        answer = f"I found information about your question in the textbook. "
        answer += f"Check out '{matches[0].path[-1]}' on pages {matches[0].pages}."
        return answer, context


def compare_extractors():
    """
    Diagnostic function: runs test queries and displays detailed extraction analysis.

    This function is primarily for testing and debugging the hybrid search pipeline.
    It runs a suite of sample queries through both methods and displays:
    - Regex extraction results (terms and matches)
    - LLM extraction results (terms and matches)
    - Differences in extracted terms between methods
    - Merged results using max-score approach
    - Generated student-facing answers

    For production use, import merge_results() and generate_student_answer() directly.
    See apps/query_student.py for an example of production usage.
    """

    index_path = "../data/bindex_tab.json"
    toc_path = "../data/ftoc_nav_tree.json"

    # Initialize both processors
    print("Initializing processors...")
    processor_regex = QueryProcessor(index_path)

    # Initialize TOC navigator for chapter/section context
    try:
        toc_navigator = TOCNavigator(toc_path)
        print(f"✓ TOC Navigator loaded ({len(toc_navigator.entries)} entries)")
    except Exception as e:
        print(f"Warning: TOC Navigator not available: {e}")
        toc_navigator = None

    # Try to initialize LLM processor (graceful degradation if unavailable)
    # Both term extraction and answer generation use the same model to avoid Ollama model swapping
    try:
        processor_llm = QueryProcessorLLM(index_path, use_fallback=True)
        ollama = OllamaClient()  # Uses same default model as QueryProcessorLLM
        llm_available = True
    except RuntimeError as e:
        print(f"LLM not available: {e}")
        llm_available = False
        ollama = None

    # Test queries
    test_queries = [
        "Where can I read about the van der Waals equation of state?",
        "What is entropy?",
        "Does the book cover activity coefficient models?",
        "I need help with energy balance equations",
        "Tell me about heat capacity",
        "What's a Carnot cycle?",
        "Explain fugacity",
        "I don't understand the difference between enthalpy and entropy",
        "What is the ideal gas law?",
        "What is the difference between constant volume and constant pressure heat capacities?",
        "What is an equation of state?",
        "Where is the Peng-Robinson equation discussed?",
        "When do I use the difference form of a balance equation?",
    ]

    print("\n" + "=" * 90)
    print("COMPARISON: Regex vs LLM Term Extraction")
    print("=" * 90)

    for i, query in enumerate(test_queries, 1):
        print(f"\n{'=' * 90}")
        print(f"Query {i}: {query}")
        print("=" * 90)

        # Regex extraction
        print("\n[REGEX METHOD]")
        terms_regex = processor_regex.extract_key_terms(query)
        print(f"Extracted terms: {sorted(terms_regex)}")

        matches_regex = processor_regex.search_index(query, max_results=5)
        print(f"Top {len(matches_regex)} matches:")
        for j, match in enumerate(matches_regex, 1):
            path = " > ".join(match.path) if len(match.path) > 1 else match.topic
            print(f"  {j}. {path} (score: {match.score:.1f})")
            if match.pages:
                print(f"     Pages: {match.pages}")

        # LLM extraction
        if llm_available:
            print("\n[LLM METHOD]")
            try:
                matches_llm, terms_llm = processor_llm.search_index(query, max_results=5)
                print(f"Extracted terms: {sorted(terms_llm)}")
                print(f"Top {len(matches_llm)} matches:")
                for j, match in enumerate(matches_llm, 1):
                    path = " > ".join(match.path) if len(match.path) > 1 else match.topic
                    print(f"  {j}. {path} (score: {match.score:.1f})")
                    if match.pages:
                        print(f"     Pages: {match.pages}")

                # Highlight differences
                terms_only_regex = set(terms_regex) - set(terms_llm)
                terms_only_llm = set(terms_llm) - set(terms_regex)

                if terms_only_regex or terms_only_llm:
                    print("\n[DIFFERENCES]")
                    if terms_only_regex:
                        print(f"  Only in regex: {sorted(terms_only_regex)}")
                    if terms_only_llm:
                        print(f"  Only in LLM: {sorted(terms_only_llm)}")

                # Merge results from both methods
                print("\n[MERGED RESULTS]")
                score_threshold = 10.0
                merged_matches = merge_results(matches_regex, matches_llm, score_threshold=score_threshold)
                print(f"Total matches after filtering: {len(merged_matches)}")
                print(f"\nAll {len(merged_matches)} merged results (filtered by score >= {score_threshold}, sorted by page):")
                for j, match in enumerate(merged_matches, 1):
                    path = " > ".join(match.path) if len(match.path) > 1 else match.topic
                    print(f"  {j}. {path} (score: {match.score:.1f})")
                    if match.pages:
                        print(f"     Pages: {match.pages}")

                # Generate student-facing answer using merged results
                print("\n[GENERATING STUDENT ANSWER]")
                student_answer, llm_context = generate_student_answer(
                    query, merged_matches, terms_llm, ollama, toc_navigator
                )

                # Show what context was sent to the LLM
                print("\n[CONTEXT SENT TO LLM]")
                print(f"Number of results sent: {min(5, len(merged_matches))}")
                print("\nFull context string:")
                print("-" * 70)
                print(llm_context)
                print("-" * 70)

                print("\n[STUDENT ANSWER]")
                print(student_answer)

            except Exception as e:
                print(f"LLM processing failed: {e}")
                print("(Using regex fallback)")

if __name__ == "__main__":
    compare_extractors()
