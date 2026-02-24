# toc_navigator.py

"""
Table of Contents Navigator for mapping page numbers to chapter/section information.

This module provides a simple interface to the ftoc_nav_tree.json file, allowing
the query processing pipeline to enrich student answers with structural context
from the textbook's table of contents.

Features:
- Maps page numbers to chapter and section information
- Filters out objectives, problems, and notation sections
- Fast lookup optimized for answer generation
- Handles edge cases gracefully (unknown pages, boundary conditions)

E. M. Furst, January 2026
"""

import json
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass


@dataclass
class TOCEntry:
    """
    Represents a chapter/section entry from the table of contents.

    Attributes:
        chapter_number: Chapter number (e.g., 1, 2, 3)
        chapter_title: Chapter title (e.g., "Introduction")
        section_number: Section number if applicable (e.g., "1.2")
        section_title: Section title if applicable
        page_start: Starting page number for this entry
        type: Entry type (chapter, section, appendix, etc.)
    """
    chapter_number: Optional[int]
    chapter_title: str
    section_number: Optional[str]
    section_title: Optional[str]
    page_start: int
    type: str


class TOCNavigator:
    """
    Navigator for the textbook table of contents.

    This class loads the TOC navigation tree and provides methods to map
    page numbers to chapter and section information. It's designed to be
    lightweight and fast for integration into the answer generation pipeline.
    """

    def __init__(self, toc_path: str):
        """
        Initialize the TOC navigator.

        Args:
            toc_path: Path to the ftoc_nav_tree.json file
        """
        self.toc_path = Path(toc_path)
        self.toc_data = self._load_toc()
        self.entries = self._build_entry_list()

    def _load_toc(self) -> Dict[str, Any]:
        """Load the TOC JSON file."""
        with open(self.toc_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _build_entry_list(self) -> List[TOCEntry]:
        """
        Build a flat list of TOC entries for fast page lookups.

        This method:
        1. Iterates through chapters and their children
        2. Filters out objectives, problems, and notation sections
        3. Creates TOCEntry objects with chapter context
        4. Sorts by page_start for efficient binary search

        Returns:
            Sorted list of TOCEntry objects
        """
        entries = []

        for item in self.toc_data.get('toc', []):
            item_type = item.get('type', '')

            # Skip non-chapter top-level items (index, etc.) for now
            # We'll handle appendices which are similar to chapters
            if item_type not in ('chapter', 'appendix'):
                continue

            chapter_number = item.get('number')
            chapter_title = item.get('title', '')
            page_start = item.get('page_start', 0)

            # Add the chapter itself as an entry
            entries.append(TOCEntry(
                chapter_number=chapter_number,
                chapter_title=chapter_title,
                section_number=None,
                section_title=None,
                page_start=page_start,
                type=item_type
            ))

            # Process children (sections, etc.)
            for child in item.get('children', []):
                child_type = child.get('type', '')

                # Filter out objectives, problems, notation
                # Only include sections and subsections
                if child_type not in ('section', 'subsection'):
                    continue

                section_number = child.get('number', '')
                section_title = child.get('title', '')
                section_page = child.get('page_start', 0)

                entries.append(TOCEntry(
                    chapter_number=chapter_number,
                    chapter_title=chapter_title,
                    section_number=section_number,
                    section_title=section_title,
                    page_start=section_page,
                    type=child_type
                ))

        # Sort by page_start for efficient lookups
        entries.sort(key=lambda x: x.page_start)

        return entries

    def get_location_for_page(self, page: int) -> Optional[TOCEntry]:
        """
        Find the TOC entry that contains a given page number.

        This method finds the entry whose page_start is <= page and whose
        next entry's page_start is > page (or it's the last entry).

        Args:
            page: Page number to lookup

        Returns:
            TOCEntry if found, None otherwise
        """
        if not self.entries:
            return None

        # Binary search would be more efficient, but linear search is fine for now
        # (typically ~100-200 entries, negligible performance impact)
        best_entry = None

        for entry in self.entries:
            if entry.page_start <= page:
                best_entry = entry
            else:
                # We've gone past the page
                break

        return best_entry

    def format_location(self, entry: Optional[TOCEntry]) -> str:
        """
        Format a TOC entry into a human-readable location string.

        Examples:
            Chapter 1 (Introduction)
            Chapter 4, Section 4.2 (Entropy Balance)
            Appendix A (Mathematical Methods)

        Args:
            entry: TOCEntry to format, or None

        Returns:
            Formatted location string, or empty string if entry is None
        """
        if not entry:
            return ""

        parts = []

        # Format chapter/appendix
        if entry.type == 'appendix':
            parts.append(f"Appendix {entry.chapter_number}")
        elif entry.chapter_number is not None:
            parts.append(f"Chapter {entry.chapter_number}")

        # Add section if available
        if entry.section_number:
            parts.append(f"Section {entry.section_number}")

        # Add title in parentheses
        if entry.section_title:
            parts.append(f"({entry.section_title})")
        elif entry.chapter_title:
            parts.append(f"({entry.chapter_title})")

        return ", ".join(parts) if len(parts) <= 2 else f"{parts[0]}, {parts[1]} {parts[2]}"

    def get_page_location_string(self, page: int) -> str:
        """
        Convenience method to get a formatted location string for a page.

        Args:
            page: Page number to lookup

        Returns:
            Formatted location string (e.g., "Chapter 4, Section 4.2 (Entropy Balance)")
        """
        entry = self.get_location_for_page(page)
        return self.format_location(entry)


def main():
    """Test the TOC navigator with example page lookups."""

    toc_path = "../data/ftoc_nav_tree.json"

    print("=" * 70)
    print("TOC Navigator Test")
    print("=" * 70)

    navigator = TOCNavigator(toc_path)

    print(f"\nLoaded {len(navigator.entries)} TOC entries")

    # Test some page lookups
    test_pages = [1, 5, 103, 260, 388, 500, 958]

    print("\nTest page lookups:")
    print("-" * 70)
    for page in test_pages:
        location = navigator.get_page_location_string(page)
        print(f"Page {page:4d}: {location}")

    # Show some raw entries
    print("\n\nSample TOC entries:")
    print("-" * 70)
    for entry in navigator.entries[:10]:
        print(f"Page {entry.page_start:4d}: {navigator.format_location(entry)}")


if __name__ == "__main__":
    main()
