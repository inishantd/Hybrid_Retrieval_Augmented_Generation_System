import logging
import re
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


class CitationVerifier:
    """
    Verifies bracketed citations in LLM responses against
    the retrieved context chunks.

    Example:
        LLM answer:
        "Employees receive 20 days of annual leave [1]."

        [1] must correspond to the first retrieved chunk.
    """

    CITATION_PATTERN = re.compile(r"\[(\d+)\]")

    @classmethod
    def verify_citations(
        cls,
        llm_answer: str,
        top_chunks: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Validate citation references such as [1], [2], [3].

        Checks:
        1. Citation format is valid.
        2. Citation index exists in top_chunks.
        3. Referenced chunk contains non-empty text.
        """

        if not llm_answer:
            return {
                "is_valid": False,
                "flagged_issues": {
                    "answer": "EMPTY_ANSWER: LLM answer is empty."
                },
                "validated_indices": [],
            }

        if not top_chunks:
            return {
                "is_valid": False,
                "flagged_issues": {
                    "context": "EMPTY_CONTEXT: No retrieved chunks were provided."
                },
                "validated_indices": [],
            }

        # Extract citations such as [1], [2], [15]
        matches = cls.CITATION_PATTERN.findall(llm_answer)

        extracted_indices = sorted(
            set(int(match) for match in matches)
        )
        if not extracted_indices:
            return {
                "is_valid": False,
                "flagged_issues": {
                    "answer": "NO_CITATIONS: Answer does not cite any context block."
                },
                "validated_indices": [],
            }

        flagged_citations: Dict[str, str] = {}
        validated_indices: List[int] = []

        for index in extracted_indices:
            array_slot = index - 1

            # Citation points outside the available context
            if array_slot < 0 or array_slot >= len(top_chunks):
                flagged_citations[f"[{index}]"] = (
                    "MALFORMED_INDEX: "
                    "Context block index does not exist."
                )
                continue

            try:
                chunk = top_chunks[array_slot]["chunk"]
                chunk_text = chunk.page_content.strip()
            except (KeyError, AttributeError, TypeError):
                flagged_citations[f"[{index}]"] = (
                    "INVALID_CONTEXT: "
                    "Referenced context block is malformed."
                )
                continue

            # Citation points to an empty chunk
            if not chunk_text:
                flagged_citations[f"[{index}]"] = (
                    "EMPTY_CONTEXT: "
                    "Target chunk contains no text content."
                )
                continue

            validated_indices.append(index)

        is_valid = len(flagged_citations) == 0

        return {
            "is_valid": is_valid,
            "flagged_issues": flagged_citations,
            "validated_indices": validated_indices,
        }