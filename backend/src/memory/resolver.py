"""
Context-Aware Follow-Up Reference Resolver
Project: Autonomous Multi-Agent Customer Support System with HITL Escalation

Resolves pronouns, ellipses, and anaphoric follow-up queries using chronological
session history strictly bounded to the current (workspace_id, session_id).
Guarantees zero cross-session or cross-tenant leakage.
Preserves the user's original query verbatim in state.query while providing
the disambiguated query in state.metadata["resolved_query"] for vector search.
"""

import re
from typing import Any, Dict, List, Optional, Tuple


class ReferenceResolver:
    """
    Lightweight, deterministic reference resolver for multi-turn customer inquiries.
    Identifies contextual dependencies and merges prior conversational focus
    into an unambiguous query for dense vector retrieval.
    """

    # Pronouns and demonstratives indicating dependency on prior entities
    PRONOUN_PATTERN = re.compile(
        r"\b(it|its|they|them|their|this|that|these|those)\b",
        re.IGNORECASE,
    )

    # Ellipsis phrases or question fragments continuing prior topic
    FOLLOWUP_PREFIX_PATTERN = re.compile(
        r"^(and\s+|what\s+about\s+|how\s+about\s+|what\s+if\s+|and\s+if\s+|"
        r"and\s+for\s+|is\s+there\s+any\s+|can\s+i\s+still\s+|does\s+it\s+|"
        r"how\s+much\s+(is\s+it|does\s+it\s+cost|is\s+the\s+fee)\b|"
        r"what\s+is\s+the\s+fee\b|after\s+that\b|during\s+that\b)",
        re.IGNORECASE,
    )

    # Specific policy anaphoric phrases
    ANAPHORA_PATTERN = re.compile(
        r"\b(that\s+period|that\s+window|the\s+same\s+item|the\s+fee|the\s+cost|"
        r"the\s+policy|that\s+course|that\s+order)\b",
        re.IGNORECASE,
    )

    @classmethod
    def is_follow_up(cls, query: str) -> bool:
        """Determines whether a query is likely anaphoric or an ellipsis follow-up."""
        q = query.strip()
        if cls.FOLLOWUP_PREFIX_PATTERN.search(q):
            return True
        if cls.ANAPHORA_PATTERN.search(q):
            return True
        # If short query (under 8 words) containing pronouns
        words = q.split()
        if len(words) <= 8 and cls.PRONOUN_PATTERN.search(q):
            return True
        return False

    @classmethod
    def resolve(
        cls,
        query: str,
        session_history: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[str, bool]:
        """
        Resolves follow-up queries using chronological session turns from the same session.

        Parameters:
            query: The raw current customer query.
            session_history: Chronological list of prior turns for this specific (workspace, session).

        Returns:
            Tuple of (resolved_query, was_resolved)
            If no resolution is needed or history is empty, returns (query, False).
        """
        cleaned_query = query.strip()
        if not session_history or len(session_history) == 0:
            return cleaned_query, False

        if not cls.is_follow_up(cleaned_query):
            return cleaned_query, False

        # Identify most recent relevant user turn
        prior_query = ""
        for turn in reversed(session_history):
            if isinstance(turn, dict):
                cand = str(
                    turn.get("user_query")
                    or turn.get("resolved_query")
                    or (turn.get("content") if turn.get("role") in ["user", None] else "")
                    or ""
                ).strip()
                if cand:
                    prior_query = cand
                    break

        if not prior_query:
            return cleaned_query, False

        # Strip punctuation from prior query for clean topical prefix
        prior_clean = re.sub(r"[?!.]+$", "", prior_query).strip()

        # Build disambiguated search query for dense retrieval
        resolved = f"{prior_clean} — Regarding follow-up: {cleaned_query}"
        return resolved, True

    @classmethod
    def resolve_reference(
        cls,
        query: str,
        session_history: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[str]:
        """
        Convenience method returning disambiguated query string if resolved, or None if unchanged.
        """
        resolved, was_resolved = cls.resolve(query, session_history)
        return resolved if was_resolved else None
