"""Strict text-only fallback routing; never authorize actions from router output."""

from dataclasses import dataclass
import json

from flzat_cognitive_pipeline.output_processing import StrictOutputProcessor


RESOLVED_MODES = {"general_qa", "visual_qa", "task_command", "social_dialogue"}


@dataclass(frozen=True)
class SemanticRoute:
    mode: str | None
    relation_to_pending: str
    needs_clarification: bool
    clarification_question: str | None


def parse_semantic_route(raw: str, *, has_pending: bool) -> SemanticRoute:
    """Reject incomplete, cross-stage, or contradictory model decisions."""
    value = json.loads(StrictOutputProcessor._strip_single_json_fence(raw))
    keys = {
        "mode", "relation_to_pending", "needs_clarification",
        "clarification_question",
    }
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError("routing output fields do not match the routing schema")
    mode = value["mode"]
    if mode is not None and (not isinstance(mode, str) or mode not in RESOLVED_MODES):
        raise ValueError("unsupported routing mode")
    relation = value["relation_to_pending"]
    if not isinstance(relation, str) or relation not in {"continue", "new_request"}:
        raise ValueError("invalid relation_to_pending")
    # Conversation existence is authoritative application state. With no
    # pending task there is nothing to continue, regardless of model wording.
    if not has_pending:
        relation = "new_request"
    clarify = value["needs_clarification"]
    if not isinstance(clarify, bool):
        raise ValueError("needs_clarification must be boolean")
    question = value["clarification_question"]
    if clarify:
        if (
            not isinstance(question, str) or not question.strip().endswith("?")
            or len(question) > 512
        ):
            raise ValueError("clarification requires one bounded question")
        question = question.strip()
    elif question is not None or mode is None:
        raise ValueError("resolved routing requires a mode and a null question")
    return SemanticRoute(mode, relation, clarify, question)
