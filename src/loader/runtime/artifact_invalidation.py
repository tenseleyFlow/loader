"""Semantic artifact invalidation and recovery-strategy selection."""

from __future__ import annotations

import re
from enum import StrEnum
from pathlib import Path

from .workflow_policy import (
    ArtifactEvidence,
    ArtifactEvidenceKind,
    ArtifactFreshness,
)


class WorkflowRecoveryStrategy(StrEnum):
    """Next workflow move when persisted artifacts drift."""

    NONE = "none"
    CLARIFY_REENTRY = "clarify_reentry"
    PLAN_REFRESH = "plan_refresh"
    FULL_REPLAN = "full_replan"


class ArtifactInvalidationAssessor:
    """Assess whether workflow artifacts still match the current task state."""

    def assess(
        self,
        *,
        task_statement: str,
        clarify_text: str | None,
        implementation_text: str | None,
        verification_text: str | None,
        acceptance_criteria: list[str],
        touched_files: list[str],
        last_verification_result: str | None,
        retry_count: int = 0,
        planned_artifacts_complete: bool = False,
    ) -> ArtifactFreshness:
        """Return stale-artifact state and the recommended recovery strategy."""

        if not clarify_text and not implementation_text and not verification_text:
            return ArtifactFreshness()

        plan_text = f"{implementation_text or ''}\n{verification_text or ''}".lower()
        brief_text = (clarify_text or "").lower()
        reasons: list[str] = []
        reason_codes: list[str] = []
        evidence: list[ArtifactEvidence] = []

        allow_repair_local_touchpoints = planned_artifacts_complete and retry_count > 0
        unexpected_paths = [
            name
            for path in touched_files
            if (name := _path_name(path))
            and not _text_covers_path_reference(plan_text, path)
        ]
        confirmed_touchpoints = [
            name
            for path in touched_files
            if (name := _path_name(path))
        ]
        confirmed_touchpoint_keys = {
            _path_reference_identity(path)
            for path in touched_files
            if _path_reference_identity(path)
        }
        inferred_touchpoints = [
            item
            for item in _extract_path_mentions(
                clarify_text,
                implementation_text,
                verification_text,
            )
            if _path_reference_identity(item) not in confirmed_touchpoint_keys
        ]
        stale_plan = False
        stale_brief = False

        for item in dict.fromkeys(confirmed_touchpoints):
            _append_evidence(
                evidence,
                ArtifactEvidenceKind.CONFIRMED_TOUCHPOINT,
                f"`{item}` was already touched during execution.",
            )
        for item in dict.fromkeys(inferred_touchpoints):
            _append_evidence(
                evidence,
                ArtifactEvidenceKind.INFERRED_TOUCHPOINT,
                f"Persisted artifacts still point at `{item}`.",
            )

        if unexpected_paths and not allow_repair_local_touchpoints:
            stale_plan = True
            reason_codes.append("touched_files_outside_plan")
            reasons.append(
                "Touched files outside the current plan: "
                + ", ".join(dict.fromkeys(unexpected_paths))
            )
        elif unexpected_paths:
            for item in dict.fromkeys(unexpected_paths):
                _append_evidence(
                    evidence,
                    ArtifactEvidenceKind.CONFIRMED_TOUCHPOINT,
                    "Verification repair touched supplemental file "
                    f"`{item}` after the originally planned artifacts were complete.",
                )

        acceptance_anchors = [
            item
            for item in acceptance_criteria
            if item.strip()
            and item.strip().lower() != task_statement.strip().lower()
            and "runtime verification evidence" not in item.strip().lower()
        ]
        for item in acceptance_anchors[:2]:
            _append_evidence(
                evidence,
                ArtifactEvidenceKind.ACCEPTANCE_ANCHOR,
                f"Current acceptance anchor: `{_short_requirement(item)}`.",
            )

        uncovered_criteria = [
            item
            for item in acceptance_anchors
            if not _text_covers_requirement(plan_text, item)
        ]
        if uncovered_criteria:
            stale_plan = True
            reason_codes.append("acceptance_criteria_outside_plan")
            reasons.append(
                "Acceptance criteria are missing from the current plan: "
                + "; ".join(uncovered_criteria[:2])
            )
            for item in uncovered_criteria[:2]:
                _append_evidence(
                    evidence,
                    ArtifactEvidenceKind.ACCEPTANCE_ANCHOR,
                    f"Plan coverage is missing acceptance anchor `{_short_requirement(item)}`.",
                )

        if brief_text:
            brief_gaps = [
                item
                for item in acceptance_criteria
                if item.strip()
                and item.strip().lower() != task_statement.strip().lower()
                and "runtime verification evidence" not in item.strip().lower()
                and not _text_covers_requirement(brief_text, item)
            ]
            if brief_gaps and last_verification_result == "failed":
                stale_brief = True
                reason_codes.append("brief_missing_acceptance_scope")
                reasons.append(
                    "The clarify brief no longer captures the active acceptance criteria: "
                    + "; ".join(brief_gaps[:2])
                )
                for item in brief_gaps[:2]:
                    _append_evidence(
                        evidence,
                        ArtifactEvidenceKind.VERIFICATION_CONTRADICTION,
                        "Failed verification exposed missing brief coverage for "
                        f"`{_short_requirement(item)}`.",
                    )

            out_of_brief_paths = [
                name
                for path in touched_files
                if (name := _path_name(path))
                and name in unexpected_paths
                and not _text_covers_path_reference(brief_text, path)
            ]
            if out_of_brief_paths:
                stale_brief = True
                reason_codes.append("touchpoints_outside_brief")
                reasons.append(
                    "The clarify brief no longer matches the touched files: "
                    + ", ".join(dict.fromkeys(out_of_brief_paths))
                )
                for item in dict.fromkeys(out_of_brief_paths):
                    _append_evidence(
                        evidence,
                        ArtifactEvidenceKind.CONTRADICTED_ASSUMPTION,
                        f"Clarify scope assumed `{item}` stayed out of scope.",
                    )

            if not _text_covers_requirement(brief_text, task_statement):
                stale_brief = True
                reason_codes.append("task_drifted_beyond_brief")
                reasons.append(
                    "The clarify brief no longer reflects the current task framing."
                )
                _append_evidence(
                    evidence,
                    ArtifactEvidenceKind.TASK_BOUNDARY_CHANGE,
                    "The active task framing outgrew the persisted clarify brief.",
                )

        recovery_strategy = WorkflowRecoveryStrategy.NONE
        if stale_brief and stale_plan:
            recovery_strategy = WorkflowRecoveryStrategy.FULL_REPLAN
        elif stale_brief:
            recovery_strategy = WorkflowRecoveryStrategy.CLARIFY_REENTRY
        elif stale_plan:
            recovery_strategy = WorkflowRecoveryStrategy.PLAN_REFRESH

        return ArtifactFreshness(
            stale_brief=stale_brief,
            stale_plan=stale_plan,
            reasons=list(dict.fromkeys(reasons)),
            reason_codes=list(dict.fromkeys(reason_codes)),
            recovery_strategy=recovery_strategy.value,
            evidence=evidence,
        )


def _path_name(path: str) -> str:
    normalized = str(path).strip()
    if not normalized:
        return ""
    return normalized.rsplit("/", maxsplit=1)[-1].strip()


def _path_reference_identity(path: str) -> str:
    normalized = _path_name(path)
    if not normalized:
        return ""
    return _canonical_path_reference(normalized)


def _text_covers_path_reference(text: str, path: str) -> bool:
    normalized_text = text.lower()
    candidates = [candidate for candidate in (str(path).strip(), _path_name(path)) if candidate]

    for candidate in candidates:
        if candidate.lower() in normalized_text:
            return True

    canonical_text = _canonical_path_reference(text)
    if any(
        canonical_candidate and canonical_candidate in canonical_text
        for canonical_candidate in (_canonical_path_reference(candidate) for candidate in candidates)
    ):
        return True

    return any(anchor in canonical_text for anchor in _directory_reference_anchors(path))


def _canonical_path_reference(value: str) -> str:
    normalized = value.lower().strip()
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
    return " ".join(normalized.split())


def _directory_reference_anchors(path: str) -> tuple[str, ...]:
    normalized = str(path).strip()
    if not normalized:
        return ()

    candidate = Path(normalized)
    directory = candidate if not candidate.suffix else candidate.parent
    parts = [part for part in directory.parts if part not in {"", "/", "~"}]
    if len(parts) < 2:
        return ()

    anchors: list[str] = []
    for width in range(min(4, len(parts)), 1, -1):
        anchor = _canonical_path_reference("/".join(parts[-width:]))
        if anchor and anchor not in anchors:
            anchors.append(anchor)
    return tuple(anchors)


def _text_covers_requirement(text: str, requirement: str) -> bool:
    normalized_text = text.lower()
    normalized_requirement = requirement.lower()
    if normalized_requirement in normalized_text:
        return True

    tokens = [
        token
        for token in re.findall(r"[a-z0-9_./-]+", normalized_requirement)
        if len(token) > 2 and token not in _STOP_WORDS
    ]
    if not tokens:
        return normalized_requirement.strip() in normalized_text
    matches = sum(1 for token in tokens if token in normalized_text)
    threshold = max(1, min(2, len(tokens)))
    return matches >= threshold


def _extract_path_mentions(*texts: str | None) -> list[str]:
    mentions: list[str] = []
    seen: set[str] = set()
    for text in texts:
        if not text:
            continue
        for match in re.findall(r"[\w./-]+\.[a-z0-9]+", text):
            normalized = match.strip("`'\",.:;()[]{}")
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            mentions.append(normalized)
    return mentions


def _short_requirement(requirement: str, *, limit: int = 72) -> str:
    normalized = " ".join(str(requirement).split()).strip()
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3].rstrip() + "..."


def _append_evidence(
    evidence: list[ArtifactEvidence],
    kind: ArtifactEvidenceKind,
    summary: str,
) -> None:
    item = ArtifactEvidence(kind=kind.value, summary=summary)
    if any(
        existing.kind == item.kind and existing.summary == item.summary
        for existing in evidence
    ):
        return
    evidence.append(item)


_STOP_WORDS = {
    "the",
    "and",
    "with",
    "that",
    "this",
    "into",
    "from",
    "without",
    "while",
    "when",
    "then",
    "must",
    "should",
    "exists",
}
