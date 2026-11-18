from dataclasses import dataclass
import json
from typing import List, Optional

from .explainer import RecommendationExplanation, explain_recommendation
from .guardrails import (
    GuardrailReport,
    calculate_global_evidence_coverage,
    check_explanation,
)
from .llm_explainer import (
    FORMAT_FAILURE_REASON,
    LLMExplanationResult,
    explain_with_gemini,
    parse_judge_violations,
    review_gemini_response,
)
from .retriever import RetrievalQuery, RetrievalResult, retrieve_tracks


@dataclass(frozen=True)
class EvaluationCase:
    """
    One predefined benchmark query for reliability evaluation.
    """

    name: str
    query: RetrievalQuery


@dataclass(frozen=True)
class EvaluationItem:
    """
    Reliability result for one evaluated recommendation.
    """

    case_name: str
    track_name: str
    raw_response: Optional[str]
    final_explanation: RecommendationExplanation
    raw_guardrail_report: Optional[GuardrailReport]
    final_guardrail_report: GuardrailReport
    used_llm: bool
    fallback_reason: Optional[str]


@dataclass(frozen=True)
class EvaluationSummary:
    """
    Aggregate reliability metrics across the benchmark cases.
    """

    items: List[EvaluationItem]
    unsupported_claim_rate: float
    fallback_rate: float
    format_failure_rate: float
    global_evidence_coverage: float


PREDEFINED_CASES = [
    EvaluationCase(
        name="high_energy_pop",
        query=RetrievalQuery(
            preferred_genre="pop",
            target_energy=0.80,
            target_acousticness=0.20,
            target_valence=0.85,
            target_danceability=0.80,
        ),
    ),
    EvaluationCase(
        name="chill_acoustic",
        query=RetrievalQuery(
            preferred_genre="acoustic",
            target_energy=0.30,
            target_acousticness=0.85,
            target_valence=0.50,
            target_danceability=0.55,
        ),
    ),
    EvaluationCase(
        name="intense_dance",
        query=RetrievalQuery(
            preferred_genre="dance",
            target_energy=0.90,
            target_acousticness=0.10,
            target_valence=0.75,
            target_danceability=0.90,
        ),
    ),
]

SIMULATED_UNSUPPORTED_CASE = (
    '{"explanation": "This track sounds like the songs you usually listen to."}',
    json.dumps({"claims": [{
        "claim": "This track sounds like the songs you usually listen to.",
        "status": "unsupported",
        "reason": "The evidence does not include the user's listening history.",
    }]}),
)

SIMULATED_GEMINI_CASES = [
    (
        '{"explanation": "The song matches your chosen genre."}',
        json.dumps({"claims": [{
            "claim": "The song matches your chosen genre.",
            "status": "supported",
            "reason": "The retriever reported a genre match.",
        }]}),
    ),
    SIMULATED_UNSUPPORTED_CASE,
    ("This is not JSON, so the parser should reject it.", ""),
]

def evaluate_records(
    records,
    use_gemini: bool = False,
    simulate_llm_failures: bool = True,
    k: int = 1,
) -> EvaluationSummary:
    """
    Run predefined retrieval cases and calculate reliability metrics.

    By default, this uses simulated Gemini-style responses so the evaluator can
    demonstrate format failure and guardrail fallback without requiring an API key.
    """
    items: List[EvaluationItem] = []

    for index, case in enumerate(PREDEFINED_CASES):
        retrieved = retrieve_tracks(query=case.query, records=records, k=k)
        if not retrieved:
            continue

        result = retrieved[0]
        if simulate_llm_failures:
            raw_response, raw_review = SIMULATED_GEMINI_CASES[index]
            llm_result = explain_simulated_response(result, raw_response, raw_review)
        elif use_gemini:
            llm_result = explain_with_gemini(result)
        else:
            explanation = explain_recommendation(result)
            llm_result = LLMExplanationResult(
                explanation=explanation,
                used_llm=False,
                raw_response=None,
                fallback_reason=None,
                guardrail_report=check_explanation(explanation),
            )

        final_report = (
            llm_result.guardrail_report
            if llm_result.used_llm and llm_result.guardrail_report is not None
            else check_explanation(llm_result.explanation)
        )
        items.append(
            EvaluationItem(
                case_name=case.name,
                track_name=result.track.track_name,
                raw_response=llm_result.raw_response,
                final_explanation=llm_result.explanation,
                raw_guardrail_report=llm_result.guardrail_report,
                final_guardrail_report=final_report,
                used_llm=llm_result.used_llm,
                fallback_reason=llm_result.fallback_reason,
            )
        )

    return summarize_items(items)


def explain_simulated_response(
    result: RetrievalResult, raw_response: str, raw_review: str
) -> LLMExplanationResult:
    """Replay a canned candidate and judge response through the live review logic."""
    return review_gemini_response(
        result, raw_response, lambda _: parse_judge_violations(raw_review)
    )


def summarize_items(items: List[EvaluationItem]) -> EvaluationSummary:
    """
    Calculate aggregate reliability metrics from item-level reports.
    """
    total = len(items)
    if total == 0:
        return EvaluationSummary(
            items=[],
            unsupported_claim_rate=0.0,
            fallback_rate=0.0,
            format_failure_rate=0.0,
            global_evidence_coverage=0.0,
        )

    unsupported_count = sum(
        1
        for item in items
        if item.raw_guardrail_report
        and any(
            violation.code in ("unsupported_claim", "contradicted_claim")
            for violation in item.raw_guardrail_report.violations
        )
    )
    fallback_count = sum(1 for item in items if item.fallback_reason)
    format_failure_count = sum(
        1 for item in items if item.fallback_reason == FORMAT_FAILURE_REASON
    )

    return EvaluationSummary(
        items=items,
        unsupported_claim_rate=unsupported_count / total,
        fallback_rate=fallback_count / total,
        format_failure_rate=format_failure_count / total,
        global_evidence_coverage=calculate_global_evidence_coverage(
            [item.final_guardrail_report for item in items]
        ),
    )
