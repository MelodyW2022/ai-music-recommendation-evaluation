import json

import pytest

from src.data_loader import TrackRecord
from src.evaluator import (
    PREDEFINED_CASES,
    SIMULATED_GEMINI_CASES,
    SIMULATED_UNSUPPORTED_CASE,
    evaluate_records,
    explain_simulated_response,
)
from src.llm_explainer import FORMAT_FAILURE_REASON, GUARDRAIL_FAILURE_REASON
from src.retriever import RetrievalQuery, retrieve_tracks


@pytest.fixture
def records():
    return [
        TrackRecord(
            track_name="Pop Match",
            artist_name="Test Artist",
            album_name="Test Album",
            genre="pop",
            vibe_tags=["high_energy", "positive", "danceable"],
            popularity=70.0,
            danceability=0.82,
            energy=0.84,
            acousticness=0.18,
            valence=0.86,
            tempo=124.0,
        ),
        TrackRecord(
            track_name="Acoustic Match",
            artist_name="Test Artist",
            album_name="Test Album",
            genre="acoustic",
            vibe_tags=["low_energy", "acoustic"],
            popularity=55.0,
            danceability=0.54,
            energy=0.31,
            acousticness=0.88,
            valence=0.52,
            tempo=86.0,
        ),
        TrackRecord(
            track_name="Dance Match",
            artist_name="Test Artist",
            album_name="Test Album",
            genre="dance",
            vibe_tags=["high_energy", "positive", "electronic", "danceable"],
            popularity=80.0,
            danceability=0.91,
            energy=0.92,
            acousticness=0.08,
            valence=0.77,
            tempo=128.0,
        ),
    ]


def test_explain_simulated_response_accepts_safe_json(records):
    retrieval_result = retrieve_tracks(
        RetrievalQuery(
            preferred_genre="pop",
            target_energy=0.80,
            target_acousticness=0.20,
            target_valence=0.85,
            target_danceability=0.80,
        ),
        records,
        k=1,
    )[0]

    result = explain_simulated_response(
        retrieval_result,
        *SIMULATED_GEMINI_CASES[0],
    )

    assert result.used_llm is True
    assert result.fallback_reason is None
    assert result.guardrail_report is not None
    assert result.guardrail_report.passed is True


def test_explain_simulated_response_falls_back_on_unsupported_claims(records):
    retrieval_result = retrieve_tracks(
        RetrievalQuery(
            preferred_genre="pop",
            target_energy=0.80,
            target_acousticness=0.20,
            target_valence=0.85,
            target_danceability=0.80,
        ),
        records,
        k=1,
    )[0]

    result = explain_simulated_response(
        retrieval_result,
        *SIMULATED_UNSUPPORTED_CASE,
    )

    assert result.used_llm is False
    assert result.fallback_reason == GUARDRAIL_FAILURE_REASON
    assert result.guardrail_report is not None
    assert result.guardrail_report.violations[0].code == "unsupported_claim"
    assert "Pop Match by Test Artist" in result.explanation.explanation


def test_explain_simulated_response_falls_back_on_format_failure(records):
    retrieval_result = retrieve_tracks(
        RetrievalQuery(
            preferred_genre="pop",
            target_energy=0.80,
            target_acousticness=0.20,
            target_valence=0.85,
            target_danceability=0.80,
        ),
        records,
        k=1,
    )[0]

    result = explain_simulated_response(retrieval_result, "not json", "")

    assert result.used_llm is False
    assert result.fallback_reason == FORMAT_FAILURE_REASON
    assert result.guardrail_report is None


def test_explain_simulated_response_falls_back_on_contradicted_claim(records):
    retrieval_result = retrieve_tracks(PREDEFINED_CASES[0].query, records, k=1)[0]
    claim = "This song is a classical track."
    raw_response = json.dumps({"explanation": claim})
    raw_review = json.dumps({"claims": [{
        "claim": claim,
        "status": "contradicted",
        "reason": "The retrieved genre is pop.",
    }]})

    result = explain_simulated_response(retrieval_result, raw_response, raw_review)

    assert result.used_llm is False
    assert result.fallback_reason == GUARDRAIL_FAILURE_REASON
    assert result.guardrail_report is not None
    assert result.guardrail_report.violations[0].code == "contradicted_claim"
    assert claim not in result.explanation.explanation


def test_evaluate_records_reports_reliability_metrics(records):
    summary = evaluate_records(records, simulate_llm_failures=True)

    assert len(summary.items) == 3
    assert summary.unsupported_claim_rate == pytest.approx(1 / 3)
    assert summary.fallback_rate == pytest.approx(2 / 3)
    assert summary.format_failure_rate == pytest.approx(1 / 3)
    assert summary.global_evidence_coverage == pytest.approx(1.0)
