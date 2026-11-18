import json
import os
from types import SimpleNamespace

import pytest

from src.data_loader import TrackRecord
from src.explainer import explain_recommendation
from src.guardrails import find_unsupported_terms
import src.llm_explainer as llm_explainer
from src.llm_explainer import (
    build_gemini_prompt,
    explain_with_gemini,
    parse_gemini_explanation,
)
from src.main import DEFAULT_REAL_DATASET_PATH, load_demo_records, make_demo_query
from src.retriever import RetrievalResult, retrieve_tracks


LISTENING_HISTORY_CLAIM = (
    "It's a pop track, so it should sound a lot like the songs you usually listen to."
)


@pytest.fixture
def retrieval_result() -> RetrievalResult:
    track = TrackRecord(
        track_name="Strong Match",
        artist_name="Test Artist",
        album_name="Test Album",
        genre="pop",
        vibe_tags=["high_energy", "positive", "danceable"],
        popularity=50.0,
        danceability=0.81,
        energy=0.82,
        acousticness=0.18,
        valence=0.76,
        tempo=120.0,
    )

    return RetrievalResult(
        track=track,
        score=6.50,
        evidence=[
            "genre matches preferred genre 'pop' (+2.00)",
            "energy closeness: 0.82 vs target 0.80 (+1.96)",
            "acousticness closeness: 0.18 vs target 0.20 (+0.98)",
            "valence closeness: 0.76 vs target 0.75 (+0.99)",
            "danceability closeness: 0.81 vs target 0.80 (+0.74)",
            "derived vibe tags for explanation: high_energy, positive, danceable",
        ],
    )


def test_build_gemini_prompt_uses_retrieved_fields_and_guardrails(
    retrieval_result: RetrievalResult,
):
    prompt = build_gemini_prompt(retrieval_result)

    assert "Use only the track metadata and retrieval evidence below." in prompt
    assert "Return exactly one valid JSON object" in prompt
    assert "name=Strong Match" in prompt
    assert "artist=Test Artist" in prompt
    assert "retrieval_score=6.50" in prompt
    assert "derived_vibe_tags=high_energy, positive, danceable" in prompt
    assert "energy closeness: 0.82 vs target 0.80 (+1.96)" in prompt
    assert "lyrics" in prompt
    assert "vocals" in prompt
    assert "storytelling" in prompt


def test_parse_gemini_explanation_accepts_plain_json():
    explanation = parse_gemini_explanation(
        '{"explanation": "This track matches the requested energy and genre."}'
    )

    assert explanation == "This track matches the requested energy and genre."


def test_parse_gemini_explanation_accepts_json_code_fence():
    explanation = parse_gemini_explanation(
        '```json\n{"explanation": "This track is grounded in retrieved evidence."}\n```'
    )

    assert explanation == "This track is grounded in retrieved evidence."


def test_parse_gemini_explanation_rejects_malformed_output():
    assert parse_gemini_explanation("This is not JSON.") is None
    assert parse_gemini_explanation("[]") is None
    assert parse_gemini_explanation('{"text": "missing explanation key"}') is None
    assert parse_gemini_explanation('{"explanation": "   "}') is None


def test_explain_with_gemini_uses_deterministic_fallback_without_key(
    monkeypatch: pytest.MonkeyPatch,
    retrieval_result: RetrievalResult,
):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    result = explain_with_gemini(retrieval_result)

    assert result.used_llm is False
    assert result.raw_response is None
    assert result.fallback_reason == "missing Gemini API key"
    assert result.guardrail_report is None
    assert result.explanation.track_name == "Strong Match"
    assert result.explanation.evidence == retrieval_result.evidence
    assert "Strong Match by Test Artist" in result.explanation.explanation


def test_explain_with_gemini_returns_safe_guardrail_checked_output(
    monkeypatch: pytest.MonkeyPatch,
    retrieval_result: RetrievalResult,
):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(
        llm_explainer,
        "_generate_gemini_response_text",
        lambda result, model: (
            '{"explanation": "Strong Match matches your pop preference. '
            'The evidence does not describe its vocals."}'
        ),
    )

    def approve(explanation, result, model):
        assert result is retrieval_result
        assert "matches your pop preference" in explanation
        return []

    monkeypatch.setattr(llm_explainer, "call_judge", approve)
    result = explain_with_gemini(retrieval_result)

    assert result.used_llm is True
    assert result.raw_response is not None
    assert result.fallback_reason is None
    assert result.guardrail_report is not None
    assert result.guardrail_report.passed is True
    assert result.guardrail_report.unsupported_terms == []
    assert result.explanation.explanation == (
        "Strong Match matches your pop preference. "
        "The evidence does not describe its vocals."
    )
    assert result.explanation.evidence == retrieval_result.evidence


def test_explain_with_gemini_falls_back_on_unsupported_claim_without_blocked_words(
    monkeypatch: pytest.MonkeyPatch,
    retrieval_result: RetrievalResult,
):
    genai = pytest.importorskip("google.genai")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    raw_response = json.dumps({"explanation": LISTENING_HISTORY_CLAIM})
    monkeypatch.setattr(
        llm_explainer,
        "_generate_gemini_response_text",
        lambda result, model: raw_response,
    )

    class FakeJudgeClient:
        def __init__(self, **kwargs):
            self.models = self

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def generate_content(self, *, contents, **kwargs):
            payload = json.loads(contents)
            assert payload["explanation"] == LISTENING_HISTORY_CLAIM
            assert payload["evidence"] == retrieval_result.evidence
            claims = [
                {
                    "claim": "This is a pop track.",
                    "status": "supported",
                    "reason": "The track genre is pop.",
                },
                {
                    "claim": "It sounds like songs the user usually listens to.",
                    "status": "unsupported",
                    "reason": "No listening history was provided.",
                },
            ]
            return SimpleNamespace(text=json.dumps({"claims": claims}))

    monkeypatch.setattr(genai, "Client", FakeJudgeClient)
    result = explain_with_gemini(retrieval_result)

    assert find_unsupported_terms(LISTENING_HISTORY_CLAIM) == []
    assert result.used_llm is False
    assert result.raw_response == raw_response
    assert result.fallback_reason == "Gemini explanation failed grounding check"
    assert result.guardrail_report is not None
    assert result.guardrail_report.passed is False
    assert result.guardrail_report.unsupported_terms == []
    violation, = result.guardrail_report.violations
    assert violation.code == "unsupported_claim"
    assert "No listening history was provided." in violation.message
    assert result.explanation == explain_recommendation(retrieval_result)


@pytest.fixture
def live_gemini_result():
    if os.getenv("RUN_LIVE_GEMINI_TESTS") != "1":
        pytest.skip("set RUN_LIVE_GEMINI_TESTS=1 to make live Gemini requests")
    if not (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")):
        pytest.skip("live Gemini tests require GEMINI_API_KEY or GOOGLE_API_KEY")
    records = load_demo_records(csv_path=DEFAULT_REAL_DATASET_PATH, limit=None)
    return retrieve_tracks(query=make_demo_query(), records=records, k=1)[0]


def test_live_gemini_judge_rejects_listening_history_claim(
    monkeypatch, live_gemini_result
):
    # Inject only the generator output. The production judge makes a real API call.
    raw_response = json.dumps({"explanation": LISTENING_HISTORY_CLAIM})
    monkeypatch.setattr(
        llm_explainer,
        "_generate_gemini_response_text",
        lambda result, model: raw_response,
    )
    result = explain_with_gemini(live_gemini_result)

    print("Handwritten hard case:", LISTENING_HISTORY_CLAIM)
    print("Fallback reason:", result.fallback_reason)
    print("Final explanation:", result.explanation.explanation)
    assert result.guardrail_report is not None, result.fallback_reason
    findings = result.guardrail_report.violations
    for finding in findings:
        print("Live judge finding:", finding.message)
    assert any(finding.code == "unsupported_claim" for finding in findings)
    assert result.fallback_reason == "Gemini explanation failed grounding check"
    assert not result.used_llm
    assert result.explanation == explain_recommendation(live_gemini_result)


def test_live_gemini_generation_and_review(live_gemini_result):
    result = explain_with_gemini(live_gemini_result)

    print("Retrieved evidence:", live_gemini_result.evidence)
    print("Live Gemini raw response:", result.raw_response)
    print("Used Gemini explanation:", result.used_llm)
    print("Fallback reason:", result.fallback_reason)
    print("Final explanation:", result.explanation.explanation)
    assert result.guardrail_report is not None, result.fallback_reason
    for finding in result.guardrail_report.violations:
        print("Live judge finding:", finding.message)
    if not result.used_llm:
        assert result.explanation == explain_recommendation(live_gemini_result)


@pytest.mark.parametrize("reply", ["not JSON", '{"claims": []}', None])
def test_explain_with_gemini_falls_back_if_judge_cannot_return_a_review(
    monkeypatch, retrieval_result, reply
):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    raw = '{"explanation": "This is pop."}'
    monkeypatch.setattr(
        llm_explainer, "_generate_gemini_response_text", lambda *args: raw
    )

    def review(*args):
        if reply is None:
            raise TimeoutError("judge timed out")
        return llm_explainer._parse_judge_violations(reply)

    monkeypatch.setattr(llm_explainer, "call_judge", review)
    result = explain_with_gemini(retrieval_result)

    assert result.used_llm is False
    assert result.raw_response == raw
    assert result.fallback_reason == "Gemini grounding check unavailable"
    assert "Strong Match by Test Artist" in result.explanation.explanation


def test_call_judge_sends_original_explanation_and_retrieval_evidence(
    monkeypatch, retrieval_result
):
    genai = pytest.importorskip("google.genai")

    class FakeClient:
        def __init__(self, http_options):
            assert http_options.timeout == 15000
            self.models = self

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def generate_content(self, *, model, contents, config):
            assert model == "judge-model"
            payload = json.loads(contents)
            assert payload["explanation"] == "This is pop."
            assert payload["track"]["genre"] == "pop"
            assert payload["evidence"] == retrieval_result.evidence
            assert "List every factual claim" in config.system_instruction
            return SimpleNamespace(
                text='{"claims": [{"claim": "This is pop.", '
                '"status": "supported", "reason": "The track genre is pop."}]}'
            )

    monkeypatch.setattr(genai, "Client", FakeClient)
    assert llm_explainer.call_judge("This is pop.", retrieval_result, "judge-model") == []
