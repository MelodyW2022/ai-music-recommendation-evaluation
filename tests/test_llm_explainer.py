import json
import os
from types import SimpleNamespace

import pytest

import src.alternative_judges as alternative_judges
from src.alternative_judges import (
    CLAUDE_JUDGE_MODEL,
    GROQ_JUDGE_MODEL,
    review_with_claude,
    review_with_groq,
)
from src.data_loader import TrackRecord
from src.evaluator import PREDEFINED_CASES, EvaluationCase
from src.explainer import explain_recommendation
from src.guardrails import GuardrailViolation
import src.llm_explainer as llm_explainer
from src.llm_explainer import (
    build_gemini_prompt,
    explain_with_gemini,
    parse_gemini_explanation,
    review_gemini_response,
)
from src.main import DEFAULT_REAL_DATASET_PATH, load_demo_records, make_demo_query
from src.retriever import RetrievalQuery, RetrievalResult, retrieve_tracks


LISTENING_HISTORY_CLAIM = (
    "It's a pop track, so it should sound a lot like the songs you usually listen to."
)

LIVE_GENERATION_CASES = [
    *PREDEFINED_CASES,
    EvaluationCase("low_energy_jazz", RetrievalQuery("jazz", 0.24, 0.82, 0.36, 0.31)),
    EvaluationCase("high_energy_rock", RetrievalQuery("rock", 0.85, 0.10, 0.26, 0.36)),
    EvaluationCase(
        "genre_unavailable", RetrievalQuery("baroque-pop", 0.36, 0.55, 0.68, 0.63)
    ),
]

# Candidate and evidence-review expectation from the frozen README baseline.
BASELINE_CANDIDATES = {
    "high_energy_pop": ("Ishqam is a great match for your preferences because it aligns with your favorite pop genre and delivers the high-energy, danceable electronic sound you enjoy.", False),
    "chill_acoustic": ("This track is a great match because it aligns with your preference for acoustic music and closely mirrors your target energy and acousticness levels.", True),
    "intense_dance": ("Tiësto and Charli XCX's 'Hot in It' is a great match because it aligns with your preferred dance genre and delivers the high-energy, positive, and danceable electronic sound you enjoy.", False),
    "low_energy_jazz": ("This track is a strong match because it aligns with your preferred jazz genre and features a moderate musical profile that closely mirrors your target energy and mood preferences.", False),
    "high_energy_rock": ("This track is a strong match for your preferences because it aligns with your favorite rock genre and features the high-energy, fast-tempo, and darker electronic vibe you enjoy.", False),
    "genre_unavailable": ("This honky-tonk track is recommended because its low-energy, positive, and danceable vibe aligns closely with your preferred acoustic and energy levels.", True),
}

# These candidates were generated normally, but their first review was unavailable.
FOLLOWUP_CANDIDATES = {
    "intense_dance": "Tiësto and Charli XCX's 'Hot in It' is a great match because it aligns with your preference for high-energy, danceable electronic music that mirrors your target valence and energy levels.",
    "high_energy_rock": "This track is a strong match for your preferences because it aligns with your favorite rock genre and features a high-energy, fast-tempo sound that closely mirrors your target acoustic and valence levels.",
}

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
    assert "make only claims these inputs support" in prompt


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
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
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
    assert result.explanation.explanation == (
        "Strong Match matches your pop preference. "
        "The evidence does not describe its vocals."
    )
    assert result.explanation.evidence == retrieval_result.evidence


def test_explain_with_gemini_falls_back_on_unsupported_listening_history_claim(
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

    assert result.used_llm is False
    assert result.raw_response == raw_response
    assert result.fallback_reason == "Gemini explanation failed grounding check"
    assert result.guardrail_report is not None
    assert result.guardrail_report.passed is False
    violation, = result.guardrail_report.violations
    assert violation.code == "unsupported_claim"
    assert "No listening history was provided." in violation.message
    assert result.explanation == explain_recommendation(retrieval_result)


@pytest.fixture(scope="module")
def live_gemini_records():
    if os.getenv("RUN_LIVE_GEMINI_TESTS") != "1":
        pytest.skip("set RUN_LIVE_GEMINI_TESTS=1 to make live Gemini requests")
    if not (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")):
        pytest.skip("live Gemini tests require GEMINI_API_KEY or GOOGLE_API_KEY")
    return load_demo_records(csv_path=DEFAULT_REAL_DATASET_PATH, limit=None)


def test_live_gemini_judge_rejects_listening_history_claim(
    monkeypatch, live_gemini_records
):
    live_gemini_result = retrieve_tracks(make_demo_query(), live_gemini_records, k=1)[0]
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


def print_live_review(stage, case, retrieved, result):
    report = result.guardrail_report
    review = {
        "stage": stage,
        "case": case.name,
        "track": retrieved.track.track_name,
        "evidence": retrieved.evidence,
        "candidate": parse_gemini_explanation(result.raw_response or ""),
        "accepted": result.used_llm,
        "fallback_reason": result.fallback_reason,
        "findings": [violation.message for violation in report.violations] if report else [],
        "displayed": result.explanation.explanation,
    }
    print("Live review:", json.dumps(review, ensure_ascii=False))


def test_live_gemini_replays_original_candidates(live_gemini_records):
    failures = []
    cases = [
        ("baseline_replay", case, *BASELINE_CANDIDATES[case.name])
        for case in LIVE_GENERATION_CASES
        if os.getenv("LIVE_GEMINI_MODE") != "run-live-gemini-followup"
    ]
    cases.extend(
        ("fresh_review_retry", case, FOLLOWUP_CANDIDATES[case.name], None)
        for case in LIVE_GENERATION_CASES
        if case.name in FOLLOWUP_CANDIDATES
    )
    for stage, case, candidate, expected in cases:
        retrieved = retrieve_tracks(case.query, live_gemini_records, k=1)[0]

        def judge(text):
            try:
                return llm_explainer.call_judge(
                    text, retrieved, llm_explainer.DEFAULT_GEMINI_MODEL
                )
            except Exception as exc:
                print(f"Live judge error for {case.name}: {type(exc).__name__}")
                if type(exc) is ValueError:
                    print(f"Judge validation: {exc}")
                raise

        result = review_gemini_response(
            retrieved,
            json.dumps({"explanation": candidate}),
            judge,
        )
        print_live_review(stage, case, retrieved, result)
        if result.guardrail_report is None:
            failures.append(f"{case.name}: {result.fallback_reason}")
        elif expected is not None and result.used_llm != expected:
            failures.append(f"{case.name}: expected accepted={expected}")
        elif not result.used_llm and result.explanation != explain_recommendation(retrieved):
            failures.append(f"{case.name}: expected the deterministic fallback")

    assert not failures, "\n".join(failures)


def test_live_gemini_generation_and_review(monkeypatch, live_gemini_records):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    failures = []
    for case in LIVE_GENERATION_CASES:
        retrieved = retrieve_tracks(case.query, live_gemini_records, k=1)[0]
        result = explain_with_gemini(retrieved)
        report = result.guardrail_report

        print_live_review("fresh_generation", case, retrieved, result)
        if report is None:
            failures.append(f"{case.name}: {result.fallback_reason}")
            continue
        if not result.used_llm and result.explanation != explain_recommendation(retrieved):
            failures.append(f"{case.name}: expected the deterministic fallback")

    assert not failures, "\n".join(failures)


def test_live_judge_comparison(live_gemini_records):
    if os.getenv("RUN_LIVE_JUDGE_COMPARISON") != "1":
        pytest.skip("set RUN_LIVE_JUDGE_COMPARISON=1 to compare live judges")

    reviewers = (
        (llm_explainer.DEFAULT_GEMINI_MODEL, lambda text, retrieved:
            llm_explainer.call_judge(text, retrieved, llm_explainer.DEFAULT_GEMINI_MODEL)),
        (CLAUDE_JUDGE_MODEL, review_with_claude),
        (GROQ_JUDGE_MODEL, review_with_groq),
    )
    baseline = [
        (case, *BASELINE_CANDIDATES[case.name])
        for case in LIVE_GENERATION_CASES
    ]
    followups = [
        (case, FOLLOWUP_CANDIDATES[case.name], False)
        for case in LIVE_GENERATION_CASES
        if case.name in FOLLOWUP_CANDIDATES
    ]
    groq_only = os.getenv("JUDGE_COMPARISON_MODE") == "run-groq-diagnostic"
    if groq_only:
        reviewers = ((GROQ_JUDGE_MODEL, review_with_groq),)
    failures = []
    for batch, cases in (("baseline", baseline), ("followup", followups)):
        for case, candidate, expected in cases:
            retrieved = retrieve_tracks(case.query, live_gemini_records, k=1)[0]
            for model, judge in reviewers:
                verdict = {
                    "batch": batch,
                    "case": case.name,
                    "track": retrieved.track.track_name,
                    "model": model,
                    "expected": "accept" if expected else "reject",
                }
                try:
                    findings = judge(candidate, retrieved)
                except Exception as exc:
                    reason = str(exc) if type(exc) in (RuntimeError, ValueError) else type(exc).__name__
                    verdict.update(decision="unavailable", error=reason)
                    failures.append(f"{batch}/{case.name}/{model}: {reason}")
                else:
                    verdict.update(
                        decision="reject" if findings else "accept",
                        findings=[finding.message for finding in findings],
                    )
                print("Judge comparison:", json.dumps(verdict, ensure_ascii=False))
                if groq_only and failures:
                    pytest.fail(failures[-1])

    assert not failures, "\n".join(failures)


@pytest.mark.parametrize("outcome", ["pass", "reject", "unavailable", "first_reject"])
def test_claude_second_review(monkeypatch, retrieval_result, outcome):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    candidate = "This track matches your selected genre."
    monkeypatch.setattr(
        llm_explainer,
        "_generate_gemini_response_text",
        lambda *_: json.dumps({"explanation": candidate}),
    )
    violation = GuardrailViolation("unsupported_claim", "Unsupported preference", "error")
    calls = []

    def gemini(text, retrieved, model):
        calls.append(("Gemini", text, retrieved))
        return [violation] if outcome == "first_reject" else []

    def claude(text, retrieved):
        calls.append(("Claude", text, retrieved))
        if outcome == "unavailable":
            raise TimeoutError
        return [violation] if outcome == "reject" else []

    monkeypatch.setattr(llm_explainer, "call_judge", gemini)
    monkeypatch.setattr(alternative_judges, "review_with_claude", claude)
    result = explain_with_gemini(retrieval_result)

    reviewers = ["Gemini"] if outcome == "first_reject" else ["Gemini", "Claude"]
    assert calls == [(reviewer, candidate, retrieval_result) for reviewer in reviewers]
    assert result.used_llm == (outcome == "pass")
    if outcome == "pass":
        assert result.explanation.explanation == candidate
    else:
        assert result.explanation == explain_recommendation(retrieval_result)
        assert result.fallback_reason == {
            "reject": "Claude explanation failed grounding check",
            "unavailable": "Claude grounding check unavailable",
            "first_reject": "Gemini explanation failed grounding check",
        }[outcome]


def test_live_claude_second_review(monkeypatch, live_gemini_records):
    if os.getenv("RUN_LIVE_CLAUDE_REVIEW") != "1":
        pytest.skip("set RUN_LIVE_CLAUDE_REVIEW=1 to test the second review")
    assert os.getenv("ANTHROPIC_API_KEY"), "ANTHROPIC_API_KEY must be configured"

    # Use saved explanations so this test checks the reviewers independently of
    # variation in generation.
    acoustic = next(case for case in LIVE_GENERATION_CASES if case.name == "chill_acoustic")
    dance = next(case for case in LIVE_GENERATION_CASES if case.name == "intense_dance")
    accepted_candidate = BASELINE_CANDIDATES[acoustic.name][0]
    rejected_candidate = FOLLOWUP_CANDIDATES[dance.name]
    production_judge = llm_explainer.call_judge
    claude_calls = []

    def generate(retrieved, model):
        candidate = accepted_candidate if retrieved.track.genre == "acoustic" else rejected_candidate
        return json.dumps({"explanation": candidate})

    def initial_judge(text, retrieved, model):
        # Reproduce Gemini's observed false acceptance in the saved dance case.
        if text == rejected_candidate:
            return []
        return production_judge(text, retrieved, model)

    def claude(text, retrieved):
        claude_calls.append(text)
        return review_with_claude(text, retrieved)

    monkeypatch.setattr(llm_explainer, "_generate_gemini_response_text", generate)
    monkeypatch.setattr(llm_explainer, "call_judge", initial_judge)
    monkeypatch.setattr(alternative_judges, "review_with_claude", claude)
    for case, expected in ((acoustic, True), (dance, False)):
        retrieved = retrieve_tracks(case.query, live_gemini_records, k=1)[0]
        result = explain_with_gemini(retrieved)
        print_live_review("claude_second_review", case, retrieved, result)
        assert result.used_llm == expected, result.fallback_reason
        if not expected:
            assert result.fallback_reason == "Claude explanation failed grounding check"
            assert result.explanation == explain_recommendation(retrieved)
    assert claude_calls == [accepted_candidate, rejected_candidate]


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
        return llm_explainer.parse_judge_violations(reply)

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
            assert config.response_json_schema == llm_explainer.JUDGE_RESPONSE_SCHEMA
            return SimpleNamespace(
                text='{"claims": [{"claim": "This is pop.", '
                '"status": "supported", "reason": "The track genre is pop."}]}'
            )

    monkeypatch.setattr(genai, "Client", FakeClient)
    assert llm_explainer.call_judge("This is pop.", retrieval_result, "judge-model") == []
