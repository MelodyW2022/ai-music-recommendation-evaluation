from dataclasses import replace

from src.data_loader import TrackRecord
from src.explainer import explain_recommendation, explain_recommendations
from src.retriever import RetrievalResult


def make_result() -> RetrievalResult:
    track = TrackRecord(
        track_name="Strong Match",
        artist_name="Test Artist",
        album_name="Test Album",
        genre="pop",
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
        ],
    )


def test_explain_recommendation_uses_retrieved_fields_and_evidence():
    result = make_result()
    explanation = explain_recommendation(result)

    assert explanation.track_name == "Strong Match"
    assert explanation.artist_name == "Test Artist"
    assert explanation.genre == "pop"
    assert explanation.score == 6.50
    assert "Strong Match by Test Artist" in explanation.explanation
    assert "Here is how it compares with your request." in explanation.explanation
    assert "energy closeness: 0.82 vs target 0.80" in explanation.explanation
    assert all(line in explanation.explanation for line in result.evidence)


def test_explanation_reports_a_large_energy_gap_without_claiming_a_close_match():
    result = replace(
        make_result(),
        evidence=["energy closeness: 0.64 vs target 0.24 (+1.20)"],
    )

    explanation = explain_recommendation(result).explanation

    assert "0.64 vs target 0.24" in explanation
    assert "close to the requested profile" not in explanation


def test_explain_recommendation_keeps_raw_evidence_for_audit():
    result = make_result()
    explanation = explain_recommendation(result)

    assert explanation.evidence == result.evidence


def test_explain_recommendations_handles_ranked_lists():
    explanations = explain_recommendations([make_result(), make_result()])

    assert len(explanations) == 2
    assert all(item.explanation.strip() for item in explanations)


def test_explain_recommendation_handles_missing_evidence():
    result = replace(make_result(), evidence=[])
    explanation = explain_recommendation(result)
    assert "The retriever did not provide detailed scoring evidence." in explanation.explanation
