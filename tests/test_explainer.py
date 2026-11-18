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


def test_explain_recommendation_uses_retrieved_fields_and_evidence():
    explanation = explain_recommendation(make_result())

    assert explanation.track_name == "Strong Match"
    assert explanation.artist_name == "Test Artist"
    assert explanation.genre == "pop"
    assert explanation.score == 6.50
    assert "Strong Match by Test Artist" in explanation.explanation
    assert "Here is how it compares with your request." in explanation.explanation
    assert "energy closeness: 0.82 vs target 0.80" in explanation.explanation
    assert "Derived vibe tags for readability: high_energy, positive, danceable" in explanation.explanation


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


def test_explain_recommendation_handles_missing_vibe_tags():
    result = make_result()
    track_without_tags = TrackRecord(
        track_name=result.track.track_name,
        artist_name=result.track.artist_name,
        album_name=result.track.album_name,
        genre=result.track.genre,
        vibe_tags=[],
        popularity=result.track.popularity,
        danceability=result.track.danceability,
        energy=result.track.energy,
        acousticness=result.track.acousticness,
        valence=result.track.valence,
        tempo=result.track.tempo,
    )

    explanation = explain_recommendation(
        RetrievalResult(
            track=track_without_tags,
            score=result.score,
            evidence=result.evidence,
        )
    )

    assert "Derived vibe tags for readability: none" in explanation.explanation
