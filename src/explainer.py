from dataclasses import dataclass
from typing import List

from .retriever import RetrievalResult


@dataclass(frozen=True)
class RecommendationExplanation:
    """
    Human-readable explanation for one retrieved recommendation.
    """

    track_name: str
    artist_name: str
    genre: str
    score: float
    explanation: str
    evidence: List[str]


def explain_recommendation(result: RetrievalResult) -> RecommendationExplanation:
    """
    Generate a grounded explanation for one retrieval result.

    The explanation only uses retrieved track fields and retriever evidence.
    """
    track = result.track
    evidence_summary = _summarize_evidence(result.evidence)

    explanation = (
        f"{track.track_name} by {track.artist_name} was selected by the recommender. "
        f"Here is how it compares with your request. {evidence_summary}"
    )

    return RecommendationExplanation(
        track_name=track.track_name,
        artist_name=track.artist_name,
        genre=track.genre,
        score=result.score,
        explanation=explanation,
        evidence=result.evidence,
    )


def explain_recommendations(
    results: List[RetrievalResult],
) -> List[RecommendationExplanation]:
    """
    Generate grounded explanations for a ranked list of retrieval results.
    """
    return [explain_recommendation(result) for result in results]


def _summarize_evidence(evidence: List[str]) -> str:
    """
    Convert retriever evidence into one compact explanation sentence.
    """
    if not evidence:
        return "The retriever did not provide detailed scoring evidence."

    return " ".join(evidence)
