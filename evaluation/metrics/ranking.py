import math
from collections.abc import Sequence


def compute_ndcg(
    recommended_relevances: Sequence[float],
    ideal_relevances: Sequence[float],
    k: int,
) -> float:
    """Compute Normalized Discounted Cumulative Gain at k.

    Args:
        recommended_relevances: Relevance values ordered by the recommender
            ranking. Values are expected to be continuous scores in [0, 1].
        ideal_relevances: Relevance values for all candidate items that may
            appear in the ideal ranking. Values are expected to be continuous
            scores in [0, 1].
        k: Ranking cutoff. Only the first k relevance scores are considered.

    Returns:
        NDCG@k as a float between 0 and 1. Returns 0.0 when no ideal gain is
        available for the requested cutoff.

    Raises:
        ValueError: If k is negative or any relevance score is outside [0, 1].
        TypeError: If k is not an integer.
    """
    if not isinstance(k, int):
        raise TypeError("k must be an integer.")
    if k < 0:
        raise ValueError("k must be greater than or equal to 0.")
    if k == 0 or not recommended_relevances:
        return 0.0

    recommended_scores = [float(score) for score in recommended_relevances]
    ideal_scores = [float(score) for score in ideal_relevances]

    for score in recommended_scores + ideal_scores:
        if score < 0.0 or score > 1.0:
            raise ValueError("relevance scores must contain values between 0 and 1.")

    cutoff_scores = recommended_scores[:k]
    dcg = _compute_dcg(cutoff_scores)

    ideal_cutoff_scores = sorted(ideal_scores, reverse=True)[:k]
    idcg = _compute_dcg(ideal_cutoff_scores)
    if idcg == 0.0:
        return 0.0

    return dcg / idcg


def _compute_dcg(relevance_scores: Sequence[float]) -> float:
    """Compute Discounted Cumulative Gain for ordered relevance scores."""
    return sum(
        relevance / math.log2(position + 2)
        for position, relevance in enumerate(relevance_scores)
    )
