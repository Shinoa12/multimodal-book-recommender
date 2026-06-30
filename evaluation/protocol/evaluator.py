from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

import numpy as np
from numpy.typing import NDArray

from evaluation.metrics.diversity import compute_ild
from evaluation.metrics.ranking import compute_ndcg
from evaluation.metrics.retrieval import compute_jaccard


Book = TypeVar("Book")
Tag = TypeVar("Tag")


@dataclass(frozen=True)
class QueryEvaluator(Generic[Book, Tag]):
    """Evaluate recommendations generated for a single query book.

    Args:
        query_book: Book used as the recommendation query.
        recommendations: Ordered recommendations returned by the recommender.
        get_book_tags: Function that returns the tags associated with a book.
        recommendation_embeddings: Embeddings for the ordered recommendations,
            with shape (k, d).
        ideal_relevances: Optional relevance scores for all valid candidate
            books in the ideal ranking, excluding the query book.
    """

    query_book: Book
    recommendations: Sequence[Book]
    get_book_tags: Callable[[Book], Iterable[Tag]]
    recommendation_embeddings: NDArray[np.floating]
    ideal_relevances: Sequence[float] | None = None

    def evaluate(self) -> dict[str, Any]:
        """Compute all metrics for this query without saving results to disk.

        Returns:
            Dictionary containing the ordered recommendations, per-item Jaccard
            scores, ILD, and the number of recommendations evaluated. NDCG is
            included only when ideal_relevances is provided.

        Raises:
            ValueError: If the number of recommendation embeddings does not
                match the number of recommendations.
        """
        if self.recommendation_embeddings.shape[0] != len(self.recommendations):
            raise ValueError(
                "recommendation_embeddings must have one row per recommendation."
            )

        query_tags = tuple(self.get_book_tags(self.query_book))
        jaccard_scores = [
            compute_jaccard(query_tags, self.get_book_tags(recommendation))
            for recommendation in self.recommendations
        ]

        k = len(self.recommendations)

        result = {
            "query_book": self.query_book,
            "recommendations": list(self.recommendations),
            "jaccard_scores": jaccard_scores,
            "ild": compute_ild(self.recommendation_embeddings),
            "num_recommendations": k,
        }

        if self.ideal_relevances is not None:
            result["ndcg"] = compute_ndcg(
                recommended_relevances=jaccard_scores,
                ideal_relevances=self.ideal_relevances,
                k=k,
            )

        return result
