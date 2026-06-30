from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

import numpy as np
from numpy.typing import NDArray

from evaluation.protocol.evaluator import QueryEvaluator


Book = TypeVar("Book")
Tag = TypeVar("Tag")


@dataclass(frozen=True)
class ExperimentRunner(Generic[Book, Tag]):
    """Run query-level evaluation over every book in a catalog.

    Args:
        catalog: Complete catalog of books to evaluate. It can be a sequence of
            book objects or a pandas DataFrame.
        recommender: Function that receives a query book and a temporary
            catalog without that query book, and returns ordered
            recommendations.
        get_book_tags: Function that returns the tags associated with a book.
        get_recommendation_embeddings: Function that receives the ordered
            recommendations and returns their embeddings with shape (k, d).
    """

    catalog: Any
    recommender: Callable[[Book, Any], Iterable[Book]]
    get_book_tags: Callable[[Book], Iterable[Tag]]
    get_recommendation_embeddings: Callable[
        [Sequence[Book]],
        NDArray[np.floating],
    ]

    def run(self) -> list[dict[str, Any]]:
        """Evaluate recommendations for every book in the catalog.

        Returns:
            A list of dictionaries, one per query book, containing the metrics
            produced by QueryEvaluator.
        """
        results: list[dict[str, Any]] = []

        for query_index in range(len(self.catalog)):
            query_book = self._get_query_book(query_index)
            candidate_catalog = self._catalog_without_query(query_index)
            raw_recommendations = self.recommender(query_book, candidate_catalog)
            recommendations = self._normalize_recommendations(raw_recommendations)
            recommendation_embeddings = self.get_recommendation_embeddings(
                recommendations
            )

            query_result = QueryEvaluator(
                query_book=query_book,
                recommendations=recommendations,
                get_book_tags=self.get_book_tags,
                recommendation_embeddings=recommendation_embeddings,
            ).evaluate()

            results.append(query_result)

        return results

    def _get_query_book(self, query_index: int) -> Book:
        """Return the query book at the given position."""
        if hasattr(self.catalog, "iloc"):
            return self.catalog.iloc[query_index]

        return self.catalog[query_index]

    def _catalog_without_query(self, query_index: int) -> Any:
        """Return a copy of the catalog excluding the query book by position."""
        if hasattr(self.catalog, "drop") and hasattr(self.catalog, "index"):
            return self.catalog.drop(self.catalog.index[query_index])

        return [
            book
            for index, book in enumerate(self.catalog)
            if index != query_index
        ]

    def _normalize_recommendations(self, recommendations: Any) -> list[Book]:
        """Convert recommender output into an ordered list of book objects."""
        if hasattr(recommendations, "iterrows"):
            return [recommendation for _, recommendation in recommendations.iterrows()]

        return list(recommendations)
