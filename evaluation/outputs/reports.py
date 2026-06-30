from collections.abc import Callable, Sequence
from dataclasses import dataclass
from statistics import mean, pstdev
from typing import Any

from evaluation.metrics.coverage import compute_catalog_coverage


@dataclass(frozen=True)
class ResultsAggregator:
    """Aggregate query-level evaluation results into experiment metrics.

    Args:
        query_results: Results returned by ExperimentRunner.
        catalog_size: Total number of books in the catalog.
        get_book_id: Optional function used to extract stable identifiers from
            recommended books for catalog coverage.
    """

    query_results: Sequence[dict[str, Any]]
    catalog_size: int
    get_book_id: Callable[[Any], Any] | None = None

    def aggregate(self) -> dict[str, float]:
        """Compute final experiment-level metrics.

        Returns:
            Dictionary with mean and standard deviation for NDCG and ILD, plus
            catalog coverage.
        """
        ndcg_values = [
            float(result["ndcg"])
            for result in self.query_results
            if "ndcg" in result
        ]
        ild_values = [float(result["ild"]) for result in self.query_results]
        all_recommendations = [
            self._get_recommendation_id(recommendation)
            for result in self.query_results
            for recommendation in result["recommendations"]
        ]

        return {
            "mean_ndcg": self._mean_or_zero(ndcg_values),
            "std_ndcg": self._std_or_zero(ndcg_values),
            "mean_ild": self._mean_or_zero(ild_values),
            "std_ild": self._std_or_zero(ild_values),
            "catalog_coverage": compute_catalog_coverage(
                all_recommendations,
                self.catalog_size,
            ),
        }

    def _get_recommendation_id(self, recommendation: Any) -> Any:
        """Return the identifier used for catalog coverage."""
        if self.get_book_id is not None:
            return self.get_book_id(recommendation)

        return recommendation

    def _mean_or_zero(self, values: Sequence[float]) -> float:
        """Return the mean of values, or 0.0 for an empty sequence."""
        if not values:
            return 0.0

        return float(mean(values))

    def _std_or_zero(self, values: Sequence[float]) -> float:
        """Return population standard deviation, or 0.0 for an empty sequence."""
        if not values:
            return 0.0

        return float(pstdev(values))
