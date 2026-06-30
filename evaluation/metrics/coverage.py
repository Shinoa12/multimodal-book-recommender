from collections.abc import Iterable
from typing import TypeVar


RecommendationId = TypeVar("RecommendationId")


def compute_catalog_coverage(
    all_recommendations: Iterable[RecommendationId],
    catalog_size: int,
) -> float:
    """Compute the fraction of the catalog recommended at least once.

    Args:
        all_recommendations: Collection of recommendation identifiers generated
            during evaluation.
        catalog_size: Total number of items in the catalog.

    Returns:
        Catalog coverage as a float between 0 and 1.

    Raises:
        TypeError: If catalog_size is not an integer.
        ValueError: If catalog_size is not positive.
    """
    if not isinstance(catalog_size, int):
        raise TypeError("catalog_size must be an integer.")
    if catalog_size <= 0:
        raise ValueError("catalog_size must be greater than 0.")

    unique_recommendations = set(all_recommendations)
    coverage = len(unique_recommendations) / catalog_size

    return min(coverage, 1.0)
