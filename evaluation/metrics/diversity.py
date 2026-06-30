import numpy as np
from numpy.typing import NDArray
from sklearn.metrics.pairwise import cosine_similarity


def compute_ild(embeddings: NDArray[np.floating]) -> float:
    """Compute Intra-List Diversity from recommendation embeddings.

    Intra-List Diversity is calculated as the average cosine distance between
    all unique pairs of recommendation embeddings.

    Args:
        embeddings: Array of shape (k, d), where k is the number of
            recommendations and d is the embedding dimension.

    Returns:
        Average pairwise cosine distance as a float. Returns 0.0 when fewer
        than two embeddings are provided.

    Raises:
        ValueError: If embeddings is not a 2D array.
    """
    if embeddings.ndim != 2:
        raise ValueError("embeddings must be a 2D array with shape (k, d).")

    num_embeddings = embeddings.shape[0]
    if num_embeddings < 2:
        return 0.0

    similarities = cosine_similarity(embeddings)
    distances = 1.0 - similarities

    upper_triangle_indices = np.triu_indices(num_embeddings, k=1)
    pairwise_distances = distances[upper_triangle_indices]

    return float(np.mean(pairwise_distances))
