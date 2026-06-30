from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from tqdm import tqdm

from evaluation.metrics.coverage import compute_catalog_coverage
from evaluation.metrics.diversity import compute_ild
from evaluation.metrics.ranking import compute_ndcg
from evaluation.metrics.retrieval import compute_jaccard


ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from catalog_utils import load_catalog  # noqa: E402
from config import CATALOG_PATH  # noqa: E402


ALPHA_VALUES = [0.3, 0.5, 0.7, 0.9]
TOP_N_VALUES = [3, 5, 7, 10, 13, 15]
DEFAULT_OUTPUT_PATH = "evaluation/outputs/hyperparameter_search.csv"


@dataclass(frozen=True)
class SearchData:
    """Preloaded data required by the hyperparameter search."""

    catalog: pd.DataFrame
    sample: pd.DataFrame
    sample_indices: NDArray[np.integer]
    book_ids: list[Any]
    tags: list[Any]
    visual_embeddings: NDArray[np.floating]
    text_embeddings: NDArray[np.floating]
    ideal_relevances: list[list[float]]


@dataclass(frozen=True)
class TopMCache:
    """Cached visual candidates and scores for all sampled queries."""

    candidate_indices: NDArray[np.integer]
    visual_scores: NDArray[np.floating]


@dataclass(frozen=True)
class TextualCache:
    """Cached textual scores for one top_n value."""

    top_n: int
    textual_scores: NDArray[np.floating]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for hyperparameter optimization."""
    parser = argparse.ArgumentParser(
        description="Optimize alpha and top_n using precomputed embeddings."
    )
    parser.add_argument("--catalog-path", default=CATALOG_PATH)
    parser.add_argument("--output-path", default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--sample-size", type=int, default=300)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--top-m", type=int, default=50)
    parser.add_argument("--tags-column", default="tags")
    parser.add_argument(
        "--visual-embedding-column",
        default="normalized_image_embeddings",
    )
    parser.add_argument(
        "--text-embedding-column",
        default="normalized_text_embeddings",
    )
    parser.add_argument("--id-column", default="book_id")

    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """Validate hyperparameter optimization arguments."""
    if args.sample_size <= 0:
        raise ValueError("sample-size must be greater than 0.")
    if args.k <= 0:
        raise ValueError("k must be greater than 0.")
    if args.top_m <= 0:
        raise ValueError("top-m must be greater than 0.")


def load_search_data(args: argparse.Namespace) -> SearchData:
    """Load catalog, embeddings, tags, IDs, and reproducible query sample."""
    validate_args(args)
    catalog = load_catalog(args.catalog_path).reset_index(drop=True)
    validate_columns(
        catalog,
        [
            args.id_column,
            args.tags_column,
            args.visual_embedding_column,
            args.text_embedding_column,
        ],
    )

    sample = sample_catalog(catalog, args.sample_size, args.random_state)
    sample_indices = sample.index.to_numpy()

    book_ids = catalog[args.id_column].tolist()
    tags = catalog[args.tags_column].tolist()

    return SearchData(
        catalog=catalog,
        sample=sample,
        sample_indices=sample_indices,
        book_ids=book_ids,
        tags=tags,
        visual_embeddings=stack_embeddings(catalog[args.visual_embedding_column]),
        text_embeddings=stack_embeddings(catalog[args.text_embedding_column]),
        ideal_relevances=compute_ideal_relevances(
            sample_indices,
            book_ids,
            tags,
        ),
    )


def validate_columns(catalog: pd.DataFrame, columns: list[str]) -> None:
    """Validate required catalog columns."""
    missing_columns = [column for column in columns if column not in catalog.columns]
    if missing_columns:
        raise ValueError(f"Missing required catalog columns: {missing_columns}")


def sample_catalog(
    catalog: pd.DataFrame,
    sample_size: int,
    random_state: int,
) -> pd.DataFrame:
    """Select a reproducible random catalog sample."""
    if len(catalog) <= sample_size:
        return catalog.sample(frac=1, random_state=random_state)

    return catalog.sample(n=sample_size, random_state=random_state)


def stack_embeddings(embedding_series: pd.Series) -> NDArray[np.floating]:
    """Stack a catalog embedding column into a 2D numpy array."""
    return np.stack(embedding_series.to_numpy()).astype(np.float32)


def compute_ideal_relevances(
    sample_indices: NDArray[np.integer],
    book_ids: list[Any],
    tags: list[Any],
) -> list[list[float]]:
    """Compute ideal relevance candidates for each sampled query."""
    ideal_relevances: list[list[float]] = []

    for query_catalog_index in tqdm(sample_indices, desc="Ideal relevances"):
        query_index = int(query_catalog_index)
        query_book_id = book_ids[query_index]
        query_tags = tags[query_index]

        query_ideal_relevances = [
            compute_jaccard(query_tags, candidate_tags)
            for candidate_book_id, candidate_tags in zip(book_ids, tags)
            if candidate_book_id != query_book_id
        ]
        ideal_relevances.append(query_ideal_relevances)

    return ideal_relevances


def get_device() -> torch.device:
    """Return CUDA device when available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def normalize_rows(tensor: torch.Tensor) -> torch.Tensor:
    """L2-normalize a 2D tensor by rows."""
    return torch.nn.functional.normalize(tensor, p=2, dim=1)


def compute_visual_similarity_matrix(
    sample_embeddings: NDArray[np.floating],
    catalog_embeddings: NDArray[np.floating],
    device: torch.device,
) -> NDArray[np.floating]:
    """Compute visual cosine similarities from sampled queries to catalog."""
    query_tensor = normalize_rows(
        torch.as_tensor(sample_embeddings, dtype=torch.float32, device=device)
    )
    catalog_tensor = normalize_rows(
        torch.as_tensor(catalog_embeddings, dtype=torch.float32, device=device)
    )

    similarities = query_tensor @ catalog_tensor.T
    return similarities.cpu().numpy()


def get_top_m_cache(
    visual_similarities: NDArray[np.floating],
    sample_indices: NDArray[np.integer],
    book_ids: list[Any],
    top_m: int,
) -> TopMCache:
    """Cache Top-M visual candidates for each sampled query."""
    similarities = visual_similarities.copy()
    for query_position, catalog_index in enumerate(sample_indices):
        query_book_id = book_ids[int(catalog_index)]
        same_book_mask = np.asarray(
            [book_id == query_book_id for book_id in book_ids],
            dtype=bool,
        )
        similarities[query_position, same_book_mask] = -np.inf

    top_m = min(top_m, similarities.shape[1] - 1)
    if top_m <= 0:
        raise ValueError("catalog must contain at least one candidate per query.")

    unordered_indices = np.argpartition(-similarities, kth=top_m - 1, axis=1)[
        :, :top_m
    ]
    unordered_scores = np.take_along_axis(similarities, unordered_indices, axis=1)
    order = np.argsort(-unordered_scores, axis=1)

    candidate_indices = np.take_along_axis(unordered_indices, order, axis=1)
    visual_scores = np.take_along_axis(unordered_scores, order, axis=1)

    return TopMCache(
        candidate_indices=candidate_indices,
        visual_scores=visual_scores.astype(np.float32),
    )


def compute_textual_scores_by_top_n(
    text_embeddings: NDArray[np.floating],
    top_m_cache: TopMCache,
    top_n_values: list[int],
    device: torch.device,
) -> dict[int, TextualCache]:
    """Compute textual candidate scores once for each top_n value."""
    text_tensor = normalize_rows(
        torch.as_tensor(text_embeddings, dtype=torch.float32, device=device)
    )
    caches: dict[int, TextualCache] = {}

    candidate_tensor = torch.as_tensor(
        top_m_cache.candidate_indices,
        dtype=torch.long,
        device=device,
    )
    candidate_embeddings = text_tensor[candidate_tensor]

    for top_n in tqdm(top_n_values, desc="Textual anchors"):
        effective_top_n = min(top_n, candidate_embeddings.shape[1])
        anchors = candidate_embeddings[:, :effective_top_n, :].mean(dim=1)
        anchors = normalize_rows(anchors)
        textual_scores = torch.sum(anchors[:, None, :] * candidate_embeddings, dim=2)
        caches[top_n] = TextualCache(
            top_n=top_n,
            textual_scores=textual_scores.cpu().numpy().astype(np.float32),
        )

    return caches


def evaluate_grid(
    search_data: SearchData,
    top_m_cache: TopMCache,
    textual_caches: dict[int, TextualCache],
    args: argparse.Namespace,
) -> pd.DataFrame:
    """Evaluate every alpha/top_n combination from cached scores."""
    rows: list[dict[str, float]] = []
    grid = [(alpha, top_n) for top_n in TOP_N_VALUES for alpha in ALPHA_VALUES]

    for alpha, top_n in tqdm(grid, desc="Grid search"):
        start_time = time.perf_counter()
        textual_scores = textual_caches[top_n].textual_scores
        multimodal_scores = (
            alpha * top_m_cache.visual_scores + (1.0 - alpha) * textual_scores
        )
        top_k_indices = rank_top_k(
            multimodal_scores,
            top_m_cache.candidate_indices,
            args.k,
        )
        metrics = evaluate_rankings(search_data, top_k_indices, args.k)

        rows.append(
            {
                "alpha": alpha,
                "top_n": top_n,
                "mean_ndcg": metrics["mean_ndcg"],
                "mean_ild": metrics["mean_ild"],
                "catalog_coverage": metrics["catalog_coverage"],
                "execution_time_seconds": time.perf_counter() - start_time,
            }
        )

    return pd.DataFrame(rows).sort_values(by="mean_ndcg", ascending=False)


def rank_top_k(
    scores: NDArray[np.floating],
    candidate_indices: NDArray[np.integer],
    k: int,
) -> NDArray[np.integer]:
    """Return catalog indices for the final Top-K recommendations."""
    effective_k = min(k, scores.shape[1])
    top_k_positions = np.argsort(-scores, axis=1)[:, :effective_k]
    return np.take_along_axis(candidate_indices, top_k_positions, axis=1)


def evaluate_rankings(
    search_data: SearchData,
    top_k_indices: NDArray[np.integer],
    k: int,
) -> dict[str, float]:
    """Compute NDCG, ILD, and catalog coverage for final rankings."""
    ndcg_values: list[float] = []
    ild_values: list[float] = []
    recommended_book_ids: list[Any] = []

    for query_position, recommendation_indices in enumerate(top_k_indices):
        query_catalog_index = int(search_data.sample_indices[query_position])
        query_tags = search_data.tags[query_catalog_index]
        relevance_scores = [
            compute_jaccard(query_tags, search_data.tags[int(recommendation_index)])
            for recommendation_index in recommendation_indices
        ]
        recommendation_embeddings = search_data.visual_embeddings[
            recommendation_indices
        ]

        ndcg_values.append(
            compute_ndcg(
                recommended_relevances=relevance_scores,
                ideal_relevances=search_data.ideal_relevances[query_position],
                k=k,
            )
        )
        ild_values.append(compute_ild(recommendation_embeddings))
        recommended_book_ids.extend(
            search_data.book_ids[int(recommendation_index)]
            for recommendation_index in recommendation_indices
        )

    return {
        "mean_ndcg": float(np.mean(ndcg_values)) if ndcg_values else 0.0,
        "mean_ild": float(np.mean(ild_values)) if ild_values else 0.0,
        "catalog_coverage": compute_catalog_coverage(
            recommended_book_ids,
            len(search_data.catalog),
        ),
    }


def print_best_combination(results_df: pd.DataFrame) -> None:
    """Print the best hyperparameter combination and its main metrics."""
    if results_df.empty:
        print("No hyperparameter results were generated.")
        return

    best_result = results_df.iloc[0]
    print("Mejor combinación encontrada:")
    print(f"alpha: {best_result['alpha']}")
    print(f"top_n: {int(best_result['top_n'])}")
    print(f"Mean NDCG: {best_result['mean_ndcg']}")
    print(f"Mean ILD: {best_result['mean_ild']}")
    print(f"Catalog Coverage: {best_result['catalog_coverage']}")


def main() -> None:
    """Run vectorized hyperparameter optimization for multimodal ranking."""
    args = parse_args()
    device = get_device()
    print(f"Using device: {device}")

    search_data = load_search_data(args)
    visual_similarities = compute_visual_similarity_matrix(
        search_data.visual_embeddings[search_data.sample_indices],
        search_data.visual_embeddings,
        device,
    )
    top_m_cache = get_top_m_cache(
        visual_similarities,
        search_data.sample_indices,
        search_data.book_ids,
        args.top_m,
    )
    textual_caches = compute_textual_scores_by_top_n(
        search_data.text_embeddings,
        top_m_cache,
        TOP_N_VALUES,
        device,
    )

    results_df = evaluate_grid(search_data, top_m_cache, textual_caches, args)

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)

    print_best_combination(results_df)


if __name__ == "__main__":
    main()
