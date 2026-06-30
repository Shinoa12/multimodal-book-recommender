from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
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
from evaluation.outputs.output import (
    export_final_metrics_to_csv,
    export_query_results_to_parquet,
)
from evaluation.outputs.reports import ResultsAggregator
from evaluation.protocol.experiment import ExperimentRunner


ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from catalog_utils import load_catalog  # noqa: E402
from config import CATALOG_PATH  # noqa: E402
from multimodal_anchor import recommend_multimodal  # noqa: E402
from visual_baseline import recommend_visual  # noqa: E402


@dataclass(frozen=True)
class VectorizedEvaluationData:
    """Catalog data and embedding matrices for vectorized evaluation."""

    catalog: pd.DataFrame
    book_ids: list[Any]
    tags: list[Any]
    visual_embeddings: NDArray[np.floating]
    text_embeddings: NDArray[np.floating] | None


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the evaluation run."""
    parser = argparse.ArgumentParser(
        description="Run the book recommendation evaluation protocol."
    )
    parser.add_argument("--catalog-path", default=CATALOG_PATH)
    parser.add_argument(
        "--recommender",
        choices=("visual", "multimodal"),
        default="visual",
    )
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--top-m", type=int, default=50)
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument(
        "--vectorized",
        action="store_true",
        help="Run final evaluation from precomputed embeddings.",
    )
    parser.add_argument("--image-path-column", default="image_path")
    parser.add_argument("--tags-column", default="tags")
    parser.add_argument(
        "--embedding-column",
        default="normalized_image_embeddings",
    )
    parser.add_argument(
        "--visual-embedding-column",
        default="normalized_image_embeddings",
    )
    parser.add_argument(
        "--text-embedding-column",
        default="normalized_text_embeddings",
    )
    parser.add_argument("--id-column", default="book_id")
    parser.add_argument(
        "--query-results-path",
        default="evaluation/outputs/query_results.parquet",
    )
    parser.add_argument(
        "--final-metrics-path",
        default="evaluation/outputs/final_metrics.csv",
    )

    return parser.parse_args()


def build_recommender(args: argparse.Namespace) -> Callable[[Any, Any], Any]:
    """Build the selected recommender adapter."""
    if args.recommender == "visual":
        return lambda query_book, catalog: recommend_visual(
            get_value(query_book, args.image_path_column),
            catalog.copy(),
            k=args.k,
        )

    return lambda query_book, catalog: recommend_multimodal(
        get_value(query_book, args.image_path_column),
        catalog.copy(),
        k=args.k,
        top_m=args.top_m,
        top_n=args.top_n,
        alpha=args.alpha,
    )


def get_value(book: Any, column: str) -> Any:
    """Return a column value from a catalog row-like object."""
    if hasattr(book, "__getitem__"):
        return book[column]

    return getattr(book, column)


def get_embeddings(recommendations: list[Any], embedding_column: str) -> np.ndarray:
    """Return recommendation embeddings as a 2D numpy array."""
    if not recommendations:
        return np.empty((0, 0))

    return np.stack(
        [
            get_value(recommendation, embedding_column)
            for recommendation in recommendations
        ]
    )


def validate_vectorized_args(args: argparse.Namespace) -> None:
    """Validate arguments required by vectorized evaluation."""
    if args.k <= 0:
        raise ValueError("k must be greater than 0.")
    if args.recommender == "multimodal" and args.top_m <= 0:
        raise ValueError("top-m must be greater than 0.")
    if args.recommender == "multimodal" and args.top_n <= 0:
        raise ValueError("top-n must be greater than 0.")


def validate_columns(catalog: pd.DataFrame, columns: list[str]) -> None:
    """Validate that required columns exist in the catalog."""
    missing_columns = [column for column in columns if column not in catalog.columns]
    if missing_columns:
        raise ValueError(f"Missing required catalog columns: {missing_columns}")


def stack_embeddings(embedding_series: pd.Series) -> NDArray[np.floating]:
    """Stack an embedding column into a 2D float32 numpy array."""
    return np.stack(embedding_series.to_numpy()).astype(np.float32)


def load_vectorized_data(args: argparse.Namespace) -> VectorizedEvaluationData:
    """Load catalog, IDs, tags, and precomputed embedding matrices once."""
    validate_vectorized_args(args)
    catalog = load_catalog(args.catalog_path).reset_index(drop=True)
    required_columns = [
        args.id_column,
        args.tags_column,
        args.visual_embedding_column,
    ]
    if args.recommender == "multimodal":
        required_columns.append(args.text_embedding_column)

    validate_columns(
        catalog,
        required_columns,
    )

    return VectorizedEvaluationData(
        catalog=catalog,
        book_ids=catalog[args.id_column].tolist(),
        tags=catalog[args.tags_column].tolist(),
        visual_embeddings=stack_embeddings(catalog[args.visual_embedding_column]),
        text_embeddings=(
            stack_embeddings(catalog[args.text_embedding_column])
            if args.recommender == "multimodal"
            else None
        ),
    )


def get_device() -> torch.device:
    """Return CUDA device when available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def normalize_rows(tensor: torch.Tensor) -> torch.Tensor:
    """L2-normalize a 2D tensor by rows."""
    return torch.nn.functional.normalize(tensor, p=2, dim=1)


def build_leave_one_out_mask(num_books: int, device: torch.device) -> torch.Tensor:
    """Build a diagonal mask for Leave-One-Out evaluation."""
    return torch.eye(num_books, dtype=torch.bool, device=device)


def compute_visual_similarity(
    visual_embeddings: NDArray[np.floating],
    device: torch.device,
) -> torch.Tensor:
    """Compute full visual similarity matrix and apply Leave-One-Out masking."""
    visual_tensor = normalize_rows(
        torch.as_tensor(visual_embeddings, dtype=torch.float32, device=device)
    )
    visual_similarity = visual_tensor @ visual_tensor.T
    visual_similarity = visual_similarity.masked_fill(
        build_leave_one_out_mask(len(visual_embeddings), device),
        -torch.inf,
    )
    return visual_similarity


def compute_top_m_candidates(
    visual_similarity: torch.Tensor,
    top_m: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return Top-M visual scores and indices for every query."""
    effective_top_m = min(top_m, visual_similarity.shape[1] - 1)
    if effective_top_m <= 0:
        raise ValueError("catalog must contain at least one candidate per query.")

    return torch.topk(visual_similarity, k=effective_top_m, dim=1)


def compute_textual_scores(
    text_embeddings: NDArray[np.floating],
    candidate_indices: torch.Tensor,
    top_n: int,
    device: torch.device,
) -> torch.Tensor:
    """Compute anchor-to-candidate textual scores for every query."""
    text_tensor = normalize_rows(
        torch.as_tensor(text_embeddings, dtype=torch.float32, device=device)
    )
    candidate_embeddings = text_tensor[candidate_indices]
    effective_top_n = min(top_n, candidate_embeddings.shape[1])
    anchors = candidate_embeddings[:, :effective_top_n, :].mean(dim=1)
    anchors = normalize_rows(anchors)
    return torch.sum(anchors[:, None, :] * candidate_embeddings, dim=2)


def rank_multimodal_top_k(
    visual_scores: torch.Tensor,
    textual_scores: torch.Tensor,
    candidate_indices: torch.Tensor,
    alpha: float,
    k: int,
) -> NDArray[np.integer]:
    """Rank final Top-K recommendations from cached visual and textual scores."""
    multimodal_scores = alpha * visual_scores + (1.0 - alpha) * textual_scores
    effective_k = min(k, multimodal_scores.shape[1])
    _, top_k_positions = torch.topk(multimodal_scores, k=effective_k, dim=1)
    top_k_indices = torch.gather(candidate_indices, dim=1, index=top_k_positions)
    return top_k_indices.cpu().numpy()


def rank_visual_top_k(
    visual_similarity: torch.Tensor,
    k: int,
) -> NDArray[np.integer]:
    """Rank final Top-K recommendations directly from visual similarities."""
    effective_k = min(k, visual_similarity.shape[1] - 1)
    if effective_k <= 0:
        raise ValueError("catalog must contain at least one candidate per query.")

    _, top_k_indices = torch.topk(visual_similarity, k=effective_k, dim=1)
    return top_k_indices.cpu().numpy()


def compute_ideal_relevances_for_query(
    query_index: int,
    tags: list[Any],
) -> list[float]:
    """Compute ideal relevance candidates for a query over the full catalog."""
    query_tags = tags[query_index]
    return [
        compute_jaccard(query_tags, candidate_tags)
        for candidate_index, candidate_tags in enumerate(tags)
        if candidate_index != query_index
    ]


def build_query_results(
    data: VectorizedEvaluationData,
    top_k_indices: NDArray[np.integer],
    k: int,
) -> list[dict[str, Any]]:
    """Build per-query evaluation results from final recommendation indices."""
    query_results: list[dict[str, Any]] = []

    for query_index, recommendation_indices in enumerate(
        tqdm(top_k_indices, desc="Evaluating queries")
    ):
        query_tags = data.tags[query_index]
        recommended_relevances = [
            compute_jaccard(query_tags, data.tags[int(recommendation_index)])
            for recommendation_index in recommendation_indices
        ]
        recommendation_embeddings = data.visual_embeddings[recommendation_indices]
        recommendation_ids = [
            data.book_ids[int(recommendation_index)]
            for recommendation_index in recommendation_indices
        ]

        query_results.append(
            {
                "query_book_id": data.book_ids[query_index],
                "recommendations": recommendation_ids,
                "jaccard_scores": recommended_relevances,
                "ndcg": compute_ndcg(
                    recommended_relevances=recommended_relevances,
                    ideal_relevances=compute_ideal_relevances_for_query(
                        query_index,
                        data.tags,
                    ),
                    k=k,
                ),
                "ild": compute_ild(recommendation_embeddings),
                "num_recommendations": len(recommendation_ids),
            }
        )

    return query_results


def aggregate_vectorized_results(
    query_results: list[dict[str, Any]],
    catalog_size: int,
) -> dict[str, float]:
    """Aggregate final metrics for vectorized evaluation."""
    ndcg_values = [float(result["ndcg"]) for result in query_results]
    ild_values = [float(result["ild"]) for result in query_results]
    all_recommendations = [
        recommendation
        for result in query_results
        for recommendation in result["recommendations"]
    ]

    return {
        "mean_ndcg": float(np.mean(ndcg_values)) if ndcg_values else 0.0,
        "std_ndcg": float(np.std(ndcg_values)) if ndcg_values else 0.0,
        "mean_ild": float(np.mean(ild_values)) if ild_values else 0.0,
        "std_ild": float(np.std(ild_values)) if ild_values else 0.0,
        "catalog_coverage": compute_catalog_coverage(
            all_recommendations,
            catalog_size,
        ),
    }


def run_vectorized_evaluation(args: argparse.Namespace) -> None:
    """Run Leave-One-Out evaluation using precomputed embeddings."""
    device = get_device()
    print(f"Using device: {device}")

    data = load_vectorized_data(args)
    visual_similarity = compute_visual_similarity(
        data.visual_embeddings,
        device,
    )
    if args.recommender == "visual":
        top_k_indices = rank_visual_top_k(visual_similarity, args.k)
    else:
        if data.text_embeddings is None:
            raise ValueError("text embeddings are required for multimodal evaluation.")

        visual_scores, candidate_indices = compute_top_m_candidates(
            visual_similarity,
            args.top_m,
        )
        textual_scores = compute_textual_scores(
            data.text_embeddings,
            candidate_indices,
            args.top_n,
            device,
        )
        top_k_indices = rank_multimodal_top_k(
            visual_scores,
            textual_scores,
            candidate_indices,
            args.alpha,
            args.k,
        )

    query_results = build_query_results(data, top_k_indices, args.k)
    final_metrics = aggregate_vectorized_results(query_results, len(data.catalog))

    export_query_results_to_parquet(query_results, args.query_results_path)
    export_final_metrics_to_csv(final_metrics, args.final_metrics_path)


def main() -> None:
    """Run the full experimental evaluation workflow."""
    args = parse_args()

    if args.vectorized:
        run_vectorized_evaluation(args)
        return

    catalog = load_catalog(args.catalog_path)
    recommender = build_recommender(args)

    runner = ExperimentRunner(
        catalog=catalog,
        recommender=recommender,
        get_book_tags=lambda book: get_value(book, args.tags_column),
        get_recommendation_embeddings=lambda recommendations: get_embeddings(
            recommendations,
            args.embedding_column,
        ),
    )
    query_results = runner.run()

    final_metrics = ResultsAggregator(
        query_results=query_results,
        catalog_size=len(catalog),
        get_book_id=lambda book: get_value(book, args.id_column),
    ).aggregate()

    export_query_results_to_parquet(query_results, args.query_results_path)
    export_final_metrics_to_csv(final_metrics, args.final_metrics_path)


if __name__ == "__main__":
    main()
