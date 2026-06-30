from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


DEFAULT_BASELINE_PATH = Path("evaluation/outputs/final_metrics_baseline.parquet")
DEFAULT_MULTIMODAL_PATH = Path("evaluation/outputs/final_metrics_multimodal.parquet")
FALLBACK_BASELINE_PATH = Path("evaluation/outputs/query_results_baseline.parquet")
FALLBACK_MULTIMODAL_PATH = Path("evaluation/outputs/query_results_multimodal.parquet")
DEFAULT_COMPARISON_OUTPUT_PATH = Path(
    "evaluation/outputs/ndcg_pairwise_comparison.parquet"
)
DEFAULT_TEST_OUTPUT_PATH = Path("evaluation/outputs/statistical_test_ndcg.csv")
NDCG_COLUMN_CANDIDATES = ("ndcg", "NDCG", "ndcg@10", "NDCG@10")
ILD_COLUMN_CANDIDATES = ("ild", "ILD")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the statistical analysis."""
    parser = argparse.ArgumentParser(
        description="Compare baseline visual vs multimodal NDCG@10 per query."
    )
    parser.add_argument("--baseline-results", default=str(DEFAULT_BASELINE_PATH))
    parser.add_argument("--multimodal-results", default=str(DEFAULT_MULTIMODAL_PATH))
    parser.add_argument("--query-id-column", default="query_book_id")
    parser.add_argument("--ndcg-column", default=None)
    parser.add_argument("--baseline-ndcg-column", default=None)
    parser.add_argument("--multimodal-ndcg-column", default=None)
    parser.add_argument("--ild-column", default=None)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument(
        "--comparison-output",
        default=str(DEFAULT_COMPARISON_OUTPUT_PATH),
    )
    parser.add_argument("--test-output", default=str(DEFAULT_TEST_OUTPUT_PATH))

    return parser.parse_args()


def resolve_input_path(path: Path, fallback_path: Path) -> Path:
    """Return the requested path, or a known per-query fallback if available."""
    if path.exists():
        return path
    if fallback_path.exists():
        print(f"Input not found at {path}. Using fallback: {fallback_path}")
        return fallback_path

    return path


def load_results(path: Path) -> pd.DataFrame:
    """Load per-query results from a parquet file."""
    return pd.read_parquet(path)


def detect_metric_column(
    df: pd.DataFrame,
    configured_column: str | None,
    candidates: tuple[str, ...],
    metric_name: str,
) -> str:
    """Detect a metric column or validate the configured column."""
    if configured_column is not None:
        if configured_column not in df.columns:
            raise ValueError(f"Configured {metric_name} column not found: {configured_column}")
        return configured_column

    for candidate in candidates:
        if candidate in df.columns:
            return candidate

    raise ValueError(
        f"Could not detect {metric_name} column. "
        f"Tried: {list(candidates)}. Use the corresponding CLI argument."
    )


def optional_metric_column(
    df: pd.DataFrame,
    configured_column: str | None,
    candidates: tuple[str, ...],
) -> str | None:
    """Detect an optional metric column if present."""
    if configured_column is not None:
        if configured_column not in df.columns:
            raise ValueError(f"Configured optional metric column not found: {configured_column}")
        return configured_column

    for candidate in candidates:
        if candidate in df.columns:
            return candidate

    return None


def validate_required_columns(
    df: pd.DataFrame,
    query_id_column: str,
    ndcg_column: str,
    label: str,
) -> None:
    """Validate required columns and uniqueness constraints."""
    missing_columns = [
        column
        for column in (query_id_column, ndcg_column)
        if column not in df.columns
    ]
    if missing_columns:
        raise ValueError(f"{label} results missing columns: {missing_columns}")

    if df[query_id_column].duplicated().any():
        raise ValueError(f"{label} results contain duplicated {query_id_column} values.")


def build_comparison_dataframe(
    baseline_df: pd.DataFrame,
    multimodal_df: pd.DataFrame,
    query_id_column: str,
    baseline_ndcg_column: str,
    multimodal_ndcg_column: str,
    baseline_ild_column: str | None,
    multimodal_ild_column: str | None,
) -> pd.DataFrame:
    """Build a paired comparison DataFrame joined by query ID."""
    baseline_columns = [query_id_column, baseline_ndcg_column]
    multimodal_columns = [query_id_column, multimodal_ndcg_column]

    if baseline_ild_column is not None:
        baseline_columns.append(baseline_ild_column)
    if multimodal_ild_column is not None:
        multimodal_columns.append(multimodal_ild_column)

    baseline = baseline_df[baseline_columns].rename(
        columns={
            baseline_ndcg_column: "baseline_ndcg",
            **({baseline_ild_column: "baseline_ild"} if baseline_ild_column else {}),
        }
    )
    multimodal = multimodal_df[multimodal_columns].rename(
        columns={
            multimodal_ndcg_column: "multimodal_ndcg",
            **(
                {multimodal_ild_column: "multimodal_ild"}
                if multimodal_ild_column
                else {}
            ),
        }
    )

    comparison = baseline.merge(multimodal, on=query_id_column, how="inner")
    validate_comparison_dataframe(
        comparison,
        baseline_count=len(baseline_df),
        multimodal_count=len(multimodal_df),
    )
    comparison["ndcg_difference"] = (
        comparison["multimodal_ndcg"] - comparison["baseline_ndcg"]
    )

    ordered_columns = [
        query_id_column,
        "baseline_ndcg",
        "multimodal_ndcg",
        "ndcg_difference",
    ]
    optional_columns = [
        column
        for column in ("baseline_ild", "multimodal_ild")
        if column in comparison.columns
    ]
    return comparison[ordered_columns + optional_columns]


def validate_comparison_dataframe(
    comparison: pd.DataFrame,
    baseline_count: int,
    multimodal_count: int,
) -> None:
    """Validate paired comparison shape and required metric values."""
    if len(comparison) != baseline_count or len(comparison) != multimodal_count:
        raise ValueError(
            "The paired merge does not contain the same number of queries as both inputs."
        )

    if comparison[["baseline_ndcg", "multimodal_ndcg"]].isna().any().any():
        raise ValueError("NDCG columns must not contain null values.")

    if len(comparison["baseline_ndcg"]) != len(comparison["multimodal_ndcg"]):
        raise ValueError("Baseline and multimodal NDCG series must have the same length.")


def compute_descriptive_statistics(comparison: pd.DataFrame) -> dict[str, Any]:
    """Compute descriptive paired-difference statistics."""
    differences = comparison["ndcg_difference"]
    return {
        "mean_difference": float(differences.mean()),
        "median_difference": float(differences.median()),
        "multimodal_wins": int((differences > 0).sum()),
        "baseline_wins": int((differences < 0).sum()),
        "ties": int((differences == 0).sum()),
        "n_queries": int(len(differences)),
    }


def run_wilcoxon_test(
    differences: pd.Series,
    alpha: float,
) -> dict[str, Any]:
    """Run one-sided Wilcoxon signed-rank test for multimodal improvement."""
    if np.all(differences.to_numpy() == 0):
        return {
            "statistic": np.nan,
            "p_value": np.nan,
            "is_significant": False,
            "note": "All paired differences are zero; Wilcoxon test is not applicable.",
        }

    result = wilcoxon(
        differences,
        alternative="greater",
        zero_method="wilcox",
    )
    p_value = float(result.pvalue)
    return {
        "statistic": float(result.statistic),
        "p_value": p_value,
        "is_significant": bool(p_value < alpha),
        "note": "",
    }


def build_test_result(
    comparison: pd.DataFrame,
    alpha: float,
) -> dict[str, Any]:
    """Build the statistical test result row."""
    descriptive_stats = compute_descriptive_statistics(comparison)
    test_stats = run_wilcoxon_test(comparison["ndcg_difference"], alpha)

    return {
        "test_name": "Wilcoxon signed-rank test",
        "metric": "NDCG@10",
        "alternative": "greater",
        "alpha": alpha,
        "statistic": test_stats["statistic"],
        "p_value": test_stats["p_value"],
        **descriptive_stats,
        "is_significant": test_stats["is_significant"],
        "note": test_stats["note"],
    }


def save_outputs(
    comparison: pd.DataFrame,
    test_result: dict[str, Any],
    comparison_output: Path,
    test_output: Path,
) -> None:
    """Save comparison and statistical test outputs."""
    comparison_output.parent.mkdir(parents=True, exist_ok=True)
    test_output.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_parquet(comparison_output, index=False)
    pd.DataFrame([test_result]).to_csv(test_output, index=False)


def print_summary(test_result: dict[str, Any], comparison_output: Path, test_output: Path) -> None:
    """Print a human-readable statistical analysis summary."""
    print("NDCG@10 paired statistical comparison")
    print(f"Test: {test_result['test_name']} ({test_result['alternative']})")
    print(f"n_queries: {test_result['n_queries']}")
    print(f"mean_difference: {test_result['mean_difference']}")
    print(f"median_difference: {test_result['median_difference']}")
    print(f"multimodal_wins: {test_result['multimodal_wins']}")
    print(f"baseline_wins: {test_result['baseline_wins']}")
    print(f"ties: {test_result['ties']}")
    print(f"statistic: {test_result['statistic']}")
    print(f"p_value: {test_result['p_value']}")
    print(f"is_significant(alpha={test_result['alpha']}): {test_result['is_significant']}")
    if test_result["note"]:
        print(f"note: {test_result['note']}")
    print(f"comparison_output: {comparison_output}")
    print(f"test_output: {test_output}")


def main() -> None:
    """Run the complementary paired NDCG@10 statistical analysis."""
    args = parse_args()

    baseline_path = resolve_input_path(Path(args.baseline_results), FALLBACK_BASELINE_PATH)
    multimodal_path = resolve_input_path(
        Path(args.multimodal_results),
        FALLBACK_MULTIMODAL_PATH,
    )

    baseline_df = load_results(baseline_path)
    multimodal_df = load_results(multimodal_path)

    baseline_ndcg_column = detect_metric_column(
        baseline_df,
        args.baseline_ndcg_column or args.ndcg_column,
        NDCG_COLUMN_CANDIDATES,
        "baseline NDCG",
    )
    multimodal_ndcg_column = detect_metric_column(
        multimodal_df,
        args.multimodal_ndcg_column or args.ndcg_column,
        NDCG_COLUMN_CANDIDATES,
        "multimodal NDCG",
    )
    baseline_ild_column = optional_metric_column(
        baseline_df,
        args.ild_column,
        ILD_COLUMN_CANDIDATES,
    )
    multimodal_ild_column = optional_metric_column(
        multimodal_df,
        args.ild_column,
        ILD_COLUMN_CANDIDATES,
    )

    validate_required_columns(
        baseline_df,
        args.query_id_column,
        baseline_ndcg_column,
        "Baseline",
    )
    validate_required_columns(
        multimodal_df,
        args.query_id_column,
        multimodal_ndcg_column,
        "Multimodal",
    )

    comparison = build_comparison_dataframe(
        baseline_df=baseline_df,
        multimodal_df=multimodal_df,
        query_id_column=args.query_id_column,
        baseline_ndcg_column=baseline_ndcg_column,
        multimodal_ndcg_column=multimodal_ndcg_column,
        baseline_ild_column=baseline_ild_column,
        multimodal_ild_column=multimodal_ild_column,
    )
    test_result = build_test_result(comparison, args.alpha)

    comparison_output = Path(args.comparison_output)
    test_output = Path(args.test_output)
    save_outputs(comparison, test_result, comparison_output, test_output)
    print_summary(test_result, comparison_output, test_output)


if __name__ == "__main__":
    main()
