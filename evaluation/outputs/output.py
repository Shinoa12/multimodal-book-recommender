from collections.abc import Mapping, Sequence
from os import PathLike
from typing import Any

import pandas as pd


PathLikeString = str | PathLike[str]


def export_query_results_to_parquet(
    query_results: Sequence[Mapping[str, Any]],
    output_path: PathLikeString,
) -> None:
    """Save query-level evaluation results to a parquet file.

    Args:
        query_results: Results produced for each evaluated query.
        output_path: Destination parquet file path.
    """
    pd.DataFrame.from_records(query_results).to_parquet(output_path, index=False)


def export_final_metrics_to_csv(
    final_metrics: Mapping[str, Any],
    output_path: PathLikeString,
) -> None:
    """Save final experiment metrics to a CSV file.

    Args:
        final_metrics: Final metrics produced by the experiment aggregator.
        output_path: Destination CSV file path.
    """
    pd.DataFrame([final_metrics]).to_csv(output_path, index=False)


__all__ = [
    "export_final_metrics_to_csv",
    "export_query_results_to_parquet",
]
