from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from tqdm import tqdm


DEFAULT_OUTPUT_CATALOG = "data/processed/catalog_with_image_paths.parquet"
DEFAULT_OUTPUT_DIR = "data/processed/covers"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Download book cover images and add local image paths."
    )
    parser.add_argument("--input-catalog", required=True)
    parser.add_argument("--output-catalog", default=DEFAULT_OUTPUT_CATALOG)
    parser.add_argument("--image-url-column", default="image_url")
    parser.add_argument("--book-id-column", default="book_id")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)

    return parser.parse_args()


def load_catalog(input_catalog: Path) -> pd.DataFrame:
    """Load a catalog from a parquet or CSV file."""
    suffix = input_catalog.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(input_catalog)
    if suffix == ".csv":
        return pd.read_csv(input_catalog)

    raise ValueError("input catalog must be a .parquet or .csv file.")


def download_cover(image_url: str, output_path: Path) -> bool:
    """Download a cover image unless it already exists."""
    if output_path.exists():
        return True

    response = requests.get(image_url, timeout=30)
    response.raise_for_status()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(response.content)
    return True


def get_cover_path(output_dir: Path, book_id: Any) -> Path:
    """Build the local cover path for a book identifier."""
    return output_dir / f"{book_id}.jpg"


def add_image_paths(
    catalog: pd.DataFrame,
    image_url_column: str,
    book_id_column: str,
    output_dir: Path,
) -> pd.DataFrame:
    """Download cover images and return a catalog with local image paths."""
    catalog = catalog.copy()
    image_paths: list[str | None] = []

    for _, row in tqdm(
        catalog.iterrows(),
        total=len(catalog),
        desc="Downloading covers",
    ):
        book_id = row[book_id_column]
        image_url = row[image_url_column]
        cover_path = get_cover_path(output_dir, book_id)

        try:
            download_cover(str(image_url), cover_path)
            image_paths.append(str(cover_path))
        except requests.RequestException as error:
            logging.error(
                "Failed to download cover for book_id=%s from %s: %s",
                book_id,
                image_url,
                error,
            )
            image_paths.append(None)

    catalog["image_path"] = image_paths
    return catalog


def validate_columns(
    catalog: pd.DataFrame,
    image_url_column: str,
    book_id_column: str,
) -> None:
    """Validate that required columns exist in the catalog."""
    missing_columns = [
        column
        for column in (image_url_column, book_id_column)
        if column not in catalog.columns
    ]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")


def main() -> None:
    """Download covers and save a catalog with local image paths."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()

    input_catalog = Path(args.input_catalog)
    output_catalog = Path(args.output_catalog)
    output_dir = Path(args.output_dir)

    catalog = load_catalog(input_catalog)
    validate_columns(catalog, args.image_url_column, args.book_id_column)

    catalog_with_paths = add_image_paths(
        catalog=catalog,
        image_url_column=args.image_url_column,
        book_id_column=args.book_id_column,
        output_dir=output_dir,
    )

    output_catalog.parent.mkdir(parents=True, exist_ok=True)
    catalog_with_paths.to_parquet(output_catalog, index=False)


if __name__ == "__main__":
    main()
