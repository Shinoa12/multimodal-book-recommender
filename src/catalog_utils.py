import pandas as pd

def load_catalog(catalog_path):
    return pd.read_parquet(catalog_path)