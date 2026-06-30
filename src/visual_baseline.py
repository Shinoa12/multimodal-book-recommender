from config import CATALOG_PATH, get_test_image_path
from catalog_utils import load_catalog
from embedding_utils import generate_query_visual_embedding, load_openclip_model
from similarity_utils import compute_cosine_similarity
from display_utils import display_recommendations_text


import pandas as pd
import numpy as np
import torch

def recommend_visual(query_image_path, catalog_df, k=10):
    model, preprocess = load_openclip_model()

    query_embedding = generate_query_visual_embedding(query_image_path, model, preprocess)
    if query_embedding is None:
        return pd.DataFrame()
    
    catalog_embeddings = catalog_embeddings = torch.tensor(
        np.stack(catalog_df["normalized_image_embeddings"].values),
        dtype=torch.float32
    )
    similarities = compute_cosine_similarity(query_embedding, catalog_embeddings)
    if similarities.size == 0:
        return pd.DataFrame()
    
    catalog_df['visual_score'] = similarities
    recommended_books = catalog_df.sort_values(by='visual_score', ascending=False).head(k)

    return recommended_books

if __name__ == "__main__":
    catalog_df = load_catalog(CATALOG_PATH)
    test_image_path = get_test_image_path(1)
    visual_recommendations = recommend_visual(test_image_path, catalog_df.copy(), k=5)

    display_recommendations_text(visual_recommendations)
