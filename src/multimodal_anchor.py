from config import CATALOG_PATH, get_test_image_path
from catalog_utils import load_catalog
from embedding_utils import generate_query_visual_embedding, load_openclip_model
from similarity_utils import compute_cosine_similarity, get_visual_candidates, compute_textual_scores, compute_multimodal_scores
from display_utils import display_recommendations_text

import pandas as pd
import numpy as np
import torch

def build_textual_anchor(visual_candidates_df, top_n):
    
    top_n_candidates = visual_candidates_df.head(top_n)
    if top_n_candidates.empty:
        print("No candidates to build textual anchor.")
        return torch.empty(0)
    
    text_embeddings = torch.tensor(np.stack(top_n_candidates['normalized_text_embeddings'].values), dtype=torch.float32)
    anchor_embedding = torch.mean(text_embeddings, dim=0)
    anchor_embedding = anchor_embedding / anchor_embedding.norm(dim=-1, keepdim=True)
    return anchor_embedding

def recommend_multimodal(query_image_path, catalog_df, k=10, top_m=50, top_n=5, alpha=0.7):
    model, preprocess = load_openclip_model()
    
    query_visual_embedding = generate_query_visual_embedding(query_image_path, model, preprocess)
    if query_visual_embedding is None:
        return pd.DataFrame()
    
    visual_candidates_df = get_visual_candidates(query_visual_embedding, catalog_df, top_m)
    if visual_candidates_df.empty:
        print("No visual candidates found.")
        return pd.DataFrame()
    
    anchor_embedding = build_textual_anchor(visual_candidates_df, top_n)
    if anchor_embedding.numel() == 0:
        print("Error: Could not build textual anchor.")
        return pd.DataFrame()
    
    candidates_with_text_scores_df = compute_textual_scores(anchor_embedding, visual_candidates_df)
    
    
    final_candidates_df = compute_multimodal_scores(candidates_with_text_scores_df, alpha)
    
    return final_candidates_df.sort_values(by='multimodal_score', ascending=False).head(k)

if __name__ == "__main__":
    catalog_df = load_catalog(CATALOG_PATH)
    test_image_path = get_test_image_path(1)

    multimodal_recommendations = recommend_multimodal(test_image_path, catalog_df.copy(), k=5, top_m=50, top_n=5, alpha=0.7)

    display_recommendations_text(multimodal_recommendations)
