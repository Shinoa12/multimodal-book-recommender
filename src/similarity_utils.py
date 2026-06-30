import torch
import numpy as np

def compute_cosine_similarity(query_embedding, catalog_embeddings):
    return torch.matmul(query_embedding, catalog_embeddings.T).squeeze(0)


def get_visual_candidates(query_embedding, catalog_df, top_m):
    catalog_image_embeddings = torch.tensor(np.stack(catalog_df['normalized_image_embeddings'].values), dtype=torch.float32)
    visual_similarities = compute_cosine_similarity(query_embedding, catalog_image_embeddings)
    candidates_df = catalog_df.copy()
    candidates_df['visual_score'] = visual_similarities.cpu().numpy()
    candidates_df = candidates_df.sort_values(by='visual_score', ascending=False).head(top_m)
    return candidates_df

def compute_textual_scores(anchor_embedding, candidates_df):
    if candidates_df.empty or anchor_embedding.numel() == 0:
        print("No candidates or anchor embedding to compute textual scores.")
        candidates_df['text_score'] = 0.0
        return candidates_df
    
    candidate_text_embeddings = torch.tensor(np.stack(candidates_df['normalized_text_embeddings'].values), dtype=torch.float32)
    text_similarities = compute_cosine_similarity(anchor_embedding, candidate_text_embeddings)
    candidates_df['text_score'] = text_similarities.cpu().numpy()
    return candidates_df

def compute_multimodal_scores(candidates_df, alpha=0.7):
    if candidates_df.empty:
        print("No candidates to compute multimodal scores.")
        candidates_df['multimodal_score'] = 0.0
        return candidates_df
    
    candidates_df['multimodal_score'] = alpha * candidates_df['visual_score'] + (1 - alpha) * candidates_df['text_score']
    return candidates_df