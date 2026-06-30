# Multimodal Book Recommender

A research project implementing a content-based multimodal book recommendation system for scenarios where the query consists only of a book cover.

The proposed approach retrieves visually similar books using OpenCLIP, builds a Textual Anchor from their textual embeddings generated with Sentence-BERT, and re-ranks the candidates through a Late Fusion strategy.

## Features

- Book cover retrieval using OpenCLIP
- Semantic textual representations with Sentence-BERT
- Textual Anchor construction
- Late Fusion ranking
- Offline evaluation using NDCG@10, ILD and Catalog Coverage

## Tech Stack

- Python
- PyTorch
- OpenCLIP
- Sentence-Transformers
- Pandas
- NumPy
- Scikit-learn

## Dataset

- GoodBooks-10K
- Google Books API (book cover retrieval)
