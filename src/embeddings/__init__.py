# src/embeddings/__init__.py

from src.embeddings.embedding_model import (
    EmbeddingModel,
    load_embedding_model,
)

__all__ = [
    "EmbeddingModel",
    "load_embedding_model",
]