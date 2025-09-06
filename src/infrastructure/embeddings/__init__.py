"""Embeddings infrastructure module."""

from .openai_embeddings import OpenAIEmbeddingClient
from .local_embeddings import LocalEmbeddingClient

__all__ = ['OpenAIEmbeddingClient', 'LocalEmbeddingClient']
