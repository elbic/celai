# Vector store abstract class

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, field
import numpy as np
from cel.rag.text2vec.utils import Embedding
from loguru import logger as log
from functools import lru_cache
import time


@dataclass
class VectorRegister:
    """A vector register with metadata and text content."""
    id: str
    text: str
    vector: List[float]
    metadata: Dict[str, Any] = None
    score: float = 0.0
    timestamp: float = field(default_factory=time.time)

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if not isinstance(self.vector, list):
            self.vector = list(self.vector)

    def __str__(self):
        return f"{self.text}"


class VectorStore(ABC):
    """Base class for vector stores. A vector store is a class that stores and retrieves by similarity and id.
    For simplicity, the vector store will define the embedding model to be used.
    We strongly recommend implement your own vector store flavor to suit your needs, combining the vector store
    with a database or a cache system and a embedding model of your choice.
    """
    
    def __init__(self, cache_size: int = 1000):
        self.cache_size = cache_size
        self._cache = {}
        self._last_cleanup = time.time()
        self._cleanup_interval = 3600  # 1 hour

    @abstractmethod
    def get_vector(self, id: str) -> VectorRegister:
        """Get the vector representation of an id"""
        pass
    
    @abstractmethod
    def get_similar(self, vector: Embedding, top_k: int) -> list[VectorRegister]:
        """Get the most similar vectors to the given vector"""
        pass
    
    @abstractmethod
    def search(self, 
               query: str, 
               top_k: int = 1,
               metadata_filter: Dict[str, Any] = None,
               min_score: float = 0.0) -> List[VectorRegister]:
        """Search for similar vectors with optional metadata filtering."""
        pass

    @abstractmethod
    def upsert(self, id: str, vector: Embedding, text: str, metadata: dict):
        """Upsert a vector to the store"""
        pass
    
    @abstractmethod
    def upsert_text(self, id: str, text: str, metadata: Dict[str, Any] = None) -> None:
        """Upsert a text document with its vector representation and metadata."""
        pass
    
    @abstractmethod
    def delete(self, id: str) -> bool:
        """Delete a document by ID."""
        pass

    def _cleanup_cache(self):
        """Clean up old cache entries."""
        current_time = time.time()
        if current_time - self._last_cleanup > self._cleanup_interval:
            # Remove entries older than 24 hours
            cutoff_time = current_time - 86400
            self._cache = {k: v for k, v in self._cache.items() 
                         if v.timestamp > cutoff_time}
            self._last_cleanup = current_time

    @lru_cache(maxsize=1000)
    def _compute_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def _filter_by_metadata(self, 
                          results: List[VectorRegister],
                          metadata_filter: Dict[str, Any]) -> List[VectorRegister]:
        """Filter results based on metadata criteria."""
        if not metadata_filter:
            return results

        filtered_results = []
        for result in results:
            matches = True
            for key, value in metadata_filter.items():
                if key not in result.metadata or result.metadata[key] != value:
                    matches = False
                    break
            if matches:
                filtered_results.append(result)

        return filtered_results

    def _sort_by_score(self, results: List[VectorRegister]) -> List[VectorRegister]:
        """Sort results by score in descending order."""
        return sorted(results, key=lambda x: x.score, reverse=True)

    def _apply_score_threshold(self, 
                             results: List[VectorRegister],
                             min_score: float) -> List[VectorRegister]:
        """Filter results by minimum score threshold."""
        return [r for r in results if r.score >= min_score]

    def batch_upsert(self, 
                    documents: List[Dict[str, Any]]) -> None:
        """Batch upsert multiple documents."""
        for doc in documents:
            self.upsert_text(
                id=doc['id'],
                text=doc['text'],
                metadata=doc.get('metadata')
            )

    def get(self, id: str) -> Optional[VectorRegister]:
        """Get a document by ID."""
        return self._cache.get(id)

    def clear(self) -> None:
        """Clear all documents from the store."""
        self._cache.clear()
        self._compute_similarity.cache_clear()