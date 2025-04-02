from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, field
from loguru import logger as log
from cel.model.common import ContextMessage
from cel.rag.stores.vector_store import VectorRegister
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

@dataclass
class SearchResult:
    """Enhanced search result with multiple similarity scores."""
    vector_register: VectorRegister
    vector_similarity: float
    text_similarity: float = 0.0
    combined_score: float = 0.0
    metadata_score: float = 0.0

class RAGRetriever(ABC):
    """Enhanced RAG retriever with hybrid search capabilities."""
    
    def __init__(self,
                 vector_weight: float = 0.7,
                 text_weight: float = 0.3,
                 metadata_weight: float = 0.1,
                 min_score: float = 0.0,
                 rerank_top_k: int = 10):
        self.vector_weight = vector_weight
        self.text_weight = text_weight
        self.metadata_weight = metadata_weight
        self.min_score = min_score
        self.rerank_top_k = rerank_top_k
        self.tfidf = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 2),
            max_features=10000
        )

    @abstractmethod
    def search(self,
               query: str,
               top_k: int = 1,
               history: List[ContextMessage] = None,
               state: Dict[str, Any] = None,
               metadata_filter: Dict[str, Any] = None) -> List[VectorRegister]:
        """Search for relevant documents using hybrid search."""
        pass

    def _compute_text_similarity(self, query: str, documents: List[str]) -> List[float]:
        """Compute text similarity using TF-IDF and cosine similarity."""
        try:
            # Fit and transform the documents
            tfidf_matrix = self.tfidf.fit_transform([query] + documents)
            
            # Compute cosine similarity between query and documents
            similarities = cosine_similarity(
                tfidf_matrix[0:1],
                tfidf_matrix[1:]
            )[0]
            
            return similarities.tolist()
        except Exception as e:
            log.error(f"Error computing text similarity: {str(e)}")
            return [0.0] * len(documents)

    def _compute_metadata_score(self, 
                              query: str,
                              metadata: Dict[str, Any]) -> float:
        """Compute metadata-based relevance score."""
        if not metadata:
            return 0.0
        
        score = 0.0
        query_terms = set(query.lower().split())
        
        # Check title and description
        for field in ['title', 'description']:
            if field in metadata and metadata[field]:
                field_terms = set(metadata[field].lower().split())
                overlap = len(query_terms & field_terms) / len(query_terms)
                score += overlap * 0.5
        
        # Check tags
        if 'tags' in metadata and metadata['tags']:
            tag_terms = set(' '.join(metadata['tags']).lower().split())
            overlap = len(query_terms & tag_terms) / len(query_terms)
            score += overlap * 0.3
        
        # Check custom metadata
        if 'custom_metadata' in metadata:
            custom_terms = set(' '.join(str(v) for v in metadata['custom_metadata'].values()).lower().split())
            overlap = len(query_terms & custom_terms) / len(query_terms)
            score += overlap * 0.2
        
        return min(score, 1.0)

    def _rerank_results(self,
                       query: str,
                       results: List[VectorRegister],
                       top_k: int) -> List[VectorRegister]:
        """Rerank results using hybrid scoring."""
        if not results:
            return []
        
        # Compute text similarity scores
        text_scores = self._compute_text_similarity(
            query,
            [r.text for r in results]
        )
        
        # Create search results with all scores
        search_results = []
        for i, (result, text_score) in enumerate(zip(results, text_scores)):
            metadata_score = self._compute_metadata_score(query, result.metadata)
            
            # Combine scores
            combined_score = (
                self.vector_weight * result.score +
                self.text_weight * text_score +
                self.metadata_weight * metadata_score
            )
            
            search_results.append(SearchResult(
                vector_register=result,
                vector_similarity=result.score,
                text_similarity=text_score,
                metadata_score=metadata_score,
                combined_score=combined_score
            ))
        
        # Sort by combined score
        search_results.sort(key=lambda x: x.combined_score, reverse=True)
        
        # Filter by minimum score
        search_results = [
            r for r in search_results 
            if r.combined_score >= self.min_score
        ]
        
        # Return top k results
        return [r.vector_register for r in search_results[:top_k]]

    def _preprocess_query(self, query: str) -> str:
        """Preprocess query for better matching."""
        # Remove special characters and extra whitespace
        query = re.sub(r'[^\w\s]', ' ', query)
        query = ' '.join(query.split())
        return query.lower()

    def _extract_keywords(self, query: str) -> List[str]:
        """Extract important keywords from query."""
        # Remove common words and short terms
        words = query.split()
        keywords = [
            word for word in words 
            if len(word) > 2 and word not in self.tfidf.stop_words
        ]
        return keywords

    def _compute_query_embedding(self, query: str) -> List[float]:
        """Compute query embedding for vector similarity."""
        # This should be implemented by the concrete class
        raise NotImplementedError()