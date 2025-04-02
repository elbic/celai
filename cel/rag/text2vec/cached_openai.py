from abc import ABC
import os
import time
from typing import cast, List, Optional, Dict, Any
from diskcache import Cache
from cel.cache import get_cache
from .cache.base_cache import BaseCache
from .cache.disk_cache import DiskCache
from .utils import Embedding, Text2VectorProvider
from loguru import logger as log

try:
    from openai import OpenAI
except ImportError:
    raise ValueError(
        "The openai python package is not installed. Please install it with `pip install openai`"
    )

class CachedOpenAIEmbedding(Text2VectorProvider):
    """Cached OpenAI embedding provider with support for multiple models and advanced caching."""
    
    def __init__(self, 
                 api_key: str = None, 
                 model: str = "text-embedding-3-small",
                 cache_backend: BaseCache = None, 
                 max_retries: int = 5,
                 CACHE_EXPIRE: int = 43200000,
                 batch_size: int = 100,
                 timeout: int = 30):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.cache_backend = cache_backend or DiskCache(cache_dir='/tmp/diskcache')
        self.max_retries = max_retries
        self.cache_expire = CACHE_EXPIRE
        self.cache_tag = f'openai_embedding_{model}'
        self.batch_size = batch_size
        self.timeout = timeout

        if self.api_key is not None:
            OpenAI.api_key = api_key
        elif OpenAI.api_key is None:
            raise ValueError(
                "Please provide an OpenAI API key. You can get one at https://platform.openai.com/account/api-keys"
            )

        # Initialize the client with retry configuration
        self.client = OpenAI(
            max_retries=self.max_retries,
            timeout=self.timeout
        )

    def text2vec(self, text: str) -> Embedding:
        """Convert a single text to vector with caching."""
        return self._cached_text2vec(text, self.model, self.max_retries)
    
    def texts2vec(self, texts: List[str]) -> List[Embedding]:
        """Convert multiple texts to vectors with batching and caching."""
        results = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_results = self._cached_texts2vec(batch, self.model, self.max_retries)
            results.extend(batch_results)
        return results

    @property
    def _cached_text2vec(self):
        """Cached single text to vector conversion."""
        return self.cache_backend.memoize(
            typed=True, 
            expire=self.cache_expire, 
            tag=self.cache_tag
        )(self._text2vec)

    @property
    def _cached_texts2vec(self):
        """Cached batch text to vector conversion."""
        return self.cache_backend.memoize(
            typed=True, 
            expire=self.cache_expire, 
            tag=self.cache_tag
        )(self._texts2vec)

    def _text2vec(self, text: str, model: str, max_retries: int) -> List[float]:
        """Convert a single text to vector using OpenAI API."""
        try:
            # Clean and normalize text
            text = self._preprocess_text(text)
            
            response = self.client.embeddings.create(
                input=[text],
                model=model
            )
            return response.data[0].embedding
        except Exception as e:
            log.error(f"Error in text2vec: {str(e)}")
            raise

    def _texts2vec(self, texts: List[str], model: str, max_retries: int) -> List[List[float]]:
        """Convert multiple texts to vectors using OpenAI API."""
        try:
            # Clean and normalize texts
            texts = [self._preprocess_text(text) for text in texts]
            
            response = self.client.embeddings.create(
                input=texts,
                model=model
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            log.error(f"Error in texts2vec: {str(e)}")
            raise

    def _preprocess_text(self, text: str) -> str:
        """Preprocess text before embedding."""
        # Remove extra whitespace
        text = " ".join(text.split())
        # Truncate if too long (OpenAI has a limit)
        if len(text) > 8000:
            text = text[:8000]
        return text

    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self.cache_backend.clear(tag=self.cache_tag)

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return self.cache_backend.get_stats(tag=self.cache_tag)

def openai_cached_text2vec(text: str, model: str, max_retries: int = 3) -> list[float]:
    client = OpenAI(max_retries=max_retries)

    # replace newlines, which can negatively affect performance.
    text = text.replace("\n", " ")

    response = client.embeddings.create(input=[text], model=model)
    embeddings = response.data[0].embedding
    return embeddings