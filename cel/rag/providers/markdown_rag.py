from pathlib import Path
import time
from halo import Halo
from loguru import logger as log
from cel.model.common import ContextMessage
from cel.rag.providers.rag_retriever import RAGRetriever, SearchResult
from cel.rag.slicers.base_slicer import Slice
from cel.rag.slicers.markdown.markdown import MarkdownSlicer
from cel.rag.stores.chroma.chroma_store import ChromaStore
from cel.rag.stores.vector_store import VectorRegister, VectorStore
from cel.rag.text2vec.cached_openai import CachedOpenAIEmbedding
from cel.rag.text2vec.utils import Text2VectorProvider
from typing import List, Dict, Any, Optional
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer


class MarkdownRAG(RAGRetriever):
    """Enhanced markdown RAG implementation with improved search capabilities."""
    
    def __init__(self,
                 name: str,
                 file_path: str | Path = None,
                 content: str | None = None,
                 encoding: str = None,
                 prefix: str = None,
                 split_table_rows: bool = False,
                 text2vec: Text2VectorProvider = None,
                 store: VectorStore = None,
                 collection: str = None,
                 metadata: dict = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.name = name
        self.file_path = file_path
        self.content = content
        self.prefix = prefix
        self.encoding = encoding
        self.split_table_rows = split_table_rows
        self.collection_name = collection or name
        self.text2vec = text2vec or CachedOpenAIEmbedding()
        self.store = store or ChromaStore(self.text2vec, collection_name=self.collection_name)
        self.metadata = metadata or {}
        self._is_loaded = False

    def load(self):
        """Load and process markdown content."""
        if self._is_loaded:
            return
            
        log.debug(f"Loading markdown content from {self.file_path}")
        spinner = Halo(text='Loading...', spinner='dots')
        spinner.start()
        
        try:
            spinner.text = 'Creating markdown slicer'
            slicer = MarkdownSlicer(
                name=self.name,
                content=self.content,
                file_path=self.file_path,
                prefix=self.prefix,
                split_table_rows=self.split_table_rows
            )
            
            spinner.text = 'Slicing markdown content...'
            log.debug('Slicing markdown content...')
            slices = slicer.slice()
            log.debug(f'Slices: {len(slices)}')
            spinner.succeed('Slicing complete')
            spinner.stop()

            spinner.start()
            spinner.text = 'Embedding markdown slices...'
            for i, slice in enumerate(slices, 1):
                spinner.text = f'Processing slice {i}/{len(slices)}'
                self._embed_slice(slice)
                
            spinner.succeed('Processing complete')
            spinner.stop()
            
            self._is_loaded = True
            log.info(f"Successfully loaded {len(slices)} slices")
            
        except Exception as e:
            spinner.fail('Processing failed')
            log.error(f"Error loading markdown content: {str(e)}")
            raise

    def search(self, 
               query: str, 
               top_k: int = 1,
               history: List[ContextMessage] = None,
               state: Dict[str, Any] = None,
               metadata_filter: Dict[str, Any] = None) -> List[VectorRegister]:
        """Enhanced search with hybrid retrieval and reranking."""
        if not self._is_loaded:
            self.load()
            
        # Preprocess query
        query = self._preprocess_query(query)
        
        # Get initial vector search results
        initial_results = self.store.search(
            query,
            top_k=self.rerank_top_k,
            metadata_filter=metadata_filter
        )
        
        # Rerank results using hybrid scoring
        reranked_results = self._rerank_results(query, initial_results, top_k)
        
        return reranked_results

    def _embed_slice(self, slice: Slice):
        """Embed a slice with enhanced metadata."""
        try:
            # Generate embedding
            vector = self.text2vec.text2vec(slice.text)
            
            # Enhance metadata
            enhanced_metadata = {
                **self.metadata,
                **slice.metadata,
                'slicer': 'markdown',
                'timestamp': time.time(),
                'source': str(self.file_path) if self.file_path else 'inline',
                'slice_type': slice.metadata.get('content_type', 'text'),
                'word_count': len(slice.text.split()),
                'has_code': '```' in slice.text,
                'has_table': '|' in slice.text,
                'has_list': bool(re.match(r'^\s*[-*]', slice.text, re.MULTILINE)),
                'has_image': '![' in slice.text
            }
            
            # Filter out None values from metadata
            filtered_metadata = {k: v for k, v in enhanced_metadata.items() if v is not None}
            
            # Store the slice
            self.store.upsert_text(slice.id, slice.text, filtered_metadata)
            
        except Exception as e:
            log.error(f"Error embedding slice {slice.id}: {str(e)}")
            raise

    def _compute_query_embedding(self, query: str) -> List[float]:
        """Compute query embedding for vector similarity."""
        return self.text2vec.text2vec(query)

    def clear(self):
        """Clear all stored vectors and reset state."""
        self.store.clear()
        self._is_loaded = False
        self.tfidf = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 2),
            max_features=10000
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the RAG system."""
        return {
            'name': self.name,
            'collection': self.collection_name,
            'is_loaded': self._is_loaded,
            'file_path': str(self.file_path) if self.file_path else None,
            'metadata': self.metadata,
            'store_stats': self.store.get_stats() if hasattr(self.store, 'get_stats') else {}
        }
            
