from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass

from cel.rag.slicers.markdown.utils import parse_markdown, Block
from ..base_slicer import Slicer, Slice
from loguru import logger as log


@dataclass
class ChunkingConfig:
    """Configuration for markdown chunking."""
    max_chunk_size: int = 1000  # Maximum characters per chunk
    min_chunk_size: int = 100   # Minimum characters per chunk
    overlap_size: int = 100     # Number of characters to overlap between chunks
    preserve_sections: bool = True  # Whether to keep sections together
    include_code_blocks: bool = True  # Whether to include code blocks in chunks
    include_tables: bool = True  # Whether to include tables in chunks
    importance_threshold: float = 0.4  # Minimum importance score to include content


class MarkdownSlicer(Slicer):
    """Slice a markdown file with enhanced RAG capabilities.
    
    Features:
    - Smart chunking with configurable sizes and overlap
    - Enhanced metadata for better filtering and retrieval
    - Section-aware chunking to preserve document structure
    - Content type specific handling
    - Importance-based filtering
    
    Args:
        name: Name of the document
        file_path: Path to the file to load
        content: Direct markdown content
        encoding: File encoding to use
        prefix: Prefix to add to slice text
        split_table_rows: Whether to split table rows
        chunking_config: Configuration for chunking behavior
    """

    def __init__(
        self,
        name: str,
        file_path: str | Path = None,
        content: str | None = None,
        encoding: str = None,
        prefix: str = None,
        split_table_rows: bool = False,
        chunking_config: Optional[ChunkingConfig] = None
    ):
        self.name = name
        self.file_path = file_path
        self.content = content
        self.prefix = prefix
        self.encoding = encoding
        self.split_table_rows = split_table_rows
        self.chunking_config = chunking_config or ChunkingConfig()
        
    def __load_from_disk(self) -> str:
        if self.file_path is None:
            raise ValueError("No file path provided.")
        
        text = ""
        try:
            with open(self.file_path, encoding=self.encoding) as f:
                text = f.read()
        except UnicodeDecodeError:
            raise ValueError("Failed to load file with specified encoding.")
        except FileNotFoundError:
            raise ValueError(f"File not found: {self.file_path}")
        except Exception as e:
            raise ValueError(f"Failed to load file: {e}")    
        
        return text

    def __load(self) -> str:
        if self.content:
            return self.content
        return self.__load_from_disk()

    def _should_include_block(self, block: Block) -> bool:
        """Determine if a block should be included based on its attributes."""
        # Check importance threshold
        if block.attributes['importance_score'] < self.chunking_config.importance_threshold:
            return False
            
        # Check content type specific rules
        if block.type == 'code' and not self.chunking_config.include_code_blocks:
            return False
        if block.type == 'table' and not self.chunking_config.include_tables:
            return False
            
        return True

    def _create_slice(self, block: Block, index: int) -> Slice:
        """Create a slice from a block with enhanced metadata."""
        bc_str = " > ".join(block.breadcrumbs or [])
        
        # Create enhanced metadata
        metadata = {
            'type': block.type,
            'name': self.name,
            'word_count': block.attributes['word_count'],
            'char_count': block.attributes['char_count'],
            'line_count': block.attributes['line_count'],
            'content_type': block.attributes['content_type'],
            'has_code': block.attributes['has_code'],
            'has_table': block.attributes['has_table'],
            'has_list': block.attributes['has_list'],
            'has_image': block.attributes['has_image'],
            'has_links': block.attributes['has_links'],
            'has_math': block.attributes['has_math'],
            'has_tasks': block.attributes['has_tasks'],
            'has_footnotes': block.attributes['has_footnotes'],
            'heading_level': block.attributes['heading_level'],
            'section_depth': block.attributes['section_depth'],
            'position': block.attributes['position'],
            'parent_type': block.attributes['parent_type'],
            'language': block.attributes['language'],
            'importance_score': block.attributes['importance_score'],
            'tags': block.attributes['tags']
        }
        
        # Add content type specific formatting
        text = f"{bc_str}\n"
        if block.type == 'code':
            text += f"```{block.attributes['language'] or ''}\n{block.text}\n```"
        elif block.type == 'table':
            text += block.text
        else:
            text += block.text
            
        return Slice(
            id=f"{self.name}-{index}",
            text=text,
            metadata=metadata,
            source=self.name
        )

    def slice(self) -> list[Slice]:
        """Slice the markdown content with enhanced RAG capabilities."""
        slices = []
        text = self.__load()
        blocks = parse_markdown(text, split_table_rows=self.split_table_rows)
        
        current_chunk = []
        current_size = 0
        
        for i, block in enumerate(blocks):
            if not self._should_include_block(block):
                continue
                
            block_size = block.attributes['char_count']
            
            # Check if adding this block would exceed max chunk size
            if (current_size + block_size > self.chunking_config.max_chunk_size and 
                current_size >= self.chunking_config.min_chunk_size):
                # Create a slice from current chunk
                if current_chunk:
                    combined_text = "\n".join(b.text for b in current_chunk)
                    combined_metadata = self._merge_metadata(current_chunk)
                    slice = Slice(
                        id=f"{self.name}-{i}",
                        text=combined_text,
                        metadata=combined_metadata,
                        source=self.name
                    )
                    slices.append(slice)
                    current_chunk = []
                    current_size = 0
            
            current_chunk.append(block)
            current_size += block_size
            
        # Add remaining chunk
        if current_chunk:
            combined_text = "\n".join(b.text for b in current_chunk)
            combined_metadata = self._merge_metadata(current_chunk)
            slice = Slice(
                id=f"{self.name}-{len(slices)}",
                text=combined_text,
                metadata=combined_metadata,
                source=self.name
            )
            slices.append(slice)
            
        return slices
        
    def _merge_metadata(self, blocks: List[Block]) -> Dict[str, Any]:
        """Merge metadata from multiple blocks into a single metadata dict."""
        if not blocks:
            return {}
            
        # Safely get parent_type from first block, defaulting to None if not set
        parent_type = blocks[0].attributes.get('parent_type') if blocks else None
            
        merged = {
            'type': 'mixed',
            'name': self.name,
            'word_count': sum(b.attributes['word_count'] for b in blocks),
            'char_count': sum(b.attributes['char_count'] for b in blocks),
            'line_count': sum(b.attributes['line_count'] for b in blocks),
            'content_type': 'mixed',
            'has_code': any(b.attributes['has_code'] for b in blocks),
            'has_table': any(b.attributes['has_table'] for b in blocks),
            'has_list': any(b.attributes['has_list'] for b in blocks),
            'has_image': any(b.attributes['has_image'] for b in blocks),
            'has_links': any(b.attributes['has_links'] for b in blocks),
            'has_math': any(b.attributes['has_math'] for b in blocks),
            'has_tasks': any(b.attributes['has_tasks'] for b in blocks),
            'has_footnotes': any(b.attributes['has_footnotes'] for b in blocks),
            'heading_level': max(b.attributes['heading_level'] for b in blocks),
            'section_depth': max(b.attributes['section_depth'] for b in blocks),
            'position': blocks[0].attributes['position'],
            'parent_type': parent_type,  # Use safely retrieved parent_type
            'language': None,  # Mixed content
            'importance_score': max(b.attributes['importance_score'] for b in blocks),
            'tags': list(set(tag for b in blocks for tag in b.attributes['tags']))
        }
        
        return merged