import marko
from marko.block import Document
from abc import ABC
from dataclasses import dataclass, field
import re
import yaml

@dataclass
class Block(ABC):
    type: str
    text: str
    index: int
    breadcrumbs: list[str]
    metadata: dict = field(default_factory=dict)

def extract_link_metadata(text: str) -> dict:
    """Extract metadata from link elements."""
    metadata = {}
    link_match = re.match(r'\[([^\]]+)\]\(([^)]+)\)', text)
    if link_match:
        metadata["link_text"] = link_match.group(1)
        metadata["url"] = link_match.group(2)
    return metadata

def extract_frontmatter(md: str) -> tuple[dict, str]:
    """Extract YAML frontmatter from markdown content."""
    frontmatter = {}
    content = md
    
    if md.startswith('---'):
        parts = md.split('---', 2)
        if len(parts) >= 3:
            try:
                frontmatter = yaml.safe_load(parts[1])
                content = parts[2]
            except yaml.YAMLError:
                pass
    
    return frontmatter, content

def build_breadcrumbs(doc: Document, current_index: int) -> list[str]:
    breadcrumbs = []
    current_level = float('inf')
    # iterate from current_index to the beginning of the document
    for block in reversed(doc.children[:current_index]):
        if doc.children.index(block) >= current_index:
            break
        if block.get_type().lower() == 'heading':
            level = block.level
            if level < current_level:
                current_level = level
                breadcrumbs.append(block.children[0].children)
    breadcrumbs.reverse()
    return breadcrumbs

def get_text_from_element(element) -> str:
    """Safely extract text from a markdown element."""
    if not element or not hasattr(element, 'children'):
        return ''
    
    if not element.children:
        return ''
        
    first_child = element.children[0] if element.children else None
    if not first_child:
        return ''
        
    return getattr(first_child, 'children', '')

def parse_markdown(md: str, split_table_rows: bool = False) -> list[Block]:
    """Parse markdown content into blocks with enhanced metadata.
    
    Args:
        md: Markdown content to parse
        split_table_rows: Whether to split table rows into separate blocks
        
    Returns:
        List of Block objects with text content and metadata
    """
    from marko.md_renderer import MarkdownRenderer
    
    # Extract frontmatter first
    frontmatter, content = extract_frontmatter(md)
    
    mdr = marko.Markdown(extensions=['gfm'], renderer=MarkdownRenderer)
    doc = mdr.parse(content)
    
    blocks = []
    
    for child in doc.children:
        type = child.get_type().lower()
        metadata = {"type": type}
        text = ''
        
        # Add frontmatter to all blocks
        if frontmatter:
            metadata["frontmatter"] = frontmatter
            
        if type == "paragraph":
            text = get_text_from_element(child)
            # Check for special inline elements
            if isinstance(text, str):
                link_meta = extract_link_metadata(text)
                if link_meta:
                    metadata.update(link_meta)
                    
        elif type == "heading":
            text = get_text_from_element(child)
            metadata["level"] = getattr(child, 'level', 1)
            
        elif type == "table":
            if not hasattr(child, 'children') or not child.children:
                continue
                
            header = child.children[0].children if child.children and hasattr(child.children[0], 'children') else []
            rows = child.children[1:] if len(child.children) > 1 else []
            
            # Extract table metadata
            metadata["columns"] = len(header)
            metadata["rows"] = len(rows)
            metadata["headers"] = [get_text_from_element(cell) for cell in header]
            
            if split_table_rows:
                header_text = ' | '.join([get_text_from_element(cell) for cell in header])
                header_separator = ' | '.join(['---' for _ in header])
                
                for row in rows:
                    if not hasattr(row, 'children'):
                        continue
                    row_cells = [get_text_from_element(cell) for cell in row.children]
                    row_text = ' | '.join(row_cells)
                    table_text = f"{header_text}\n{header_separator}\n{row_text}"
                    
                    child_index = doc.children.index(child)
                    bc = build_breadcrumbs(doc, child_index)
                    blocks.append(Block(
                        type='table_row',
                        text=table_text,
                        index=child_index,
                        breadcrumbs=bc,
                        metadata=metadata
                    ))
                continue
            else:
                text = mdr.render(child)
        else:
            text = get_text_from_element(child)
            
        child_index = doc.children.index(child)
        bc = build_breadcrumbs(doc, child_index)
        
        blocks.append(Block(
            type=type,
            text=text,
            index=child_index,
            breadcrumbs=bc,
            metadata=metadata
        ))

    return blocks