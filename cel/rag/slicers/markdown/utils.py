import marko
from marko.block import Document
from abc import ABC
from dataclasses import dataclass, field
import re
from typing import List, Dict, Any

@dataclass
class Block(ABC):
    type: str
    text: str
    index: int
    breadcrumbs: list[str]
    # Enhanced metadata fields
    attributes: dict = field(default_factory=lambda: {
        'word_count': 0,
        'char_count': 0,
        'line_count': 0,
        'content_type': 'text',
        'has_code': False,
        'has_table': False,
        'has_list': False,
        'has_image': False,
        'has_links': False,
        'has_math': False,
        'has_tasks': False,
        'has_footnotes': False,
        'heading_level': 0,
        'section_depth': 0,
        'position': 0,
        'parent_type': None,
        'child_types': [],
        'language': None,
        'tags': [],
        'importance_score': 0.0
    })

def analyze_content(text: str) -> dict:
    """Analyze content for metadata attributes."""
    return {
        'word_count': len(text.split()),
        'char_count': len(text),
        'line_count': text.count('\n') + 1,
        'has_code': '```' in text,
        'has_table': '|' in text,
        'has_list': bool(re.search(r'^\s*[-*]\s', text, re.MULTILINE)),
        'has_image': '![' in text,
        'has_links': '[' in text and '](' in text,
        'has_math': bool(re.search(r'\$\$|\$', text)),
        'has_tasks': bool(re.search(r'- \[[ x]\]', text)),
        'has_footnotes': bool(re.search(r'\[\^.*?\]', text))
    }

def generate_tags(block: Block) -> list[str]:
    """Generate tags based on block content and metadata."""
    tags = []
    
    # Content type tags
    tags.append(f"type:{block.type}")
    tags.append(f"content:{block.attributes.get('content_type', 'text')}")
    
    # Feature tags
    if block.attributes.get('has_code', False):
        tags.append('has_code')
    if block.attributes.get('has_table', False):
        tags.append('has_table')
    if block.attributes.get('has_list', False):
        tags.append('has_list')
    if block.attributes.get('has_image', False):
        tags.append('has_image')
    if block.attributes.get('has_links', False):
        tags.append('has_links')
    if block.attributes.get('has_math', False):
        tags.append('has_math')
    if block.attributes.get('has_tasks', False):
        tags.append('has_tasks')
    if block.attributes.get('has_footnotes', False):
        tags.append('has_footnotes')
    
    # Language tag for code blocks
    if block.type == 'code' and block.attributes.get('language'):
        tags.append(f"lang:{block.attributes['language']}")
    
    # Section depth tag
    section_depth = block.attributes.get('section_depth', 0)
    if section_depth > 0:
        tags.append(f"depth:{section_depth}")
    
    # Importance tag
    importance = block.attributes.get('importance_score', 0.0)
    if importance >= 0.8:
        tags.append('importance:high')
    elif importance >= 0.6:
        tags.append('importance:medium')
    else:
        tags.append('importance:low')
    
    return tags

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

def parse_markdown(md: str, split_table_rows: bool = False) -> list[Block]:
    """Parse markdown content into structured blocks with enhanced metadata."""
    from marko.md_renderer import MarkdownRenderer
    mdr = marko.Markdown(extensions=['gfm'], renderer=MarkdownRenderer)
    doc = mdr.parse(md)
    
    blocks = []
    current_section = []
    section_stack = []  # Track section hierarchy
    last_block_type = None  # Track last block type for parent_type
    
    for child in doc.children:
        type = child.get_type().lower()
        child_index = doc.children.index(child)
        
        # Get current section depth and heading level
        current_depth = len(section_stack)
        current_heading_level = section_stack[-1]['level'] if section_stack else 0
        
        # Handle headings as section markers
        if type == 'heading':
            level = child.level
            
            # Update section stack
            while section_stack and section_stack[-1]['level'] >= level:
                section_stack.pop()
            
            section_stack.append({
                'level': level,
                'index': child_index
            })
            
            # Process previous section
            if current_section:
                blocks.extend(current_section)
                current_section = []
            
            # Create heading block with enhanced metadata
            heading_text = mdr.render(child).strip()
            heading_block = Block(
                type='heading',  # Ensure type is 'heading'
                text=heading_text,
                index=child_index,
                breadcrumbs=build_breadcrumbs(doc, child_index),
                attributes={
                    **analyze_content(heading_text),
                    'level': level,
                    'section_depth': len(section_stack),
                    'position': child_index,
                    'content_type': 'heading',
                    'importance_score': 1.0 / level,  # Higher level headings are more important
                    'heading_level': level,  # Add heading_level to attributes
                    'parent_type': last_block_type  # Set parent type from last block
                }
            )
            blocks.append(heading_block)  # Add heading block directly to blocks
            current_section = []  # Reset current section
            last_block_type = 'heading'  # Update last block type
            continue
            
        # Handle code blocks
        if type == 'code':
            code_text = child.children[0].children
            code_block = Block(
                type='code',
                text=code_text,
                index=child_index,
                breadcrumbs=build_breadcrumbs(doc, child_index),
                attributes={
                    **analyze_content(code_text),
                    'language': child.lang,
                    'content_type': 'code',
                    'position': child_index,
                    'parent_type': last_block_type,  # Set parent type from last block
                    'importance_score': 0.8,  # Code blocks are generally important
                    'section_depth': current_depth,
                    'heading_level': current_heading_level  # Set heading level from current section
                }
            )
            current_section.append(code_block)
            last_block_type = 'code'  # Update last block type
            continue
            
        # Handle tables
        if type == 'table':
            table_blocks = handle_table(child, doc, split_table_rows, last_block_type, mdr, current_depth)
            # Set heading level for table blocks
            for block in table_blocks:
                block.attributes['heading_level'] = current_heading_level
                block.attributes['parent_type'] = last_block_type  # Set parent type
            current_section.extend(table_blocks)
            last_block_type = 'table'  # Update last block type
            continue
            
        # Handle lists
        if type in ['list', 'listitem']:
            list_text = mdr.render(child)
            list_block = Block(
                type=type,
                text=list_text,
                index=child_index,
                breadcrumbs=build_breadcrumbs(doc, child_index),
                attributes={
                    **analyze_content(list_text),
                    'is_ordered': type == 'list' and child.ordered,
                    'content_type': 'list',
                    'position': child_index,
                    'parent_type': last_block_type,  # Set parent type from last block
                    'importance_score': 0.6,
                    'section_depth': current_depth,
                    'heading_level': current_heading_level  # Set heading level from current section
                }
            )
            current_section.append(list_block)
            last_block_type = type  # Update last block type
            continue
            
        # Handle other block types
        if type in ['paragraph', 'blockquote', 'html', 'hr']:
            block_text = child.children[0].children if child.children else ''
            block = Block(
                type=type,
                text=block_text,
                index=child_index,
                breadcrumbs=build_breadcrumbs(doc, child_index),
                attributes={
                    **analyze_content(block_text),
                    'content_type': type,
                    'position': child_index,
                    'parent_type': last_block_type,  # Set parent type from last block
                    'importance_score': 0.4,
                    'section_depth': current_depth,
                    'heading_level': current_heading_level  # Set heading level from current section
                }
            )
            current_section.append(block)
            last_block_type = type  # Update last block type
            continue
            
        # Handle inline elements
        if type in ['strong', 'em', 'codespan', 'br', 'del', 'link', 'image', 'text']:
            if current_section:
                current_section[-1].text += mdr.render(child)
                # Update parent block's metadata
                parent = current_section[-1]
                parent.attributes.update(analyze_content(parent.text))
            continue
    
    # Add remaining section
    if current_section:
        blocks.extend(current_section)
    
    # Update child types and final metadata
    for i, block in enumerate(blocks):
        if i > 0:
            block.attributes['parent_type'] = blocks[i-1].type
        if i < len(blocks) - 1:
            block.attributes['child_types'] = [blocks[i+1].type]
            
        # Add tags based on content analysis
        block.attributes['tags'] = generate_tags(block)
    
    return blocks

def handle_table(child, doc, split_table_rows: bool, parent_type: str, mdr, current_depth: int) -> list[Block]:
    """Handle table blocks with enhanced metadata."""
    header = child.children[0].children
    rows = child.children[1:]
    child_index = doc.children.index(child)
    bc = build_breadcrumbs(doc, child_index)
    
    if split_table_rows:
        blocks = []
        # Add header block
        header_text = ' | '.join([cell.children[0].children for cell in header])
        header_block = Block(
            type='table_header',
            text=header_text,
            index=child_index,
            breadcrumbs=bc,
            attributes={
                **analyze_content(header_text),
                'columns': len(header),
                'header_text': header_text,
                'content_type': 'table_header',
                'position': child_index,
                'parent_type': parent_type,
                'importance_score': 0.7,
                'section_depth': current_depth
            }
        )
        blocks.append(header_block)
        
        # Add row blocks
        for i, row in enumerate(rows):
            row_text = ' | '.join([cell.children[0].children for cell in row.children])
            row_block = Block(
                type='table_row',
                text=row_text,
                index=child_index,
                breadcrumbs=bc,
                attributes={
                    **analyze_content(row_text),
                    'row_index': i,
                    'header_text': header_text,
                    'content_type': 'table_row',
                    'position': child_index,
                    'parent_type': parent_type,
                    'importance_score': 0.5,
                    'section_depth': current_depth
                }
            )
            blocks.append(row_block)
        return blocks
    else:
        # Single table block
        table_text = mdr.render(child)
        return [Block(
            type='table',
            text=table_text,
            index=child_index,
            breadcrumbs=bc,
            attributes={
                **analyze_content(table_text),
                'rows': len(rows),
                'columns': len(header),
                'content_type': 'table',
                'position': child_index,
                'parent_type': parent_type,
                'importance_score': 0.6,
                'section_depth': current_depth
            }
        )]