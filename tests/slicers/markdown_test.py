import pytest
from cel.rag.slicers.markdown import MarkdownSlicer
from cel.rag.slicers.markdown.utils import build_breadcrumbs, analyze_content, generate_tags, parse_markdown

# @pytest.fixture
# def lead():
#     lead = ChatLead('123', 'test', 'tenant1', 'assistant1')
#     return lead

# @pytest.fixture
# def redis_client():
#     redis_client = fakeredis.FakeRedis()
#     return redis_client

# @pytest.fixture
# def store(redis_client):
#     return RedisChatStateProvider(redis_client, 's')

def test_do():
    mds = MarkdownSlicer('test', './tests/slicers/sample.md')
    slices = mds.slice()
    print(slices)
    assert 1==1
    

def test_parse_markdown_basic():
    with open('./tests/slicers/sample.md', 'r') as f:
        md_content = f.read()
    
    blocks = parse_markdown(md_content)
    
    # Test basic structure
    assert len(blocks) > 0
    assert blocks[0].type == 'heading'
    assert blocks[0].text == '# Introduction'
    assert blocks[0].attributes['heading_level'] == 1

def test_parse_markdown_sections():
    with open('./tests/slicers/sample.md', 'r') as f:
        md_content = f.read()
    
    blocks = parse_markdown(md_content)
    
    # Test section hierarchy
    headings = [block for block in blocks if block.type == 'heading']
    assert len(headings) == 4  # Introduction, Section 1, Section 2, Demo Table
    
    # Test breadcrumbs
    section2_block = next(block for block in blocks if block.text == '## Section 2')
    assert len(section2_block.breadcrumbs) > 0
    assert 'Introduction' in section2_block.breadcrumbs

def test_parse_markdown_table():
    with open('./tests/slicers/sample.md', 'r') as f:
        md_content = f.read()
    
    # Test without splitting rows
    blocks = parse_markdown(md_content)
    table_blocks = [block for block in blocks if block.type == 'table']
    assert len(table_blocks) == 1
    assert table_blocks[0].attributes['rows'] == 2
    assert table_blocks[0].attributes['columns'] == 2
    
    # Test with splitting rows
    blocks_split = parse_markdown(md_content, split_table_rows=True)
    table_blocks = [block for block in blocks_split if block.type in ['table_header', 'table_row']]
    assert len(table_blocks) == 3  # 1 header + 2 rows

def test_parse_markdown_metadata():
    with open('./tests/slicers/sample.md', 'r') as f:
        md_content = f.read()
    
    blocks = parse_markdown(md_content)
    
    # Test metadata attributes
    for block in blocks:
        assert 'word_count' in block.attributes
        assert 'char_count' in block.attributes
        assert 'line_count' in block.attributes
        assert 'content_type' in block.attributes
        assert 'importance_score' in block.attributes
        assert 'section_depth' in block.attributes
        assert 'tags' in block.attributes

def test_parse_markdown_tags():
    with open('./tests/slicers/sample.md', 'r') as f:
        md_content = f.read()
    
    blocks = parse_markdown(md_content)
    
    # Test tag generation
    for block in blocks:
        assert len(block.attributes['tags']) > 0
        assert f"type:{block.type}" in block.attributes['tags']
        
        # Test content-specific tags
        if block.type == 'table':
            assert 'has_table' in block.attributes['tags']
        if block.type == 'heading':
            assert f"depth:{block.attributes['section_depth']}" in block.attributes['tags']
            assert f"importance:{'high' if block.attributes['importance_score'] >= 0.8 else 'medium' if block.attributes['importance_score'] >= 0.6 else 'low'}" in block.attributes['tags']

def test_parse_markdown_hierarchy():
    with open('./tests/slicers/sample.md', 'r') as f:
        md_content = f.read()
    
    blocks = parse_markdown(md_content)
    
    # Test parent-child relationships
    for i, block in enumerate(blocks):
        if i > 0:
            assert block.attributes['parent_type'] == blocks[i-1].type
        if i < len(blocks) - 1:
            assert blocks[i+1].type in block.attributes['child_types']

def test_parse_markdown_smoothy_content():
    with open('./tests/slicers/smoothy.md', 'r') as f:
        md_content = f.read()
    
    blocks = parse_markdown(md_content)
    
    # Test company introduction
    intro_block = next(block for block in blocks if block.type == 'paragraph' and 'SMOOTHY INC.' in block.text)
    assert intro_block is not None
    assert intro_block.attributes['word_count'] > 20
    assert intro_block.attributes['has_links'] == False

def test_parse_markdown_smoothy_lists():
    with open('./tests/slicers/smoothy.md', 'r') as f:
        md_content = f.read()
    
    blocks = parse_markdown(md_content)
    
    # Test product list
    product_section = next(block for block in blocks if block.text == '## Products')
    product_list = next(block for block in blocks[blocks.index(product_section):] 
                       if block.type == 'list')
    assert product_list is not None
    assert product_list.attributes['has_list'] == True
    assert 'Strawberry Banana' in product_list.text
    assert 'Mango Pineapple' in product_list.text
    assert 'Green Detox' in product_list.text
    
    # Test location list
    location_section = next(block for block in blocks if block.text == '## Locations')
    location_list = next(block for block in blocks[blocks.index(location_section):] 
                        if block.type == 'list')
    assert location_list is not None
    assert location_list.attributes['has_list'] == True
    assert 'New York' in location_list.text
    assert 'Los Angeles' in location_list.text
    assert 'Chicago' in location_list.text

def test_parse_markdown_smoothy_table():
    with open('./tests/slicers/smoothy.md', 'r') as f:
        md_content = f.read()
    
    blocks = parse_markdown(md_content)
    
    # Test table structure
    table_block = next(block for block in blocks if block.type == 'table')
    assert table_block is not None
    assert table_block.attributes['rows'] == 3  # header + 2 data rows
    assert table_block.attributes['columns'] == 3  # Product, Price, size
    
    # Test table content
    assert 'Strawberry Banana' in table_block.text
    assert '$5' in table_block.text
    assert '16 oz' in table_block.text
    
    # Test with split rows
    blocks_split = parse_markdown(md_content, split_table_rows=True)
    table_blocks = [block for block in blocks_split if block.type in ['table_header', 'table_row']]
    assert len(table_blocks) == 4  # 1 header + 3 data rows
    
    # Test header block
    header_block = next(block for block in table_blocks if block.type == 'table_header')
    assert 'Product' in header_block.text
    assert 'Price' in header_block.text
    assert 'size' in header_block.text

def test_parse_markdown_smoothy_section_hierarchy():
    with open('./tests/slicers/smoothy.md', 'r') as f:
        md_content = f.read()
    
    blocks = parse_markdown(md_content)
    
    # Test section structure
    headings = [block for block in blocks if block.type == 'heading']
    assert len(headings) == 4  # Introduction, Products, Locations, Demo Table
    
    # Test section hierarchy
    intro_heading = next(block for block in headings if block.text == '# Introduction')
    products_heading = next(block for block in headings if block.text == '## Products')
    locations_heading = next(block for block in headings if block.text == '## Locations')
    
    assert intro_heading.attributes['heading_level'] == 1
    assert products_heading.attributes['heading_level'] == 2
    assert locations_heading.attributes['heading_level'] == 2
    
    # Test breadcrumbs for nested sections
    assert len(products_heading.breadcrumbs) > 0
    assert 'Introduction' in products_heading.breadcrumbs

def test_parse_markdown_smoothy_metadata_analysis():
    with open('./tests/slicers/smoothy.md', 'r') as f:
        md_content = f.read()
    
    blocks = parse_markdown(md_content)
    
    # Test content analysis
    for block in blocks:
        # Test word count
        assert block.attributes['word_count'] > 0
        
        # Test character count
        assert block.attributes['char_count'] > 0
        
        # Test line count
        assert block.attributes['line_count'] > 0
        
        # Test content type
        assert block.attributes['content_type'] in ['text', 'heading', 'list', 'table', 'paragraph']
        
        # Test importance score
        assert 0 <= block.attributes['importance_score'] <= 1
        
        # Test section depth
        assert block.attributes['section_depth'] >= 0

def test_parse_markdown_smoothy_tag_generation():
    with open('./tests/slicers/smoothy.md', 'r') as f:
        md_content = f.read()
    
    blocks = parse_markdown(md_content)
    
    # Test tag generation for different block types
    for block in blocks:
        # Basic type tag
        assert f"type:{block.type}" in block.attributes['tags']
        
        # Content-specific tags
        if block.type == 'table':
            assert 'has_table' in block.attributes['tags']
        if block.type == 'list':
            assert 'has_list' in block.attributes['tags']
        if block.type == 'heading':
            assert f"depth:{block.attributes['section_depth']}" in block.attributes['tags']
            assert f"importance:{'high' if block.attributes['importance_score'] >= 0.8 else 'medium' if block.attributes['importance_score'] >= 0.6 else 'low'}" in block.attributes['tags']
        
        # Test content type tag
        assert f"content:{block.attributes['content_type']}" in block.attributes['tags']

