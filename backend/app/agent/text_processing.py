from bs4 import BeautifulSoup 
from markdownify import markdownify 
from langchain_text_splitters import MarkdownHeaderTextSplitter,RecursiveCharacterTextSplitter 

def process_webpage(html : str) -> list[str] :
    soup  = BeautifulSoup(html,'html.parser')
    for element in soup.select(
        "script, style, nav, footer, header, aside, "
        "iframe, noscript" 
    ):
        element.decompose() 
    main_content = soup.find('main') or soup.find('article')

    if main_content is None :
        main_content = soup.body or soup  
    markdown_text = markdownify(
        str(main_content),
        heading_style = 'ATX',
        strip = ['img'],
    ).strip()
    if not markdown_text :
        return [] 
    headers_to_split_on = [
        ('#','h1'),
        ('##','h2'),
        ('###','h3')
    ] 
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on = headers_to_split_on,
        strip_headers = False 
    )
    sections = markdown_splitter.split_text(markdown_text)
    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = []
    for section in sections :
        heading_context = " > ".join(section.metadata.values())
        section_text = section.page_content
        if heading_context :
            section_text = f"{heading_context}\n\n{section_text}" 
        section_chunks = recursive_splitter.split(section_text)
        chunks.extend(section_chunks)

    return chunks     


