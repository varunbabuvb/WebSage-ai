from bs4 import BeautifulSoup
from markdownify import markdownify
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)


def process_webpage(html: str) -> tuple[list[str], str]:
    """
    Convert webpage HTML into Markdown and split it into
    semantically organized chunks.

    Returns:
        chunks: List of text chunks for embedding and retrieval.
        markdown_text: Cleaned Markdown before chunking.
    """

    # 1. Parse the HTML
    soup = BeautifulSoup(html, "html.parser")

    # 2. Remove elements that generally contain non-content material
    for element in soup.select(
        "script, style, nav, footer, header, aside, "
        "iframe, noscript, svg"
    ):
        element.decompose()

    # 3. Prefer the main content area when available
    main_content = (
        soup.find("main")
        or soup.find("article")
        or soup.body
        or soup
    )

    # 4. Convert HTML into Markdown
    markdown_text = markdownify(
        str(main_content),
        heading_style="ATX",
        strip=["img"],
    ).strip()

    if not markdown_text:
        return [], ""

    # 5. Split the Markdown using heading structure
    headers_to_split_on = [
        ("#", "h1"),
        ("##", "h2"),
        ("###", "h3"),
    ]

    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=True,
    )

    sections = markdown_splitter.split_text(markdown_text)

    # 6. Split large sections while preserving heading context
    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", " ", ""],
    )

    chunks = []

    for section in sections:
        section_text = section.page_content.strip()

        # Skip empty sections
        if not section_text:
            continue

        # Add heading metadata to each chunk for context
        heading_context = " > ".join(section.metadata.values())

        if heading_context:
            section_text = f"{heading_context}\n\n{section_text}"

        section_chunks = recursive_splitter.split_text(section_text)

        chunks.extend(
            chunk for chunk in section_chunks if chunk.strip()
        )

    return chunks, markdown_text
