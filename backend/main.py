from app.agent.graph import ask_page


if __name__ == "__main__":
    page_text = """
    WebSage AI helps users understand webpages.
    It summarizes content and extracts important points.
    Users can ask follow-up questions about the page.
    """

    result = ask_page(
        question="What does WebSage AI help users do?",
        page_text=page_text,
        page_url="https://example.com/websage",
    )

    print("\nANSWER")
    print(result["answer"])

    print("\nANSWERABLE")
    print(result["answerable"])

    print("\nCITATIONS")
    for citation in result["citations"]:
        print(f'[{citation["chunk_id"]}] {citation["excerpt"]}')