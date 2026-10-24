
import os
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings


# Load the backend/.env file
BACKEND_DIR = Path(__file__).resolve().parents[2]
load_dotenv(BACKEND_DIR / ".env")

embeddings_model = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
)


def retrieve_relevant_chunks(
    question: str,
    chunks: list[str],
    top_k: int = 3,
) -> list[dict]:
    """
    Retrieve the most relevant webpage chunks for a question.

    Returns a ranked list containing each chunk and its
    cosine similarity score.
    """

    if not question.strip() or not chunks:
        return []
    valid_chunks = [
        (index, chunk)
        for index, chunk in enumerate(chunks)
        if chunk.strip()
    ]

    if not valid_chunks:
        return []

    question_vector = np.asarray(
        embeddings_model.embed_query(question),
        dtype=np.float64,
    )

    chunk_texts = [chunk for _, chunk in valid_chunks]

    chunk_vectors = np.asarray(
        embeddings_model.embed_documents(chunk_texts),
        dtype=np.float64,
    )
    question_norm = np.linalg.norm(question_vector)
    chunk_norms = np.linalg.norm(chunk_vectors, axis=1)

    if question_norm == 0 or np.any(chunk_norms == 0):
        raise ValueError("An embedding has zero magnitude.")

    scores = (
        chunk_vectors @ question_vector
    ) / (chunk_norms * question_norm)

    ranked_indices = np.argsort(scores)[::-1][:max(1, top_k)]

    results = []

    for index in ranked_indices:
        original_index, chunk = valid_chunks[index]

        results.append({
            "chunk_index": original_index,
            "chunk": chunk,
            "score": float(scores[index]),
        })

    return results
