import json
from pathlib import Path

from google import genai
from google.genai import types

from kung_fu_agent.chunking import build_chunk_text

EMBEDDING_MODEL = "gemini-embedding-2"
EMBEDDING_DIM = 768


def embed_text(client: genai.Client, text: str) -> list[float]:
    result = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text,
        config=types.EmbedContentConfig(output_dimensionality=EMBEDDING_DIM),
    )
    values = result.embeddings[0].values
    if values is None:
        raise ValueError("A API devolveu um embedding sem valores.")
    return values


if __name__ == "__main__":
    path = Path(__file__).parents[2] / "data" / "stances.json"
    with path.open(encoding="utf-8") as f:
        stances = json.load(f)

    text = build_chunk_text(stances[0])

    with genai.Client() as client:
        vector = embed_text(client, text)

    print(len(vector), vector[:5])
