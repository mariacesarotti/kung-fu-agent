import json
from pathlib import Path

import psycopg
from google import genai
from pgvector import Vector
from psycopg import sql

from kung_fu_agent.chunking import build_chunk_text
from kung_fu_agent.embeddings import EMBEDDING_DIM, embed_text
from kung_fu_agent.db import connect

DATA_PATH = Path(__file__).parents[2] / "data" / "stances.json"


CREATE_TABLE = sql.SQL("""
    CREATE TABLE IF NOT EXISTS stances (
        id        text PRIMARY KEY,
        nome      text NOT NULL,
        nivel     text NOT NULL CHECK (nivel IN ('iniciante', 'intermediario', 'avancado')),
        confianca text NOT NULL,
        content   text NOT NULL,
        embedding vector({dim}) NOT NULL
    )
""").format(dim=sql.Literal(EMBEDDING_DIM))

UPSERT = """
    INSERT INTO stances (id, nome, nivel, confianca, content, embedding)
    VALUES (%s, %s, %s, %s, %s, %s)
    ON CONFLICT (id) DO UPDATE SET
        nome      = EXCLUDED.nome,
        nivel     = EXCLUDED.nivel,
        confianca = EXCLUDED.confianca,
        content   = EXCLUDED.content,
        embedding = EXCLUDED.embedding
"""


def ensure_schema(conn: psycopg.Connection) -> None:
    conn.execute(CREATE_TABLE)


def load_stances(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)

def ingest(conn: psycopg.Connection, client: genai.Client, stances: list[dict]) -> int:
    for stance in stances:
        text = build_chunk_text(stance)
        vector = embed_text(client, text)
        conn.execute(
            UPSERT,
            (
                stance["id"],
                stance["nome"],
                stance["nivel"],
                stance["confianca"],
                text,
                Vector(vector),
            ),
        )
    return len(stances)


if __name__ == "__main__":
    stances = load_stances(DATA_PATH)

    with connect() as conn, genai.Client() as client:
        ensure_schema(conn)
        processed = ingest(conn, client, stances)
        total = conn.execute("SELECT count(*) FROM stances").fetchone()[0]

    print(f"{processed} posturas processadas, {total} linhas na tabela.")