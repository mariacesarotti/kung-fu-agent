import psycopg
from google import genai
from pgvector import Vector
from psycopg.rows import dict_row

from kung_fu_agent.db import connect
from kung_fu_agent.embeddings import embed_text

SEARCH_SQL = """
    SELECT id, nome, nivel, confianca, content,
           embedding <=> %s AS distance
    FROM stances
    ORDER BY distance
    LIMIT %s
"""


def build_query_text(query: str) -> str:
    return f"task: search result | query: {query}"


def search_stances(
    conn: psycopg.Connection, client: genai.Client, query: str, limit: int = 3
) -> list[dict]:
    vector = embed_text(client, build_query_text(query))
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(SEARCH_SQL, (Vector(vector), limit))
        return cur.fetchall()


if __name__ == "__main__":
    queries = [
        "quero fortalecer as pernas sem me mexer",
        "base para chutar rápido",
        "como sair da frente de um golpe girando o corpo",
        "receita de bolo de cenoura",
    ]

    with connect() as conn, genai.Client() as client:
        for query in queries:
            print(f"\n{query}")
            for row in search_stances(conn, client, query):
                print(f"  {row['distance']:.4f}  {row['nome']}")