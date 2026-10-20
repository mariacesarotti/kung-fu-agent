import unicodedata

import psycopg
from psycopg.rows import dict_row

VALID_LEVELS = ("iniciante", "intermediario", "avancado")
MAX_COUNT = 10

TRAINING_SQL = """
    SELECT id, nome, nivel, confianca, content
    FROM stances
    WHERE nivel = %s
    ORDER BY random()
    LIMIT %s
"""


def normalize_level(level: object) -> str:
    if not isinstance(level, str):
        raise ValueError("level precisa ser um texto.")
    text = unicodedata.normalize("NFKD", level.strip().lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    if text not in VALID_LEVELS:
        raise ValueError(f"level inválido: {level!r}. Use um de: {', '.join(VALID_LEVELS)}.")
    return text


def normalize_count(count: object) -> int:
    if isinstance(count, bool) or not isinstance(count, (int, float)) or count != int(count):
        raise ValueError(f"count precisa ser um número inteiro, veio {count!r}.")
    if count < 1:
        raise ValueError(f"count precisa ser pelo menos 1, veio {count}.")
    return min(int(count), MAX_COUNT)


def build_training(conn: psycopg.Connection, level: object, count: object) -> list[dict]:
    level = normalize_level(level)
    count = normalize_count(count)
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(TRAINING_SQL, (level, count))
        return cur.fetchall()