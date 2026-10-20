import json
import psycopg
import logging

from google import genai

from kung_fu_agent.db import connect
from kung_fu_agent.search import search_stances
from kung_fu_agent.training import MAX_COUNT, VALID_LEVELS, build_training

TOOLS = [
    {
        "type": "function",
        "name": "search_stances",
        "description": (
            "Busca posturas (bases) de Kung Fu Choy Lay Fut por significado. "
            "Use para perguntas sobre o que uma postura é, seus benefícios, "
            "músculos trabalhados ou qual postura serve para um objetivo "
            "(por exemplo: equilíbrio, chutar, fortalecer as pernas). "
            "Cada resultado traz distance: quanto menor, mais relevante."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "A necessidade da pessoa, em linguagem natural.",
                }
            },
            "required": ["query"],
        },
    },
    {
        "type": "function",
        "name": "build_training",
        "description": (
            "Monta um treino com posturas de um nível. Use quando a pessoa pedir "
            "um treino, uma sequência ou uma lista de posturas para iniciante, "
            "intermediário ou avançado. Não faz busca por significado."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "level": {
                    "type": "string",
                    "enum": list(VALID_LEVELS),
                    "description": "Nível das posturas.",
                },
                "count": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": MAX_COUNT,
                    "description": "Quantas posturas incluir no treino.",
                },
            },
            "required": ["level", "count"],
        },
    },
]

logger = logging.getLogger(__name__)

def dispatch_tool(
    conn: psycopg.Connection, client: genai.Client, name: str, arguments: dict
) -> tuple[str, bool]:
    try:
        if name == "search_stances":
            result = search_stances(conn, client, arguments["query"])
        elif name == "build_training":
            result = build_training(conn, arguments["level"], arguments["count"])
        else:
            return _error(f"Ferramenta desconhecida: {name}.")
    except KeyError as exc:
        return _error(f"Faltou o argumento {exc} para {name}.")
    except ValueError as exc:
        return _error(str(exc))
    except psycopg.Error:
        logger.exception("Erro no banco ao executar %s", name)
        conn.rollback()
        return _error("Erro ao consultar o banco de dados.")
    except Exception:
        logger.exception("Erro ao executar %s", name)
        return _error("Erro interno ao executar a ferramenta.")
    return json.dumps(result, ensure_ascii=False), False


def _error(message: str) -> tuple[str, bool]:
    return json.dumps({"error": message}, ensure_ascii=False), True


if __name__ == "__main__":
    tests = [
        ("build_training", {"level": "Iniciante", "count": 2}),
        ("build_training", {"level": "intermediário", "count": 500}),
        ("build_training", {"level": "mestre", "count": 3}),
        ("build_training", {"level": "iniciante", "count": -1}),
        ("build_training", {"level": "iniciante"}),
        ("voar", {}),
    ]

    with connect() as conn, genai.Client() as client:
        for name, arguments in tests:
            text, is_error = dispatch_tool(conn, client, name, arguments)
            print(f"{name}({arguments}) -> is_error={is_error}")
            print(f"  {text[:160]}")