import pytest

from kung_fu_agent.chunking import build_chunk_text
from kung_fu_agent.search import build_query_text


@pytest.fixture
def stance() -> dict:
    return {
        "id": "sei_ping_ma",
        "nome": "Sei Ping Ma",
        "caracteres": "四平馬",
        "outros_nomes": ["base do cavalo", "horse stance"],
        "nivel": "iniciante",
        "descricao": "Base larga e simétrica.",
        "beneficios": ["força das pernas", "enraizamento"],
        "musculos": ["quadríceps", "glúteo máximo"],
        "confianca": "alta",
    }


def test_chunk_text_uses_document_format(stance):
    text = build_chunk_text(stance)

    assert text.startswith("title: Sei Ping Ma (四平馬) | text: ")


def test_chunk_text_has_no_python_list_syntax(stance):
    text = build_chunk_text(stance)

    assert "[" not in text
    assert "'" not in text


def test_chunk_text_includes_chosen_fields(stance):
    text = build_chunk_text(stance)

    assert "Base larga e simétrica." in text
    assert "base do cavalo, horse stance" in text
    assert "iniciante" in text
    assert "força das pernas, enraizamento" in text
    assert "quadríceps, glúteo máximo" in text


def test_chunk_text_leaves_out_internal_fields(stance):
    text = build_chunk_text(stance)

    assert "sei_ping_ma" not in text
    assert "alta" not in text


def test_chunk_text_fails_loudly_on_missing_field(stance):
    del stance["musculos"]

    with pytest.raises(KeyError):
        build_chunk_text(stance)


def test_query_text_uses_search_prefix():
    assert build_query_text("base para chutar") == "task: search result | query: base para chutar"