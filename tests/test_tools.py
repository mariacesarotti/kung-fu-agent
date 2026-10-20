import json
from unittest.mock import MagicMock

import psycopg
import pytest

from kung_fu_agent import tools


@pytest.fixture
def conn() -> MagicMock:
    return MagicMock()


@pytest.fixture
def client() -> MagicMock:
    return MagicMock()


def test_unknown_tool_is_an_error(conn, client):
    text, is_error = tools.dispatch_tool(conn, client, "voar", {})

    assert is_error is True
    assert "voar" in json.loads(text)["error"]


def test_missing_argument_is_an_error(conn, client):
    text, is_error = tools.dispatch_tool(conn, client, "build_training", {"level": "iniciante"})

    assert is_error is True
    assert "count" in json.loads(text)["error"]


def test_validation_message_reaches_the_model(conn, client):
    text, is_error = tools.dispatch_tool(
        conn, client, "build_training", {"level": "mestre", "count": 3}
    )

    assert is_error is True
    assert "mestre" in json.loads(text)["error"]
    conn.cursor.assert_not_called()


def test_database_error_rolls_back_and_hides_details(conn, client, monkeypatch):
    def broken(*args):
        raise psycopg.OperationalError("connection to 127.0.0.1 failed: password=segredo")

    monkeypatch.setattr(tools, "build_training", broken)

    text, is_error = tools.dispatch_tool(
        conn, client, "build_training", {"level": "iniciante", "count": 3}
    )

    assert is_error is True
    conn.rollback.assert_called_once()
    assert "segredo" not in text


def test_unexpected_error_hides_details(conn, client, monkeypatch):
    def broken(*args):
        raise RuntimeError("detalhe interno")

    monkeypatch.setattr(tools, "search_stances", broken)

    text, is_error = tools.dispatch_tool(conn, client, "search_stances", {"query": "equilíbrio"})

    assert is_error is True
    assert "detalhe interno" not in text


def test_success_returns_valid_json(conn, client, monkeypatch):
    rows = [{"nome": "Sei Ping Ma", "caracteres": "四平馬", "distance": 0.3246}]
    fake_search = MagicMock(return_value=rows)
    monkeypatch.setattr(tools, "search_stances", fake_search)

    text, is_error = tools.dispatch_tool(conn, client, "search_stances", {"query": "pernas"})

    assert is_error is False
    assert json.loads(text) == rows
    assert "四平馬" in text
    fake_search.assert_called_once_with(conn, client, "pernas")