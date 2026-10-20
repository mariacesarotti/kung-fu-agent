from unittest.mock import MagicMock

import psycopg
import pytest
from fastapi.testclient import TestClient

from kung_fu_agent import api
from kung_fu_agent.agent import ModelError


@pytest.fixture
def http(monkeypatch):
    api.app.dependency_overrides[api.get_conn] = lambda: MagicMock()
    api.app.dependency_overrides[api.get_client] = lambda: MagicMock()
    yield TestClient(api.app)
    api.app.dependency_overrides.clear()


def test_valid_question(http, monkeypatch):
    calls = [{"name": "search_stances", "arguments": {"query": "equilíbrio"}}]
    monkeypatch.setattr(api, "run_agent", MagicMock(return_value=("Dok Lap Ma.", calls)))

    response = http.post("/ask", json={"question": "Quais posturas trabalham equilíbrio?"})

    assert response.status_code == 200
    assert response.json() == {"answer": "Dok Lap Ma.", "tool_calls": calls}


@pytest.mark.parametrize(
    "body",
    [{}, {"question": ""}, {"question": "   "}, {"question": "x" * 501}, {"question": 5}],
)
def test_invalid_question_is_422(http, monkeypatch, body):
    run_agent = MagicMock()
    monkeypatch.setattr(api, "run_agent", run_agent)

    response = http.post("/ask", json=body)

    assert response.status_code == 422
    run_agent.assert_not_called()


@pytest.mark.parametrize(("retryable", "status"), [(True, 503), (False, 502)])
def test_model_error_maps_to_status(http, monkeypatch, retryable, status):
    monkeypatch.setattr(api, "run_agent", MagicMock(side_effect=ModelError(retryable)))

    response = http.post("/ask", json={"question": "equilíbrio?"})

    assert response.status_code == status
    assert "detail" in response.json()


def test_database_down_is_503_without_details(http):
    def broken_conn():
        raise psycopg.OperationalError("connection failed: password=segredo")

    api.app.dependency_overrides[api.get_conn] = broken_conn

    response = http.post("/ask", json={"question": "equilíbrio?"})

    assert response.status_code == 503
    assert "segredo" not in response.text