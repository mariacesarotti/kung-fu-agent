from unittest.mock import MagicMock

import httpx
import pytest
from google.genai._gaos.lib.compat_errors import APIError, APITimeoutError
from google.genai.interactions import FunctionCallStep, Interaction

from kung_fu_agent import agent
from kung_fu_agent.agent import LIMIT_MESSAGE, MAX_TURNS, ModelError, run_agent

QUESTION = "Monte um treino com 3 posturas para iniciante"
ARGUMENTS = {"level": "iniciante", "count": 3}


def fake_interaction(interaction_id: str, steps=(), text=None) -> MagicMock:
    interaction = MagicMock(spec=Interaction)
    interaction.id = interaction_id
    interaction.steps = list(steps)
    interaction.output_text = text
    return interaction


def tool_call(call_id: str = "call-1") -> FunctionCallStep:
    return FunctionCallStep(id=call_id, name="build_training", arguments=ARGUMENTS)


def api_error(status: int) -> APIError:
    request = httpx.Request("POST", "https://example.test")
    response = httpx.Response(status, request=request)
    return APIError.generate(status, {"error": "falha"}, None, response)


@pytest.fixture
def conn() -> MagicMock:
    return MagicMock()


@pytest.fixture
def client() -> MagicMock:
    return MagicMock()


@pytest.fixture
def dispatch(monkeypatch) -> MagicMock:
    fake = MagicMock(return_value=('[{"nome": "Diu Ma"}]', False))
    monkeypatch.setattr(agent, "dispatch_tool", fake)
    return fake


def test_answer_without_tools(conn, client, dispatch):
    client.interactions.create.return_value = fake_interaction("int-1", text="Só ajudo com posturas.")

    answer, tool_calls = run_agent(conn, client, "Qual a capital da França?")

    assert answer == "Só ajudo com posturas."
    assert tool_calls == []
    assert client.interactions.create.call_count == 1
    dispatch.assert_not_called()


def test_one_tool_round_trip(conn, client, dispatch):
    client.interactions.create.side_effect = [
        fake_interaction("int-1", steps=[tool_call("call-1")]),
        fake_interaction("int-2", text="Treino pronto."),
    ]

    answer, tool_calls = run_agent(conn, client, QUESTION)

    assert answer == "Treino pronto."
    assert tool_calls == [{"name": "build_training", "arguments": ARGUMENTS}]
    dispatch.assert_called_once_with(conn, client, "build_training", ARGUMENTS)

    first, second = client.interactions.create.call_args_list
    assert first.kwargs["input"] == QUESTION
    assert "previous_interaction_id" not in first.kwargs

    assert second.kwargs["previous_interaction_id"] == "int-1"
    [result] = second.kwargs["input"]
    assert result["call_id"] == "call-1"
    assert result["is_error"] is False
    assert second.kwargs["tools"] == first.kwargs["tools"]
    assert second.kwargs["system_instruction"] == first.kwargs["system_instruction"]


def test_tool_error_goes_back_to_the_model(conn, client, dispatch):
    dispatch.return_value = ('{"error": "level inválido"}', True)
    client.interactions.create.side_effect = [
        fake_interaction("int-1", steps=[tool_call()]),
        fake_interaction("int-2", text="Não existe esse nível."),
    ]

    answer, _ = run_agent(conn, client, QUESTION)

    [result] = client.interactions.create.call_args_list[1].kwargs["input"]
    assert result["is_error"] is True
    assert answer == "Não existe esse nível."


def test_stops_after_max_turns(conn, client, dispatch):
    client.interactions.create.return_value = fake_interaction("int-x", steps=[tool_call()])

    answer, tool_calls = run_agent(conn, client, QUESTION)

    assert answer == LIMIT_MESSAGE
    assert client.interactions.create.call_count == MAX_TURNS
    assert len(tool_calls) == MAX_TURNS


@pytest.mark.parametrize(
    ("error", "retryable"),
    [
        (api_error(503), True),
        (api_error(500), True),
        (api_error(429), True),
        (APITimeoutError(httpx.Request("POST", "https://example.test")), True),
        (api_error(400), False),
        (api_error(401), False),
    ],
)
def test_gemini_errors_become_model_error(conn, client, dispatch, error, retryable):
    client.interactions.create.side_effect = error

    with pytest.raises(ModelError) as caught:
        run_agent(conn, client, QUESTION)

    assert caught.value.retryable is retryable
    assert caught.value.__cause__ is error
