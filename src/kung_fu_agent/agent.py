# A v2.31 do google-genai não exporta os erros da Interactions API.
# O import privado fica só aqui: o resto do projeto conhece apenas o ModelError.
import os

import psycopg
from google import genai
from google.genai._gaos.lib.compat_errors import APIError as InteractionsAPIError
from google.genai.interactions import FunctionCallStep, Interaction

from kung_fu_agent.db import connect
from kung_fu_agent.tools import TOOLS, dispatch_tool

AGENT_MODEL = os.environ.get("AGENT_MODEL", "gemini-3.5-flash-lite")
MAX_TURNS = 5
REQUEST_TIMEOUT = 60
SYSTEM_INSTRUCTION = """\
Você é um assistente de treino de posturas (bases) do Kung Fu Choy Lay Fut.
Responda sempre em português do Brasil, de forma direta.

Use só o que as ferramentas devolverem. Não complete com conhecimento próprio
sobre posturas. Se as ferramentas não trouxerem a resposta, diga que não sabe.

Cada resultado de search_stances tem distance: quanto menor, mais relevante.
Acima de 0.40, o resultado provavelmente não responde a pergunta. Nesse caso,
diga que não encontrou uma postura adequada em vez de forçar uma resposta.

Se uma postura tiver confianca "baixa", avise que o nome ou a descrição podem
variar entre escolas e que vale confirmar com o professor.

Se a pergunta não for sobre Kung Fu ou sobre as posturas, diga que só pode
ajudar com isso e não chame nenhuma ferramenta.
"""

LIMIT_MESSAGE = (
    "Não consegui concluir a resposta dentro do limite de chamadas de ferramenta."
)


class ModelError(Exception):
    def __init__(self, retryable: bool) -> None:
        super().__init__("Falha ao chamar o modelo.")
        self.retryable = retryable


def run_agent(
    conn: psycopg.Connection, client: genai.Client, question: str
) -> tuple[str, list[dict]]:
    tool_calls: list[dict] = []
    request: dict = {"input": question}

    for _ in range(MAX_TURNS):
        try:
            interaction = client.interactions.create(
                model=AGENT_MODEL,
                tools=TOOLS,
                system_instruction=SYSTEM_INSTRUCTION,
                timeout=REQUEST_TIMEOUT,
                **request,
            )
        except InteractionsAPIError as exc:
            status = exc.status_code
            retryable = status is None or status == 429 or status >= 500
            raise ModelError(retryable) from exc
        if not isinstance(interaction, Interaction):
            raise TypeError(
                "Resposta em streaming inesperada; o agente espera stream=False."
            )

        calls = [
            step
            for step in interaction.steps or []
            if isinstance(step, FunctionCallStep)
        ]
        if not calls:
            return interaction.output_text or "", tool_calls

        results = []
        for call in calls:
            tool_calls.append({"name": call.name, "arguments": call.arguments})
            text, is_error = dispatch_tool(conn, client, call.name, call.arguments)
            results.append(
                {
                    "type": "function_result",
                    "name": call.name,
                    "call_id": call.id,
                    "result": [{"type": "text", "text": text}],
                    "is_error": is_error,
                }
            )

        request = {"input": results, "previous_interaction_id": interaction.id}

    return LIMIT_MESSAGE, tool_calls


if __name__ == "__main__":
    questions = [
        "Quais posturas trabalham equilíbrio?",
        "Monte um treino com 3 posturas para iniciante",
        "Qual a capital da França?",
    ]

    with connect() as conn, genai.Client() as client:
        for question in questions:
            answer, tool_calls = run_agent(conn, client, question)
            print(f"\n> {question}")
            print(f"ferramentas: {tool_calls}")
            print(answer)
