import logging
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from typing import Annotated, Any

import psycopg
from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse
from google import genai
from pydantic import BaseModel, Field, StringConstraints

from kung_fu_agent.agent import ModelError, run_agent
from kung_fu_agent.db import connect

logger = logging.getLogger(__name__)


class AskRequest(BaseModel):
    question: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=3, max_length=500),
        Field(
            description="Pergunta sobre as posturas de Choy Lay Fut.",
            examples=["Quais posturas trabalham equilíbrio?"],
        ),
    ]


class ToolCall(BaseModel):
    name: str = Field(description="Nome da ferramenta que o modelo pediu.")
    arguments: dict[str, Any] = Field(description="Argumentos enviados pelo modelo.")


class AskResponse(BaseModel):
    answer: str = Field(description="Resposta final do agente.")
    tool_calls: list[ToolCall] = Field(description="Ferramentas chamadas, na ordem.")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    with genai.Client() as client:
        app.state.client = client
        yield


app = FastAPI(title="Kung Fu Agent", lifespan=lifespan)


def get_conn() -> Iterator[psycopg.Connection]:
    with connect() as conn:
        yield conn


def get_client(request: Request) -> genai.Client:
    return request.app.state.client


Conn = Annotated[psycopg.Connection, Depends(get_conn)]
Client = Annotated[genai.Client, Depends(get_client)]


@app.post("/ask")
def ask(body: AskRequest, conn: Conn, client: Client) -> AskResponse:
    answer, tool_calls = run_agent(conn, client, body.question)
    return AskResponse(
        answer=answer, tool_calls=[ToolCall(**call) for call in tool_calls]
    )


@app.exception_handler(psycopg.OperationalError)
async def database_unavailable(
    request: Request, exc: psycopg.OperationalError
) -> JSONResponse:
    logger.error("Banco indisponível", exc_info=exc)
    return JSONResponse(
        status_code=503,
        content={
            "detail": "Serviço temporariamente indisponível. Tente de novo em instantes."
        },
    )


@app.exception_handler(ModelError)
async def model_failed(request: Request, exc: ModelError) -> JSONResponse:
    logger.error("Falha ao chamar o modelo", exc_info=exc)
    if exc.retryable:
        return JSONResponse(
            status_code=503,
            content={
                "detail": "O modelo está indisponível agora. Tente de novo em instantes."
            },
        )
    return JSONResponse(
        status_code=502,
        content={"detail": "Não foi possível consultar o modelo."},
    )
