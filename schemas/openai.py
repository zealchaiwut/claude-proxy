"""Pydantic schemas for the OpenAI Chat Completions API (M1-M3 fields only)."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: str
    content: str | None = None


class Choice(BaseModel):
    model_config = ConfigDict(extra="allow")

    message: ChatMessage
    finish_reason: str | None = None


class Usage(BaseModel):
    model_config = ConfigDict(extra="allow")

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str
    messages: list[Any]
    max_tokens: int | None = None
    stream: bool = False
    tools: Any | None = None
    tool_choice: Any | None = None


class ChatResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    choices: list[Choice]
    usage: Usage
