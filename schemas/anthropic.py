"""Pydantic schemas for the Anthropic Messages API (M1-M3 fields only)."""
from __future__ import annotations

from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field


class TextBlock(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: Literal["text"] = "text"
    text: str


class ImageBlock(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: Literal["image"] = "image"
    source: Any


class ToolUseBlock(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: Literal["tool_use"] = "tool_use"
    id: str
    name: str
    input: Any


class ToolResultBlock(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: Literal["tool_result"] = "tool_result"
    tool_use_id: str
    content: Any = None


ContentBlock = Annotated[
    Union[TextBlock, ImageBlock, ToolUseBlock, ToolResultBlock],
    Field(discriminator="type"),
]


class Usage(BaseModel):
    model_config = ConfigDict(extra="allow")

    input_tokens: int | None = None
    output_tokens: int | None = None


class MessagesRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str
    system: Union[str, list[TextBlock]] | None = None
    messages: list[Any]
    max_tokens: int
    stream: bool = False
    tools: Any | None = None
    tool_choice: Any | None = None
    thinking: Any | None = None


class MessagesResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    role: str
    content: list[ContentBlock]
    stop_reason: str
    usage: Usage
