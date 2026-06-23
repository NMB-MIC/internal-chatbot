from __future__ import annotations

import json
import time
from typing import Any, TypeVar

import requests
from pydantic import BaseModel, ValidationError

from app.config import settings


SchemaType = TypeVar("SchemaType", bound=BaseModel)


class OllamaClientError(RuntimeError):
    """Raised when the local Ollama server returns an invalid response."""


class OllamaClient:
    def __init__(
        self,
        base_url: str = settings.ollama_base_url,
        model: str = settings.ollama_model,
        timeout_seconds: int = settings.ollama_request_timeout_seconds,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.session = requests.Session()

    def list_models(self) -> list[dict[str, Any]]:
        response = self.session.get(
            f"{self.base_url}/api/tags",
            timeout=10,
        )
        response.raise_for_status()

        return response.json().get("models", [])

    def healthcheck(self) -> dict[str, Any]:
        models = self.list_models()
        installed_model_names = [model.get("name") for model in models]

        return {
            "base_url": self.base_url,
            "configured_model": self.model,
            "server_reachable": True,
            "configured_model_available": self.model in installed_model_names,
            "installed_models": installed_model_names,
        }

    def show_model(self) -> dict[str, Any]:
        response = self.session.post(
            f"{self.base_url}/api/show",
            json={"model": self.model},
            timeout=30,
        )
        response.raise_for_status()

        return response.json()

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        system_prompt: str | None = None,
        temperature: float | None = None,
        response_schema: dict[str, Any] | None = None,
        think: bool | None = None,
    ) -> dict[str, Any]:
        prepared_messages = list(messages)

        if system_prompt:
            prepared_messages = [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                *prepared_messages,
            ]

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": prepared_messages,
            "stream": False,
            "think": settings.ollama_think if think is None else think,
            "keep_alive": settings.ollama_keep_alive,
            "options": {
                "temperature": (
                    settings.ollama_temperature
                    if temperature is None
                    else temperature
                ),
                "num_ctx": settings.ollama_num_ctx,
            },
        }

        if response_schema is not None:
            payload["format"] = response_schema

        started_at = time.perf_counter()

        try:
            response = self.session.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            data = response.json()

        except requests.RequestException as exc:
            raise OllamaClientError(
                f"Ollama request failed: {exc}"
            ) from exc

        elapsed_seconds = time.perf_counter() - started_at

        message = data.get("message", {})
        content = message.get("content", "")

        eval_count = data.get("eval_count", 0) or 0
        eval_duration_ns = data.get("eval_duration", 0) or 0

        tokens_per_second = None

        if eval_count and eval_duration_ns:
            tokens_per_second = eval_count / (eval_duration_ns / 1_000_000_000)

        return {
            "content": content,
            "thinking": message.get("thinking"),
            "metrics": {
                "elapsed_seconds": round(elapsed_seconds, 3),
                "prompt_eval_count": data.get("prompt_eval_count"),
                "eval_count": eval_count,
                "tokens_per_second": (
                    round(tokens_per_second, 3)
                    if tokens_per_second is not None
                    else None
                ),
            },
            "raw": data,
        }

    def chat_json(
        self,
        messages: list[dict[str, str]],
        schema: type[SchemaType],
        *,
        system_prompt: str | None = None,
        think: bool | None = False,
    ) -> tuple[SchemaType, dict[str, Any]]:
        schema_json = schema.model_json_schema()

        augmented_system_prompt = (
            f"{system_prompt or ''}\n\n"
            "Return only a JSON object that strictly follows this JSON schema:\n"
            f"{json.dumps(schema_json, ensure_ascii=False)}"
        ).strip()

        result = self.chat(
            messages=messages,
            system_prompt=augmented_system_prompt,
            temperature=0.0,
            response_schema=schema_json,
            think=think,
        )

        try:
            parsed = schema.model_validate_json(result["content"])

        except ValidationError as exc:
            raise OllamaClientError(
                "The model returned JSON that did not match the required schema.\n"
                f"Raw model response:\n{result['content']}"
            ) from exc

        return parsed, result