from __future__ import annotations

import base64
import json
import mimetypes
import os
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_RESPONSES_DIR = ROOT_DIR / "responses"
DEFAULT_CACHE_DIR = ROOT_DIR / ".cache" / "openai_compatible"


def resolve_api_key(api_key: str | None = None) -> str:
    if api_key:
        return api_key

    for env_var in ("OPENAI_API_KEY",):
        value = os.environ.get(env_var)
        if value:
            return value

    raise KeyError("Set OPENAI_API_KEY or pass --api-key before using the OpenAI-compatible scripts.")


def resolve_base_url(base_url: str | None = None) -> str:
    resolved = base_url or os.environ.get("OPENAI_BASE_URL")
    if not resolved:
        raise KeyError("Set OPENAI_BASE_URL or pass --base-url before running.")

    resolved = resolved.rstrip("/")
    parsed = urlsplit(resolved)
    if parsed.path.rstrip("/"):
        return resolved
    return urlunsplit(parsed._replace(path="/v1"))


def sanitize_name(value: str) -> str:
    allowed = []
    for char in value:
        if char.isalnum() or char in ("-", "_", "."):
            allowed.append(char)
        else:
            allowed.append("_")
    return "".join(allowed).strip("._") or "run"


def ensure_record(item: Any, dataset_path: Path, *, line_number: int | None = None) -> dict[str, Any]:
    if isinstance(item, dict):
        return item
    location = f"{dataset_path}:{line_number}" if line_number is not None else str(dataset_path)
    raise ValueError(f"Expected each dataset entry in {location} to be an object.")


def load_records(dataset_path: Path) -> list[dict[str, Any]]:
    raw_text = dataset_path.read_text(encoding="utf-8").strip()
    if not raw_text:
        raise ValueError(f"{dataset_path} is empty.")

    if raw_text.startswith("["):
        data = json.loads(raw_text)
        if not isinstance(data, list):
            raise ValueError(f"{dataset_path} must contain a JSON list.")
        return [ensure_record(item, dataset_path) for item in data]

    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(raw_text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        records.append(ensure_record(json.loads(stripped), dataset_path, line_number=line_number))
    return records


def resolve_image_path(raw_image_path: str, dataset_path: Path) -> Path:
    image_path = Path(raw_image_path)
    candidates = []
    if image_path.is_absolute():
        candidates.append(image_path)
    else:
        candidates.append(dataset_path.parent / image_path)
        candidates.append(ROOT_DIR / image_path)

    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if resolved.exists():
            return resolved

    raise FileNotFoundError(
        f"Could not resolve image_path={raw_image_path!r} relative to {dataset_path}."
    )


def load_image_payload(image_path: Path) -> tuple[str, str]:
    mime_type = mimetypes.guess_type(image_path.name)[0] or "image/png"
    if not mime_type.startswith("image/"):
        raise ValueError(f"Unsupported image MIME type for {image_path}: {mime_type}")
    image_b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return image_b64, mime_type


def validate_record(record: dict[str, Any]) -> tuple[str, str]:
    prompt = record.get("prompt")
    raw_image_path = record.get("image_path")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError(f"Record {record.get('eval_id')} is missing a non-empty prompt.")
    if not isinstance(raw_image_path, str) or not raw_image_path.strip():
        raise ValueError(f"Record {record.get('eval_id')} is missing a non-empty image_path.")
    return prompt, raw_image_path


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True) + "\n")


def load_completed_eval_ids(output_path: Path, *, method: str, model_name: str) -> set[str]:
    completed: set[str] = set()
    if not output_path.exists():
        return completed

    with output_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError:
                continue

            eval_id = payload.get("eval_id")
            status = payload.get("status")
            same_run = (
                payload.get("method") in (None, method)
                and payload.get("model_name") in (None, model_name)
            )
            if isinstance(eval_id, str) and eval_id and same_run and status == "ok":
                completed.add(eval_id)

    return completed


def default_output_path(dataset_path: Path, model_name: str, output_dir: Path | None = None) -> Path:
    if output_dir is None:
        output_dir = DEFAULT_RESPONSES_DIR / dataset_path.parent.name
    return output_dir / f"{dataset_path.stem}__openai_compatible__{sanitize_name(model_name)}.jsonl"


def build_success_payload(
    *,
    record: dict[str, Any],
    model_name: str,
    dataset_path: Path,
    image_path: Path,
    reasoning: str | None,
    content: str,
    input_tokens: int,
    output_tokens: int,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "status": "ok",
        "eval_id": record.get("eval_id"),
        "puzzle_dir": record.get("puzzle_dir"),
        "rq": record.get("rq"),
        "variant": record.get("variant"),
        "method": "openai_compatible",
        "model_name": model_name,
        "dataset_file": str(dataset_path),
        "image_path": str(image_path),
        "content": content,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }
    if reasoning:
        payload["reasoning"] = reasoning
    return payload


def coerce_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(coerce_text(item) for item in value)
    if isinstance(value, dict):
        if "text" in value:
            return coerce_text(value["text"])
        if value.get("type") == "text":
            return coerce_text(value.get("text"))
        return ""

    text = getattr(value, "text", None)
    if text is not None:
        return coerce_text(text)

    item_type = getattr(value, "type", None)
    if item_type == "text":
        return coerce_text(getattr(value, "text", ""))

    return ""


def coerce_reasoning(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(coerce_reasoning(item) for item in value)
    if isinstance(value, dict):
        return coerce_text(value)
    return coerce_text(value)


class OpenAICompatibleStreamingClient:
    def __init__(
        self,
        model_name: str,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout_seconds: float = 300.0,
    ):
        self.model_name = model_name
        self.api_key = resolve_api_key(api_key)
        self.base_url = resolve_base_url(base_url)
        self.timeout_seconds = timeout_seconds

    def _make_client(self) -> Any:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "The `openai` Python package is required to run the OpenAI-compatible scripts. "
                "Install it in the active environment first."
            ) from exc

        return OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout_seconds,
        )

    def list_models(self) -> list[str]:
        client = self._make_client()
        response = client.models.list()
        items = getattr(response, "data", response)
        model_ids: list[str] = []
        for item in items:
            model_id = getattr(item, "id", None)
            if model_id is None and isinstance(item, dict):
                model_id = item.get("id")
            if isinstance(model_id, str) and model_id:
                model_ids.append(model_id)
        return sorted(set(model_ids))

    def generate_with_images(
        self,
        *,
        text: str,
        images: list[str],
        image_media_type: str,
        max_tokens: int | None,
        temperature: float | None,
        presence_penalty: float | None,
        verbose: bool,
    ) -> tuple[str | None, str, int, int]:
        content_parts: list[dict[str, Any]] = []
        for b64 in images:
            content_parts.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/{image_media_type};base64,{b64}"},
                }
            )
        content_parts.append({"type": "text", "text": text})

        client = self._make_client()
        stream = client.chat.completions.create(
            messages=[{"role": "user", "content": content_parts}],
            model=self.model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            presence_penalty=presence_penalty,
            stream=True,
            stream_options={"include_usage": True},
        )

        reasoning_parts: list[str] = []
        output_parts: list[str] = []
        input_tokens = 0
        output_tokens = 0

        for chunk in stream:
            usage = getattr(chunk, "usage", None)
            if usage is not None:
                input_tokens = getattr(usage, "prompt_tokens", 0) or 0
                output_tokens = getattr(usage, "completion_tokens", 0) or 0

            if not getattr(chunk, "choices", None):
                continue

            delta = chunk.choices[0].delta
            reasoning_chunk = coerce_reasoning(
                getattr(delta, "reasoning", None) or getattr(delta, "reasoning_content", None)
            )
            if reasoning_chunk:
                if verbose:
                    print(reasoning_chunk, end="", flush=True)
                reasoning_parts.append(reasoning_chunk)

            content_chunk = coerce_text(getattr(delta, "content", None))
            if content_chunk:
                if verbose:
                    print(content_chunk, end="", flush=True)
                output_parts.append(content_chunk)

        reasoning = "".join(reasoning_parts).strip() if reasoning_parts else None
        content = "".join(output_parts).strip()
        return reasoning, content, input_tokens, output_tokens
