"""Structured JSON-Lines request logging, ELK/ECS style (Task 12).

One line per request (HTTP request, WebSocket message turn, WebSocket disconnect) with a trace id and
timing. Every string written is passed through the same phone-number masker the input guardrail uses,
so a fixed-format PII value can never reach disk in the clear, whatever the caller passes in.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import uuid
from pathlib import Path
from typing import Any

import config
from ola_agent.guardrails import mask_pii

LOG_FILE = config.LOG_DIR / "requests.jsonl"
_LOGGER_NAME = "ola.requests"


def new_trace_id() -> str:
    return uuid.uuid4().hex


def _mask_strings(value: Any) -> Any:
    if isinstance(value, str):
        return mask_pii(value)[0]
    if isinstance(value, list):
        return [_mask_strings(v) for v in value]
    if isinstance(value, dict):
        return {k: _mask_strings(v) for k, v in value.items()}
    return value


class JsonLinesFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "@timestamp": dt.datetime.fromtimestamp(record.created, dt.timezone.utc).isoformat(timespec="milliseconds"),
            "log.level": record.levelname.lower(),
            "service.name": "ola-support-agent",
            "message": record.getMessage(),
            **getattr(record, "fields", {}),
        }
        return json.dumps(_mask_strings(entry), ensure_ascii=False)


def get_request_logger(path: Path = LOG_FILE) -> logging.Logger:
    logger = logging.getLogger(_LOGGER_NAME)
    if not any(getattr(h, "baseFilename", None) == str(path.resolve()) for h in logger.handlers):
        for h in list(logger.handlers):
            logger.removeHandler(h)
            h.close()
        path.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(path, encoding="utf-8")
        handler.setFormatter(JsonLinesFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def log_request(event_action: str, trace_id: str, duration_ms: float, **fields: Any) -> None:
    get_request_logger().info(
        event_action,
        extra={"fields": {"trace.id": trace_id, "event.action": event_action,
                          "event.duration_ms": round(duration_ms, 2), **fields}},
    )


def read_log(path: Path = LOG_FILE) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def reset_log(path: Path = LOG_FILE) -> None:
    logger = logging.getLogger(_LOGGER_NAME)
    for h in list(logger.handlers):
        logger.removeHandler(h)
        h.close()
    path.unlink(missing_ok=True)
