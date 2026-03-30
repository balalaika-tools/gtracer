"""JSON logging setup for gtracer.

Configures the "gtracer" logger at import time with its own JSON stdout
handler. No user setup required.

To silence span output in production set the env var before starting the app:

    GTRACER_ENABLED=false

Spans will still be created and timed — only stdout output is suppressed.
"""

from __future__ import annotations

import json
import logging
from typing import Any

# Standard LogRecord attribute names — excluded from the extra-fields pass
# so we don't double-emit Python internals alongside our payload fields.
_STD_RECORD_KEYS: frozenset[str] = frozenset({
    "name", "msg", "args", "created", "filename", "funcName",
    "levelname", "levelno", "lineno", "module", "msecs", "pathname",
    "process", "processName", "relativeCreated", "stack_info",
    "exc_info", "exc_text", "thread", "threadName", "taskName",
    "message", "asctime",
})


class _JSONFormatter(logging.Formatter):
    """Structured JSON formatter.

    Each log line is a single JSON object with keys:
      ts, level, logger, message, [exception], [stack_info], [<extras>]

    Any extra={} fields passed to logger.trace() / logger.info() etc. are
    iterated from the LogRecord and included as top-level JSON fields.
    This is what makes span payloads (event, span_name, trace_id, attrs, ...)
    appear as structured fields rather than buried in the message string.
    """

    _DATEFMT = "%Y-%m-%dT%H:%M:%S"

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts":      self.formatTime(record, self._DATEFMT),
            "level":   record.levelname,
            "logger":  record.name,
            "message": record.getMessage(),
        }

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        if record.stack_info:
            payload["stack_info"] = self.formatStack(record.stack_info)

        # Forward any extra={} fields added by the caller.
        for key, val in record.__dict__.items():
            if key not in _STD_RECORD_KEYS and not key.startswith("_"):
                payload[key] = val

        return json.dumps(payload, ensure_ascii=False, default=str)


def _configure(enabled: bool) -> None:
    """Internal: attach JSON handler to the gtracer logger.

    Sets level to TRACE (25) when enabled, WARNING (30) when disabled.
    WARNING causes isEnabledFor(25) to return False — records are dropped
    before the formatter or handler ever runs.
    """
    level = 25 if enabled else logging.WARNING

    logger = logging.getLogger("gtracer")
    logger.handlers.clear()
    logger.setLevel(level)
    logger.propagate = False  # own handler — never touches root logger

    handler = logging.StreamHandler()
    handler.setLevel(level)
    handler.setFormatter(_JSONFormatter())
    logger.addHandler(handler)
