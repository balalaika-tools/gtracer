"""JSON logging setup for gtracer.

Configures the "gtracer" logger at import time with its own JSON stdout
handler. No user setup required.

Environment variables:

    GTRACER_ENABLED=false
        Silence all span output. Tracing mechanics stay active.

    GTRACER_LOG_TO_FILE=true
        Write spans to a file in addition to stdout.
        Creates Logs/gtracer_<YYYYMMDD_HHMMSS>.jsonl in the working directory.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
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

    formatter = _JSONFormatter()

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(level)
    stdout_handler.setFormatter(formatter)
    logger.addHandler(stdout_handler)

    log_to_file = os.environ.get("GTRACER_LOG_TO_FILE", "false").strip().lower() == "true"
    if log_to_file:
        log_dir = Path.cwd() / "Logs"
        log_dir.mkdir(exist_ok=True)
        log_file = log_dir / f"gtracer_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
