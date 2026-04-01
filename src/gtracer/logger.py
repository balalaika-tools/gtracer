"""JSON logging setup for gtracer.

Configures the "gtracer" logger at import time with its own JSON stdout
handler. No user setup required.

Environment variables:

    GTRACER_ENABLED=false
        Silence all span output. Tracing mechanics stay active.

    GTRACER_LOG_TO_FILE=true
        Write spans to a file in addition to (or instead of) stdout.
        Creates logs/gtracer_<YYYYMMDD_HHMMSS>.jsonl in the working directory.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# Custom TRACE level — below DEBUG so it never leaks through INFO-level handlers.
_TRACE_LEVEL: int = 5
logging.addLevelName(_TRACE_LEVEL, "TRACE")

# Standard LogRecord attribute names — excluded from the extra-fields pass
# so we don't double-emit Python internals alongside our payload fields.
_STD_RECORD_KEYS: frozenset[str] = frozenset({
    "name", "msg", "args", "created", "filename", "funcName",
    "levelname", "levelno", "lineno", "module", "msecs", "pathname",
    "process", "processName", "relativeCreated", "stack_info",
    "exc_info", "exc_text", "thread", "threadName", "taskName",
    "message", "asctime",
})

_JSON_STR_FALLBACK_LIMIT: int = 1000


def _json_default(obj: Any) -> str:
    """Fallback serializer for json.dumps — truncates large str() representations."""
    s = str(obj)
    if len(s) > _JSON_STR_FALLBACK_LIMIT:
        return s[:_JSON_STR_FALLBACK_LIMIT] + " ...[truncated]"
    return s


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

        return json.dumps(payload, ensure_ascii=False, default=_json_default)


class _ImmediateFileHandler(logging.FileHandler):
    """FileHandler that opens the file line-buffered.

    Each JSON record ends with ``\\n`` (the ``StreamHandler.terminator``),
    so line buffering ensures every record is flushed to the OS kernel
    immediately on ``stream.write()`` — before the explicit ``flush()``
    call in ``StreamHandler.emit()``.  This eliminates the race window
    where data sits only in Python's internal buffer and would be lost
    if the process is killed.
    """

    def _open(self):  # type: ignore[override]
        return open(
            self.baseFilename,
            self.mode,
            buffering=1,  # line-buffered
            encoding=self.encoding,
            errors=self.errors,
        )


_own_handlers: list[logging.Handler] = []

# Track current state so configure() can change one setting without re-reading env for the other.
_current_enabled: bool = True
_current_log_to_file: bool = False


def _configure(enabled: bool, log_to_file: bool = False) -> None:
    """Internal: attach JSON handler(s) to the gtracer logger.

    ``enabled`` controls stdout output.  ``log_to_file`` controls file output.
    The two are independent — ``enabled=False`` with ``log_to_file=True``
    silences the console but still writes to file.

    Only removes handlers previously added by gtracer — user-added handlers
    on the ``"gtracer"`` logger are preserved.
    """
    global _own_handlers, _current_enabled, _current_log_to_file  # noqa: PLW0603
    _current_enabled = enabled
    _current_log_to_file = log_to_file
    has_output = enabled or log_to_file

    logger = logging.getLogger("gtracer")
    for h in _own_handlers:
        logger.removeHandler(h)
        h.close()
    _own_handlers = []
    logger.setLevel(_TRACE_LEVEL if has_output else logging.CRITICAL + 1)
    logger.propagate = False  # own handler — never touches root logger

    formatter = _JSONFormatter()

    if enabled:
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setLevel(_TRACE_LEVEL)
        stdout_handler.setFormatter(formatter)
        logger.addHandler(stdout_handler)
        _own_handlers.append(stdout_handler)

    if log_to_file:
        log_dir = Path.cwd() / "logs"
        log_dir.mkdir(exist_ok=True)
        log_file = log_dir / f"gtracer_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jsonl"
        file_handler = _ImmediateFileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(_TRACE_LEVEL)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        _own_handlers.append(file_handler)


class InMemoryHandler(logging.Handler):
    """Collects span log records in a list for testing and programmatic access.

    Attach to the ``"gtracer"`` logger to inspect spans without parsing
    JSON from stdout::

        from gtracer import InMemoryHandler
        import logging

        handler = InMemoryHandler()
        logging.getLogger("gtracer").addHandler(handler)

        # ... run your agent ...

        assert len(handler.records) == 6
        assert handler.records[0].span_name == "run"
        handler.clear()
    """

    def __init__(self) -> None:
        super().__init__(level=_TRACE_LEVEL)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)

    def clear(self) -> None:
        """Remove all collected records."""
        self.records.clear()
