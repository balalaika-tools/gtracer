"""gtracer — lightweight span-based tracing for LangChain/LangGraph agents."""

import os

from gtracer.logger import _configure
from gtracer.tracer import (
    VALID_CHILDREN,
    SpanContext,
    Tracer,
    _agent_name,
    _span_id,
    _span_name,
    _tags,
    _trace_id,
    configure,
    serialise_lc_messages,
    tracer,
)
from gtracer.callbacks import TracingCallbackHandler, tracing_handler

# Auto-configure at import time.
# Set GTRACER_ENABLED=false to silence all span output in production.
# Tracing mechanics (spans, callbacks, token counts) remain active either way.
_enabled = os.environ.get("GTRACER_ENABLED", "true").strip().lower() != "false"
_configure(_enabled)

__all__ = [
    # Core tracer
    "Tracer",
    "SpanContext",
    "tracer",
    "configure",
    "serialise_lc_messages",
    "VALID_CHILDREN",
    # ContextVars (needed for tool integration)
    "_trace_id",
    "_span_id",
    "_span_name",
    "_tags",
    "_agent_name",
    # Callbacks
    "TracingCallbackHandler",
    "tracing_handler",
]
