"""gtracer — lightweight span-based tracing for LangChain/LangGraph agents."""

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
