"""gtracer — lightweight span-based tracing for LangChain/LangGraph agents.

Emits structured JSONL spans via Python's standard ``logging`` module at a
custom TRACE level (5).  Zero infrastructure required — spans appear on
stdout the moment you import the package.

Quick start::

    from gtracer import tracer, tracing_handler, span_id, configure

    configure(truncation_limit=50_000)           # optional
    configure(enabled=False, log_to_file=True)   # or via code instead of env vars

    async def run(session_id, user_input):
        tracer.start_trace(session_id)
        with tracer.span("run", tags={"session_id": session_id}):
            with tracer.span("agent", attrs={"agent": "main"}) as span:
                result = await agent.ainvoke(
                    {"messages": [{"role": "user", "content": user_input}]},
                    config={"callbacks": [tracing_handler]},
                )
                span.set_attr("output_type", type(result).__name__)
                return result

Environment variables (overridable via ``configure()``):

    GTRACER_ENABLED     — set to "false" to suppress all output (default: "true")
    GTRACER_LOG_TO_FILE — set to "true" to write spans to a local .jsonl file
                         (works independently of GTRACER_ENABLED)
"""

import os

try:
    from importlib.metadata import version as _pkg_version
    __version__ = _pkg_version("gtracer")
except Exception:
    __version__ = "0.0.0-dev"

from gtracer.logger import InMemoryHandler, _configure
from gtracer.tracer import (
    VALID_CHILDREN,
    SpanContext,
    Tracer,
    _agent_name as agent_name,
    _span_id as span_id,
    _span_name as span_name,
    _tags as tags,
    _trace_id as trace_id,
    _trace_metadata as trace_metadata,
    configure,
    serialize_lc_messages,
    tracer,
)
from gtracer.callbacks import TracingCallbackHandler, tracing_handler

# Auto-configure at import time.
# Set GTRACER_ENABLED=false to silence all span output in production.
# Tracing mechanics (spans, callbacks, token counts) remain active either way.
_enabled = os.environ.get("GTRACER_ENABLED", "true").strip().lower() != "false"
_log_to_file = os.environ.get("GTRACER_LOG_TO_FILE", "false").strip().lower() == "true"
_configure(_enabled, _log_to_file)

__all__ = [
    # Core tracer
    "Tracer",
    "SpanContext",
    "tracer",
    "configure",
    "serialize_lc_messages",
    "VALID_CHILDREN",
    # ContextVars — async-safe, propagate through LangGraph tasks automatically
    "trace_id",    # ContextVar[str | None] — current trace/session ID
    "span_id",     # ContextVar[str | None] — current active span ID
    "span_name",   # ContextVar[str | None] — current active span name
    "tags",        # ContextVar[dict] — inherited tags propagated down span tree
    "agent_name",     # ContextVar[str] — current agent name
    "trace_metadata", # ContextVar[dict] — trace-level metadata set by start_trace()
    # Callbacks
    "TracingCallbackHandler",
    "tracing_handler",
    # Testing
    "InMemoryHandler",
    # Version
    "__version__",
]
