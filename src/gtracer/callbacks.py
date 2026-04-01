"""LangChain callback handler — instruments every LLM call as a tracer span.

on_chat_model_start  →  opens an llm_call span (captures message delta + model)
on_llm_end           →  closes the span (captures tokens, stop_reason, response)
on_llm_error         →  errors the span

Spans are stored in _open_spans keyed by LangChain's run_id (UUID) so concurrent
or sequential LLM calls within the same session are tracked independently.

Message delta:
  Each on_chat_model_start receives the full accumulated message history.
  We track how many messages were seen on the previous call (per trace_id)
  and log only the new messages as `delta`, keeping payloads lean.

  Sub-agent reset: if a sub-agent starts with a shorter message history than
  the main agent accumulated, prev_count is reset to 0 so the sub-agent's
  first call logs all of its messages correctly.
"""

from __future__ import annotations

import threading
import time
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from gtracer.tracer import SpanContext, _agent_name, _span_id, _span_name, _trace_id, serialize_lc_messages, tracer


class TracingCallbackHandler(BaseCallbackHandler):
    """LangChain callback handler that instruments every LLM call as a tracer span.

    Automatically captures model name, token counts (input/output/cache),
    message deltas, latency, and stop reason for each LLM call.

    Attach to any LangChain/LangGraph agent::

        from gtracer import tracing_handler

        result = await agent.ainvoke(
            {"messages": [{"role": "user", "content": "..."}]},
            config={"callbacks": [tracing_handler]},
        )

    Use the module-level ``tracing_handler`` singleton or instantiate your own.
    Call ``last_llm_span()`` inside tools to establish causal span parenting.
    """

    def __init__(self, max_traces: int = 10_000, stale_span_seconds: float = 600.0) -> None:
        self._max_traces = max_traces
        self._stale_span_seconds = stale_span_seconds
        self._open_spans:     dict[str, SpanContext] = {}   # run_id → open span
        self._msg_counts:     dict[tuple[str, str], int] = {}  # (trace_key, agent_key) → prev msg count
        self._seq_counter:    dict[str, int]         = {}   # trace_key → seq
        self._last_llm_spans: dict[str, str]         = {}   # agent_span_id → llm_span_id
        self._lock = threading.Lock()

    def last_llm_span(self, agent_span_id: str | None = None) -> str | None:
        """Return the span_id of the most recent llm_call under the given agent span.

        Required for tool tracing — LangGraph runs tool nodes in separate asyncio
        Tasks where ContextVar mutations from ``on_llm_end`` are invisible.  This
        method bridges that gap via a shared dict on the handler instance.

        Args:
            agent_span_id: The agent's span ID.  Defaults to the current
                           ``span_id`` ContextVar if not provided.

        Returns:
            The span_id of the last completed llm_call under this agent span,
            or ``None`` if no LLM call has completed yet.

        Example::

            from gtracer import tracer, tracing_handler

            @tool
            async def my_tool(param: str) -> str:
                llm_parent = tracing_handler.last_llm_span()
                with tracer.span("tool_call",
                                 attrs={"tool": "my_tool", "input": {"param": param}},
                                 parent_span_id=llm_parent) as span:
                    result = await do_work(param)
                    span.set_attr("result", result)
                    return result
        """
        if agent_span_id is None:
            agent_span_id = _span_id.get()
        if agent_span_id is None:
            return None
        with self._lock:
            return self._last_llm_spans.get(agent_span_id)

    def reset(self) -> None:
        """Clear all internal state accumulated across traces.

        Call this between test runs or long-running batch jobs to release
        memory held by the internal tracking dicts.  Not needed for normal
        single-session or short-lived service usage.
        """
        with self._lock:
            self._open_spans.clear()
            self._msg_counts.clear()
            self._seq_counter.clear()
            self._last_llm_spans.clear()

    def _evict_oldest(self) -> None:
        """Trim tracking dicts to prevent unbounded growth. Caller holds _lock."""
        # Evict stale open spans first (likely orphaned by cancellation).
        if self._open_spans:
            now = time.monotonic()
            stale_ids = [
                rid for rid, ctx in self._open_spans.items()
                if now - ctx._start > self._stale_span_seconds
            ]
            for rid in stale_ids:
                self._open_spans.pop(rid, None)

        # FIFO safety valve for remaining dicts.
        for d in (self._open_spans, self._seq_counter, self._msg_counts, self._last_llm_spans):
            while len(d) > self._max_traces:
                d.pop(next(iter(d)))

    # ------------------------------------------------------------------ #
    # LLM start — opens llm_call span                                      #
    # ------------------------------------------------------------------ #

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages:   list[list[BaseMessage]],
        *,
        run_id:        UUID,
        parent_run_id: UUID | None = None,
        tags:          list[str] | None = None,
        **kwargs:      Any,
    ) -> None:
        all_msgs: list[BaseMessage] = messages[0] if messages else []

        # Key by trace_id so seq increments across all LLM calls in the session.
        # LangGraph creates a new parent_run_id per step, so keying by parent_run_id
        # would reset seq to 1 on every call.
        trace_key = _trace_id.get() or str(parent_run_id) or str(run_id)
        # Key msg_counts by (trace_key, agent_key) so each agent tracks its
        # own message history independently — prevents concurrent or nested
        # sub-agents from corrupting each other's delta baselines.
        agent_key = _span_id.get() or ""
        msg_key = (trace_key, agent_key)

        # Delta tracking is only meaningful inside an agent loop where the
        # same LLM is called repeatedly with accumulating message history.
        # Direct LLM calls under "run" (preprocessing, classifiers) are
        # one-shot — always log the full message list to avoid collisions
        # when concurrent tasks share the same _span_id.
        inside_agent = _span_name.get() == "agent"

        with self._lock:
            if (len(self._open_spans) > self._max_traces
                    or len(self._msg_counts) > self._max_traces):
                self._evict_oldest()
            self._seq_counter[trace_key] = self._seq_counter.get(trace_key, 0) + 1
            seq        = self._seq_counter[trace_key]
            if inside_agent:
                prev_count = self._msg_counts.get(msg_key, 0)
                # If all_msgs is shorter than prev_count, we're in a sub-agent
                # that has its own fresh message history — reset the delta baseline.
                if len(all_msgs) < prev_count:
                    prev_count = 0
                self._msg_counts[msg_key] = len(all_msgs)
            else:
                prev_count = 0

        delta = serialize_lc_messages(all_msgs[prev_count:])

        # Model name from serialized kwargs — varies by provider
        kw    = serialized.get("kwargs", {})
        model = kw.get("model_id") or kw.get("model") or kw.get("model_name") or ""

        ctx = tracer.open_span("llm_call", attrs={
            "agent":         _agent_name.get(),
            "model":         model,
            "seq":           seq,
            "delta":         delta,
            "message_count": len(all_msgs),
        })
        with self._lock:
            self._open_spans[str(run_id)] = ctx

    # ------------------------------------------------------------------ #
    # LLM end — closes llm_call span                                       #
    # ------------------------------------------------------------------ #

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        with self._lock:
            ctx = self._open_spans.pop(str(run_id), None)
        if ctx is None:
            return

        for gen_list in response.generations:
            for gen in gen_list:
                if not isinstance(gen, ChatGeneration):
                    continue

                msg       = gen.message
                usage     = getattr(msg, "usage_metadata",   None) or {}
                resp_meta = getattr(msg, "response_metadata", None) or {}

                model_id = (
                    resp_meta.get("model_id")
                    or resp_meta.get("model_name")
                    or resp_meta.get("model")
                    or ctx.attrs.get("model", "")
                )

                tokens: dict[str, Any] = {
                    "input":  usage.get("input_tokens",  0),
                    "output": usage.get("output_tokens", 0),
                    "total":  usage.get("total_tokens",  0),
                }
                for k, v in (usage.get("input_token_details") or {}).items():
                    if v:
                        tokens[f"input_{k}"] = v
                for k, v in (usage.get("output_token_details") or {}).items():
                    if v:
                        tokens[f"output_{k}"] = v

                tracer.close_span(ctx, end_attrs={
                    "model":       model_id,
                    "stop_reason": resp_meta.get("stop_reason", ""),
                    "tokens":      tokens,
                    "response":    serialize_lc_messages([msg]),
                })
                if ctx.parent_span_id:
                    with self._lock:
                        self._last_llm_spans[ctx.parent_span_id] = ctx.span_id
                return  # one generation per call is the norm

        tracer.close_span(ctx)  # no ChatGeneration found — close without end attrs
        if ctx.parent_span_id:
            with self._lock:
                self._last_llm_spans[ctx.parent_span_id] = ctx.span_id

    # ------------------------------------------------------------------ #
    # LLM error — errors llm_call span                                     #
    # ------------------------------------------------------------------ #

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        with self._lock:
            ctx = self._open_spans.pop(str(run_id), None)
        if ctx is None:
            return
        tracer.error_span(
            ctx,
            error if isinstance(error, Exception) else Exception(str(error)),
        )


# Singleton — import and pass via config={"callbacks": [tracing_handler]}
tracing_handler = TracingCallbackHandler()
