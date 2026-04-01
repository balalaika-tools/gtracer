# Changelog

## 0.2.1 — 2026-04-01

### Changes

- Extended span taxonomy: `run` now allows direct `llm_call` children (`run → llm_call`), enabling preprocessing steps, classifiers, and other single LLM calls that are not full agents to be traced without wrapping them in a fake `agent` span.
- Fixed delta tracking for direct LLM calls under `run`: concurrent tasks sharing the same `_span_id` no longer collide in `_msg_counts`. Delta tracking now only activates inside `agent` spans; direct LLM calls always log the full message list.

---

## 0.2.0 — 2026-03-31

### Breaking changes

- `SpanContext.set()` renamed to `set_attr()`.
- `serialise_lc_messages()` renamed to `serialize_lc_messages()` (American spelling).

### New features

- **`@tracer.tool()` decorator** — instruments any sync or async function as a `tool_call` span. Auto-resolves the parent `llm_call`, captures input arguments, and records the return value. Stackable above `@tool` with or without parentheses.
- **`tracer.start_trace(metadata=...)` parameter** — key/value pairs merged into every span in the trace as top-level JSON fields.
- **`configure(enabled, log_to_file, extra_children)`** — `configure()` now controls stdout output, file logging, and the span taxonomy in addition to truncation limit.
- **`GTRACER_ENABLED` env var** — set to `false` to suppress all stdout output. Tracing mechanics remain active.
- **`GTRACER_LOG_TO_FILE` env var** — set to `true` to write spans to a timestamped `.jsonl` file under `logs/`. Works independently of `GTRACER_ENABLED`.
- **`InMemoryHandler`** — testing utility. Attach to the `"gtracer"` logger to collect span records in memory instead of parsing JSON from stdout.
- **`trace_metadata` ContextVar** — exported alongside `span_id`, `trace_id`, `span_name`, `tags`, `agent_name`.
- **`tracer.current_span_id()` / `tracer.current_trace_id()`** — convenience accessors for the ContextVar values.
- **`TracingCallbackHandler(max_traces, stale_span_seconds)`** — bounded memory: evicts stale open spans and trims tracking dicts when `max_traces` is exceeded.
- **`TracingCallbackHandler.reset()`** — clears all internal state. Useful between test runs or batch jobs.
- **`__version__`** — package version now exported from `gtracer`.
- Auto-configures at import time — no setup call needed.

### Changes

- TRACE log level changed from **25 to 5** (below DEBUG). Spans no longer appear in INFO-level log captures.
- `span_id` is now a full **32-character** hex UUID (was 8 characters).
- `span.end` only re-emits attributes set after `span.start` (via `set_attr()` or `close_span(end_attrs=...)`). Large start-time payloads such as `delta` are no longer duplicated on close.
- Delta baseline is now tracked per `(trace_id, agent_span_id)` instead of per `trace_id` alone — concurrent or nested sub-agents no longer corrupt each other's message deltas.
- Hierarchy validation is skipped when `parent_span_id` is passed explicitly (cross-task bridging).
- A `warnings.warn` is issued when `span("run")` is opened without a prior `start_trace()` call.
- ContextVars are now exported under public names (`span_id`, `trace_id`, etc.) instead of private `_span_id`, `_trace_id`.
- Fixed a missing lock around `_open_spans` writes in `on_chat_model_start`.

---

## 0.1.0 — 2026-03-30

Initial release.

- `Tracer` class with `span()` context manager and `open_span()` / `close_span()` open/close API
- `SpanContext` with `.set_attr()` and `.fail()` methods
- `TracingCallbackHandler` for automatic LangChain/LangGraph LLM call instrumentation
- Span taxonomy enforcement: `run → agent → llm_call → tool_call → agent`
- Cross-task bridging via `_last_llm_spans` dict for causal tool → llm_call parenting
- Sub-agent delta reset: message history baseline resets when a sub-agent starts with a shorter history
- `configure()` for truncation limit
- Custom TRACE log level (25) emitted via standard Python `logging`
