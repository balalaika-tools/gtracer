# Changelog

## 0.1.0 — 2026-03-30

Initial release.

- `Tracer` class with `span()` context manager and `open_span()` / `close_span()` open/close API
- `SpanContext` with `.set()` and `.fail()` methods
- `TracingCallbackHandler` for automatic LangChain/LangGraph LLM call instrumentation
- Span taxonomy enforcement: `run → agent → llm_call → tool_call → agent`
- Cross-task bridging via `_last_llm_spans` dict for causal tool → llm_call parenting
- Sub-agent delta reset: message history baseline resets when a sub-agent starts with a shorter history
- `configure()` for truncation limit
- Custom TRACE log level (25) emitted via standard Python `logging`
