<div align="center">

<pre>
 ██████╗ ████████╗██████╗  █████╗  ██████╗███████╗██████╗
██╔════╝ ╚══██╔══╝██╔══██╗██╔══██╗██╔════╝██╔════╝██╔══██╗
██║  ███╗   ██║   ██████╔╝███████║██║     █████╗  ██████╔╝
██║   ██║   ██║   ██╔══██╗██╔══██║██║     ██╔══╝  ██╔══██╗
╚██████╔╝   ██║   ██║  ██║██║  ██║╚██████╗███████╗██║  ██║
 ╚═════╝    ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝╚══════╝╚═╝  ╚═╝
</pre>

**Lightweight span-based tracing for LangChain and LangGraph agents.**

Emits structured JSONL spans via Python's standard `logging` —
no new infrastructure, no agents, no dashboards required.

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB.svg?logo=python&logoColor=white)](https://www.python.org)
[![PyPI](https://img.shields.io/pypi/v/gtracer.svg?color=blue)](https://pypi.org/project/gtracer/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![LangChain](https://img.shields.io/badge/LangChain-compatible-1C3C3C.svg?logo=chainlink&logoColor=white)](https://python.langchain.com)
[![LangGraph](https://img.shields.io/badge/LangGraph-compatible-1C3C3C.svg?logo=chainlink&logoColor=white)](https://langchain-ai.github.io/langgraph/)

</div>

---

## What it does

Every time an LLM call happens inside your agent, gtracer captures it as a structured span and writes it to stdout as JSON:

```
run
├── llm_call              ← direct LLM calls (preprocessing, classifiers)
└── agent "main"
    ├── llm_call seq:1    ← tokens, model, message delta, latency
    │   └── tool_call search_database  ← input, result, duration
    ├── llm_call seq:2
    │   └── tool_call calculator
    └── llm_call seq:3    ← final answer
```

Works with CloudWatch, Datadog, or any stdout log consumer. Zero configuration — spans are live the moment you import the package.

---

## Install

```bash
pip install gtracer
# or
uv add gtracer
```

---

## Quick Start

### 1. Import and go

```python
import gtracer  # spans are live immediately — nothing else needed
```

gtracer auto-configures at import time. It attaches its own JSON handler with `propagate=False` — it **never touches your app's root logger**, no double-emission, no interference.

### 2. Wrap your agent

```python
from gtracer import tracer, tracing_handler

async def run(session_id: str, user_input: str):
    tracer.start_trace(session_id)

    with tracer.span("run", tags={"session_id": session_id}):
        with tracer.span("agent", attrs={"agent": "main"}) as agent_span:
            result = await my_agent.ainvoke(
                {"messages": [{"role": "user", "content": user_input}]},
                config={"callbacks": [tracing_handler]},
            )
            agent_span.set_attr("output_type", type(result).__name__)
            return result
```

Every LLM call is now automatically captured — tokens, model, latency, message deltas.

### 3. Instrument your tools

```python
from langchain_core.tools import tool
from gtracer import tracer

@tool
@tracer.tool()
async def search_database(query: str) -> str:
    """Run a database query."""
    return await execute_query(query)
```

The `@tracer.tool()` decorator automatically captures input arguments, records the return value, and resolves the parent `llm_call` span. For full control, use the context manager directly:

```python
from gtracer import tracer, tracing_handler, span_id

@tool
async def search_database(query: str) -> str:
    """Run a database query."""
    llm_parent = tracing_handler.last_llm_span()
    with tracer.span("tool_call",
                     attrs={"tool": "search_database", "input": {"query": query}},
                     parent_span_id=llm_parent) as span:
        result = await execute_query(query)
        span.set_attr("result", result)
        return result
```

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `GTRACER_ENABLED` | `true` | Set to `false` to suppress all stdout output. Tracing mechanics stay fully active. |
| `GTRACER_LOG_TO_FILE` | `false` | Set to `true` to write spans to a file on disk. |

### Silence in Production

```bash
GTRACER_ENABLED=false python your_app.py
```

Spans are still created and timed — only output is suppressed.

### Save Logs Locally

> ⚠️ **Local scripts only.** `GTRACER_LOG_TO_FILE` is intended for running Python scripts directly on your machine. Do not use it in Docker, ECS, Lambda, or any containerised/cloud environment — those environments have no persistent local filesystem and stdout is already captured by their log infrastructure.

```bash
GTRACER_LOG_TO_FILE=true python your_app.py
```

Creates `logs/gtracer_<YYYYMMDD_HHMMSS>.jsonl` in the directory where the script is run. The `logs/` folder is created automatically if it doesn't exist. Spans are written to both the file and stdout.

Both variables can be combined:

```bash
GTRACER_ENABLED=false GTRACER_LOG_TO_FILE=true python your_app.py
# silences console output, still writes to file
```

---

## Span Schema

Every span event is a flat JSON object on a single line:

```json
{
  "ts": "2026-03-30T10:00:00",
  "level": "TRACE",
  "event": "span.end",
  "span_name": "llm_call",
  "trace_id": "abc123",
  "span_id": "a1b2c3d4",
  "parent_span_id": "e5f6a7b8",
  "status": "ok",
  "duration_ms": 1823,
  "attrs": {
    "agent": "main",
    "model": "claude-sonnet-4-6",
    "seq": 2,
    "tokens": {
      "input": 461,
      "output": 277,
      "total": 738,
      "input_cache_read": 15541
    },
    "stop_reason": "tool_use"
  }
}
```

Each `llm_call` span captures:

| Field | Description |
|---|---|
| `attrs.tokens` | input, output, total, cache_read, cache_creation |
| `attrs.model` | exact model ID from the provider response |
| `attrs.delta` | new messages added since the previous LLM call |
| `duration_ms` | wall-clock latency in milliseconds |
| `attrs.stop_reason` | `tool_use`, `end_turn`, etc. |

---

## Configuration

```python
from gtracer import configure

configure(truncation_limit=50_000)  # max chars for message content fields (default)

# You can also control output via code instead of env vars:
configure(enabled=False, log_to_file=True)

# Add custom span types to the taxonomy:
configure(extra_children={
    "agent": {"retrieval", "embedding"},
    "retrieval": {"llm_call"},
})
```

### Trace-level metadata

Attach metadata to every span in a trace:

```python
tracer.start_trace(session_id, metadata={"user_id": "u42", "env": "prod"})
```

### Testing with InMemoryHandler

Collect spans in-memory for assertions instead of parsing JSON from stdout:

```python
from gtracer import InMemoryHandler
import logging

handler = InMemoryHandler()
logging.getLogger("gtracer").addHandler(handler)

# ... run your agent ...

assert len(handler.records) == 6
assert handler.records[0].span_name == "run"
handler.clear()
```

---

## API Reference

### `tracer` — the singleton you import

| Method | Description |
|---|---|
| `tracer.start_trace(trace_id, metadata=None)` | Set the current session/trace ID and optional trace-level metadata. Call once per invocation before any spans. |
| `tracer.span(name, attrs, tags, parent_span_id)` | Context manager — opens a span, yields `SpanContext`, auto-closes on exit. |
| `@tracer.tool(name=None)` | Decorator — instruments a function as a `tool_call` span. Auto-captures input/result and resolves parent. |
| `tracer.current_span_id()` | Return the current active span ID. |
| `tracer.current_trace_id()` | Return the current trace/session ID. |
| `tracer.open_span(name, attrs, tags, parent_span_id)` | Open a span without a context manager (for LangChain callbacks). |
| `tracer.close_span(ctx, end_attrs)` | Close a span opened with `open_span()`. |
| `tracer.error_span(ctx, exc)` | Mark a span opened with `open_span()` as failed due to an exception. |

### `SpanContext` — the object yielded by `span()`

| Method | Description |
|---|---|
| `span.set_attr(key, value)` | Accumulate an end-time attribute. Flushed into `attrs` on `span.end`. |
| `span.fail(reason="")` | Mark as a business-level failure. Emits `span.end status:error` (no exception needed). |

### `tracing_handler` — the LangChain callback

| Method | Description |
|---|---|
| `tracing_handler.last_llm_span(agent_span_id=None)` | Returns the `span_id` of the most recent `llm_call` under the given agent span. Defaults to the current `span_id` ContextVar. |

Attach to any LangChain/LangGraph agent: `config={"callbacks": [tracing_handler]}`.

### ContextVars — for tool integration

| Name | Type | Purpose |
|---|---|---|
| `span_id` | `ContextVar[str \| None]` | Current active span ID. Pass to `last_llm_span()` inside tools. |
| `trace_id` | `ContextVar[str \| None]` | Current session ID set by `start_trace()`. |
| `span_name` | `ContextVar[str \| None]` | Current active span name. |
| `tags` | `ContextVar[dict]` | Inherited tags, merged down the span tree. |
| `agent_name` | `ContextVar[str]` | Current agent name, set when opening an `agent` span. |
| `trace_metadata` | `ContextVar[dict]` | Trace-level metadata set by `start_trace()`. Merged into every span. |

### `configure(truncation_limit, enabled, log_to_file, extra_children)`

Call once at startup. `truncation_limit` (default 50,000) limits characters kept in `delta`, `response`, and `result` fields. `enabled` and `log_to_file` (both default `None` = preserve current state) override `GTRACER_ENABLED` and `GTRACER_LOG_TO_FILE` respectively. `extra_children` extends the span taxonomy with custom span types.

### `serialize_lc_messages(messages)`

Converts a list of LangChain `BaseMessage` objects to JSON-serializable dicts. Respects the truncation limit.

---

## Supported Patterns

| Pattern | Description |
|---|---|
| `create_agent` | ReAct loop with tool use and structured output |
| `StateGraph` | LangGraph graphs with custom nodes |
| Nested agents | Agent-as-a-tool with causal span parenting |
| Deep Agents | LangChain `create_deep_agent` with sub-agents |
| Parallel tools | Concurrent tool calls under the same `llm_call` parent |

See [docs/documentation.md](https://github.com/balalaika-tools/gtracer/blob/main/docs/documentation.md) for full integration patterns, API reference, and gotchas.

---

## Known Limitations

**Sync tools on Python 3.10–3.11:** gtracer uses `ContextVar` to track the active span and trace. On Python <3.12, `loop.run_in_executor()` does not propagate context to worker threads. If LangGraph runs a **synchronous** tool from an async graph, the tool's span will be disconnected from the span tree (orphaned `parent_span_id`). Span lifecycle (open/close/tokens) still works correctly — only parent-child linking is affected.

**Workarounds:**
- Use **async tools** (always works — `asyncio.create_task()` copies context on all Python versions)
- Upgrade to **Python 3.12+** (where `run_in_executor()` propagates context)
- Pass `parent_span_id` explicitly in sync tools via `tracing_handler.last_llm_span(agent_span_id)`

---

## Requirements

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB.svg?logo=python&logoColor=white)](https://www.python.org)
[![langchain-core](https://img.shields.io/badge/langchain--core-%3E%3D1.2.23-1C3C3C.svg?logo=chainlink&logoColor=white)](https://pypi.org/project/langchain-core/)
