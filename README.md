<div align="center">

# gtracer

**Lightweight span-based tracing for LangChain and LangGraph agents.**

Emits structured JSONL spans via Python's standard `logging` —
no new infrastructure, no agents, no dashboards required.

[![Python](https://img.shields.io/badge/Python-3.13+-3776AB.svg?logo=python&logoColor=white)](https://www.python.org)
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
└── agent "main"
    ├── llm_call seq:1  ← tokens, model, message delta, latency
    │   └── tool_call search_database  ← input, result, duration
    ├── llm_call seq:2
    │   └── tool_call calculator
    └── llm_call seq:3  ← final answer
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
            agent_span.set("output_type", type(result).__name__)
            return result
```

Every LLM call is now automatically captured — tokens, model, latency, message deltas.

### 3. Instrument your tools

```python
from langchain_core.tools import tool
from gtracer import tracer, tracing_handler, _span_id

@tool
async def search_database(query: str) -> str:
    """Run a database query."""
    llm_parent = tracing_handler.last_llm_span(_span_id.get())
    with tracer.span("tool_call",
                     attrs={"tool": "search_database", "input": {"query": query}},
                     parent_span_id=llm_parent) as span:
        result = await execute_query(query)
        span.set("result", result)
        return result
```

---

## Silence in Production

Set the env var — no code change needed:

```bash
GTRACER_ENABLED=false
```

Tracing mechanics (spans, callbacks, token counts) stay fully active. Only stdout output is suppressed.

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
```

---

## Supported Patterns

| Pattern | Description |
|---|---|
| `create_agent` | ReAct loop with tool use and structured output |
| `StateGraph` | LangGraph graphs with custom nodes |
| Nested agents | Agent-as-a-tool with causal span parenting |
| Deep Agents | LangChain `create_deep_agent` with sub-agents |
| Parallel tools | Concurrent tool calls under the same `llm_call` parent |

See [docs/comprehensive.md](docs/comprehensive.md) for full integration patterns, API reference, and gotchas.

---

## Requirements

[![Python](https://img.shields.io/badge/Python-3.13+-3776AB.svg?logo=python&logoColor=white)](https://www.python.org)
[![langchain-core](https://img.shields.io/badge/langchain--core-%3E%3D1.2.23-1C3C3C.svg?logo=chainlink&logoColor=white)](https://pypi.org/project/langchain-core/)
