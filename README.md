# gtracer

Lightweight span-based tracing for LangChain and LangGraph agents.

Emits structured JSONL spans via Python's standard `logging` — no new infrastructure required. Works with CloudWatch, Datadog, stdout, or any log handler.

## Install

```bash
pip install gtracer
# or
uv add gtracer
```

## Quick Start

### 1. Enable trace output

```python
import logging
logging.getLogger("gtracer").setLevel(25)  # TRACE level
```

Add a JSON formatter if you want structured output (recommended):

```bash
pip install python-json-logger
```

```python
from pythonjsonlogger import jsonlogger

handler = logging.StreamHandler()
handler.setFormatter(jsonlogger.JsonFormatter())
logging.getLogger("gtracer").addHandler(handler)
```

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

That's it. Every LLM call is now automatically traced with tokens, model, latency, and message deltas.

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

## What You Get

Each trace run produces a tree of JSONL span events:

```
run  (tags: session_id=...)
└── agent "main"
    ├── llm_call seq:1  (tokens, model, delta messages)
    │   └── tool_call search_database  (input, result, duration)
    ├── llm_call seq:2
    │   └── tool_call calculator
    └── llm_call seq:3  (final answer)
```

Each `llm_call` span captures:
- **Tokens**: input, output, total, cache_read, cache_creation
- **Model**: exact model ID from the provider response
- **Delta**: new messages added since the previous LLM call
- **Latency**: wall-clock duration in milliseconds
- **Stop reason**: `tool_use`, `end_turn`, etc.

## Configuration

```python
from gtracer import configure

configure(truncation_limit=50_000)  # max chars for message content fields (default)
```

## Log Schema

Every span event is a flat JSON object:

```json
{
  "level": "TRACE",
  "event": "span.end",
  "span_name": "llm_call",
  "trace_id": "abc123",
  "span_id": "a1b2c3d4",
  "parent_span_id": "e5f6a7b8",
  "session_id": "...",
  "status": "ok",
  "duration_ms": 1823,
  "attrs": {
    "agent": "main",
    "model": "claude-sonnet-4-6",
    "seq": 2,
    "tokens": {"input": 461, "output": 277, "total": 738, "input_cache_read": 15541},
    "stop_reason": "tool_use"
  }
}
```

## Supported Patterns

- `create_agent` ReAct loop (tool use + structured output)
- `StateGraph` with custom nodes
- Nested sub-agents (agent-as-a-tool)
- LangChain Deep Agents (`create_deep_agent`)
- Parallel tool execution

See [docs/comprehensive.md](docs/comprehensive.md) for full integration patterns, API reference, and gotchas.

## Requirements

- Python 3.11+
- `langchain-core >= 0.3`
