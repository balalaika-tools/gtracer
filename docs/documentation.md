# gtracer — Comprehensive Documentation

A lightweight, generic tracing framework for LLM-based agentic applications.
Produces structured JSONL logs that any log consumer (CloudWatch, Datadog,
a Streamlit monitoring app, etc.) can parse into per-session traces, span
waterfalls, and cost analytics.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Core Components](#2-core-components)
   - [ContextVars](#21-contextvars)
   - [SpanContext](#22-spancontext)
   - [Tracer Class](#23-tracer-class)
   - [Span Taxonomy](#24-span-taxonomy)
   - [Callback Handler](#25-callback-handler)
   - [Logger Integration](#26-logger-integration)
3. [How It Works — Full Data Flow](#3-how-it-works--full-data-flow)
   - [Span Lifecycle](#31-span-lifecycle)
   - [Parent Resolution](#32-parent-resolution)
   - [Cross-Task Bridging](#33-cross-task-bridging)
   - [Truncation](#34-truncation)
   - [Error Semantics](#35-error-semantics)
4. [Log Schema Reference](#4-log-schema-reference)
5. [API Reference](#5-api-reference)
6. [Integration Patterns](#6-integration-patterns)
   - [Pattern 1: LangChain `create_agent` with tools](#pattern-1-langchain-create_agent-with-tools)
   - [Pattern 2: LangGraph `StateGraph` with custom nodes](#pattern-2-langgraph-stategraph-with-custom-nodes)
   - [Pattern 3: Nested agents — manual agent-as-a-tool](#pattern-3-nested-agents--manual-agent-as-a-tool)
   - [Pattern 4: LangChain Deep Agents (`create_deep_agent`)](#pattern-4-langchain-deep-agents-create_deep_agent)
   - [Pattern 5: Parallel tool execution](#pattern-5-parallel-tool-execution)
   - [Pattern 6: Adding a new tool](#pattern-6-adding-a-new-tool)
7. [Gotchas](#7-gotchas)
8. [Configuration](#8-configuration)

---

## 1. Architecture Overview

```
                                    +-----------------------+
                                    |     Application       |
                                    |  (entrypoint)         |
                                    +-----------+-----------+
                                                |
                                    tracer.start_trace(session_id)
                                                |
                            +-------------------v--------------------+
                            |          tracer.span("run")            |
                            |    tags: {user_id: "..."}              |
                            +-------------------+--------------------+
                                                |
                            +-------------------v--------------------+
                            |        tracer.span("agent")            |
                            |    attrs: {agent: "main"}              |
                            |    sets _agent_name ContextVar         |
                            +---+---------------------------+--------+
                                |                           |
                   LangGraph Task 1              LangGraph Task 2
                   (LLM node)                    (Tool node)
                                |                           |
                   +------------v-----------+  +------------v-----------+
                   | on_chat_model_start    |  | @tool function         |
                   |   tracer.open_span     |  |   tracing_handler      |
                   |     ("llm_call")       |  |     .last_llm_span()   |
                   | on_llm_end             |  |   tracer.span          |
                   |   tracer.close_span    |  |     ("tool_call",      |
                   |   stores in            |  |      parent_span_id=   |
                   |   _last_llm_spans{}    |  |      llm_parent)       |
                   +------------------------+  +------------------------+
                                                         |
                                                         v
                                              (optional sub-agent)
                                              tracer.span("agent")
                                                         |
                                              +----------v-----------+
                                              | on_chat_model_start  |
                                              |   ("llm_call")       |
                                              | tool_call → ...      |
                                              +----------------------+
```

**Key design decisions:**

1. **JSONL via Python logging** — no new infrastructure. Spans are emitted via `logger.trace()` at a custom TRACE level (25). Any `logging.Handler` captures them automatically.

2. **Two APIs** — `span()` (context manager, updates ContextVars) for code you control, and `open_span()`/`close_span()` for LangChain callbacks where start/end happen in separate methods.

3. **ContextVar-based hierarchy** — async-safe, per-task isolation. Each `asyncio.Task` inherits ContextVars via `copy_context()` at creation time.

4. **Instance-level dict for cross-task bridging** — LangGraph runs LLM nodes and Tool nodes as separate tasks. ContextVar mutations in one are invisible to the other. The callback handler's `_last_llm_spans` dict (shared object state) bridges this gap.

5. **Causal, not temporal, nesting** — `tool_call` is a child of the `llm_call` that triggered it, even though the LLM call is already closed when the tool runs.

---

## 2. Core Components

### 2.1 ContextVars

```
src/gtracer/tracer.py
```

Five ContextVars provide async-safe span context propagation:

| ContextVar | Type | Purpose |
|---|---|---|
| `_span_id` | `str \| None` | Current active span ID. Used by `_make_span()` to resolve `parent_span_id`. |
| `_span_name` | `str \| None` | Current active span name. Used for hierarchy validation. |
| `_trace_id` | `str \| None` | Session identifier. Set once by `start_trace()`, inherited by all descendant spans. |
| `_tags` | `dict` | Inherited tags. Merged down the tree — child spans automatically carry all ancestor tags. |
| `_agent_name` | `str` | Current agent name. Set when opening an `agent` span. Read by the callback handler to label `llm_call` spans. |

**Propagation rules:**

- ContextVars are per-task in async code and per-thread in threaded code.
- When LangGraph creates a new `asyncio.Task` (for an LLM node or Tool node), it uses `copy_context()` — the child task gets a **snapshot** of the parent's ContextVar state at task creation time.
- Mutations inside the child task are invisible to the parent and to sibling tasks.
- `_span_id` is set by `tracer.span("agent")` **before** LangGraph creates any tasks. So both the LLM node task and Tool node task inherit the same `_span_id` value (the agent's span_id).

### 2.2 SpanContext

A mutable container created fresh for each open span:

```python
@dataclass
class SpanContext:
    span_id:        str                    # 8-char hex UUID
    name:           str                    # "run", "agent", "llm_call", "tool_call"
    parent_span_id: str | None             # Links to parent
    trace_id:       str | None             # Session identifier
    tags:           dict[str, Any]         # Inherited tags at span open time
    attrs:          dict[str, Any]         # Mutable, accumulated via .set()
    _start:         float                  # Monotonic timer at creation
    _failed:        bool                   # Business-level failure flag
    _fail_reason:   str                    # Reason string for business failure
```

**Methods:**

| Method | Purpose |
|---|---|
| `set(key, value)` | Accumulate end-time data. All `.set()` calls are flushed into `attrs` on `span.end`. |
| `fail(reason)` | Mark as business-level failure. Emits `span.end status:error` (not `span.error`). |

**Ownership:** One `SpanContext` per span, owned by exactly one thread/task. Never pass it to another task and call `.set()` concurrently — `attrs` dict is not protected by a lock.

### 2.3 Tracer Class

```python
class Tracer:
    def __init__(self, logger: logging.Logger)
    def start_trace(self, trace_id: str) -> None
    def span(self, name, attrs, tags, parent_span_id) -> Generator[SpanContext]
    def open_span(self, name, attrs, tags, parent_span_id) -> SpanContext
    def close_span(self, ctx, end_attrs) -> None
    def error_span(self, ctx, exc) -> None
```

Module-level singleton:

```python
from gtracer import tracer   # always import this
```

### 2.4 Span Taxonomy

Four span types with strict hierarchy enforcement:

```
null → run → agent → llm_call → tool_call → agent (sub-agent)
```

```python
VALID_CHILDREN: dict[str | None, set[str]] = {
    None:        {"run"},
    "run":       {"agent"},
    "agent":     {"llm_call"},
    "llm_call":  {"tool_call"},
    "tool_call": {"agent"},   # sub-agent nested inside a tool
}
```

On violation: logs `WARNING`, never raises. The span is still emitted with its actual `parent_span_id`.

**Causal hierarchy visualised:**

```
run (s1)
└── agent "main" (s2, parent=s1)
    ├── llm_call seq:1 (s3, parent=s2)  ← decides to call a tool
    │   └── tool_call (s4, parent=s3)   ← triggered by llm_call seq:1
    │       └── agent "fixer" (s5, parent=s4)  ← sub-agent inside tool
    │           ├── llm_call seq:1 (s6, parent=s5)
    │           │   └── tool_call (s7, parent=s6)
    │           └── llm_call seq:2 (s8, parent=s5)
    ├── llm_call seq:2 (s9, parent=s2)  ← processes tool result
    │   └── tool_call (s10, parent=s9)
    └── llm_call seq:3 (s11, parent=s2) ← final answer
```

### 2.5 Callback Handler

```
src/gtracer/callbacks.py
```

Bridges LangChain's callback system to the tracer. Instruments every LLM call automatically.

```python
class TracingCallbackHandler(BaseCallbackHandler):
    _open_spans:     dict[str, SpanContext]  # run_id → open span
    _msg_counts:     dict[str, int]          # trace_key → prev msg count
    _seq_counter:    dict[str, int]          # trace_key → seq number
    _last_llm_spans: dict[str, str]          # agent_span_id → llm_span_id
    _lock:           threading.Lock          # protects all dicts
```

**Lifecycle hooks:**

| Hook | Action |
|---|---|
| `on_chat_model_start` | Opens `llm_call` span via `tracer.open_span()`. Captures model, seq, delta messages, message count. Reads `_agent_name.get()` for the agent label. |
| `on_llm_end` | Closes `llm_call` span via `tracer.close_span()`. Captures tokens, stop_reason, response. Stores `_last_llm_spans[parent_span_id] = span_id`. |
| `on_llm_error` | Errors `llm_call` span via `tracer.error_span()`. |

**Delta tracking:** Each `on_chat_model_start` receives the full accumulated message history. The handler tracks how many messages were seen on the previous call (keyed by `trace_id`) and logs only the new messages as `delta`.

**Sub-agent reset:** If a sub-agent starts with a shorter message history than the current `prev_count` (because it has a fresh conversation), the baseline resets to 0 so all of the sub-agent's initial messages are captured correctly.

**Seq counter:** Global per session (keyed by `trace_id`, not `parent_run_id`). LangGraph creates a new `parent_run_id` per step, so keying by it would reset `seq` to 1 on every call.

**Singleton:**

```python
from gtracer import tracing_handler
# Pass via config={"callbacks": [tracing_handler]}
```

### 2.6 Logger Integration

gtracer emits spans via a custom TRACE log level (25) using Python's standard `logging` module. The module-level singleton writes to the `"gtracer"` logger:

```python
tracer = Tracer(logging.getLogger("gtracer"))
```

**Enabling span output:**

Spans are only emitted when the `"gtracer"` logger (or its parent) is configured at level 25 or below. Two approaches:

```python
# Option A — set the level directly
import logging
logging.getLogger("gtracer").setLevel(25)

# Option B — if you have a custom TRACE level in your app
logging.getLogger("gtracer").setLevel(logging.getLevelName("TRACE"))
```

**JSON output:** The tracer emits via `logger.trace(msg, extra=payload)`. Python's logging flattens `extra={}` fields onto the `LogRecord`. No third-party formatter required — if your app already has a JSON formatter on the root logger, spans are serialised by it automatically.

**CloudWatch / stdout:** Ensure the `"gtracer"` logger propagates to the root handler (the default), or attach your own handler directly to `logging.getLogger("gtracer")`.

**Emit pipeline:**

```
tracer._emit(event, ctx, ...)
  └── self._log.trace(msg, extra=payload)
       └── Python logging flattens extra{} onto LogRecord
            └── Your JSON formatter → one JSONL line to stdout / CloudWatch
```

---

## 3. How It Works — Full Data Flow

### 3.1 Span Lifecycle

**Context manager API (`span()`):**

```
span() called
  ├── _make_span() creates SpanContext
  │   ├── Resolves parent (ContextVar or explicit override)
  │   ├── Validates hierarchy (VALID_CHILDREN)
  │   ├── Emits span.start
  │   └── Returns SpanContext
  ├── Updates ContextVars (_span_id, _span_name, _tags, _agent_name)
  ├── Yields SpanContext to caller
  │   └── Caller calls span.set() / span.fail() during execution
  ├── On clean exit:
  │   └── _emit_end() → span.end with duration + status + flushed attrs
  ├── On exception:
  │   └── _emit_error() → span.error with error + error_type + duration
  │   └── Re-raises the exception
  └── Resets ContextVars via token.reset()
```

**Open/close API (`open_span()` / `close_span()`):**

```
open_span() called
  ├── _make_span() creates SpanContext
  │   ├── Captures parent from ContextVar state at call time
  │   ├── Does NOT update ContextVars
  │   └── Emits span.start
  └── Returns SpanContext (caller stores it)

close_span(ctx, end_attrs) called
  ├── Merges end_attrs into ctx.attrs
  └── _emit_end() → span.end

error_span(ctx, exc) called
  └── _emit_error() → span.error
```

### 3.2 Parent Resolution

In `_make_span()`:

```python
if parent_span_id is not None:
    # Explicit causal override
    resolved_parent_id   = parent_span_id
    resolved_parent_name = "llm_call"   # assumed logical parent
else:
    # ContextVar-based
    resolved_parent_id   = _span_id.get()
    resolved_parent_name = _span_name.get()
```

**When to use explicit `parent_span_id`:**
- Tool spans — to causally link `tool_call` → `llm_call` across task boundaries
- Sub-agent spans — to parent `agent` under `tool_call` in parallel tool scenarios

**When ContextVar-based resolution works:**
- `run` → `agent` nesting (same task)
- `agent` span opening (before any tasks are created)
- Any spans within the same `asyncio.Task`

### 3.3 Cross-Task Bridging

**The problem:** LangGraph runs each node (LLM, Tool) as a separate `asyncio.Task` with `copy_context()`. The LLM node's `on_llm_end` sets data that the Tool node needs — but ContextVar mutations in one task are invisible to the other.

**The solution:** A shared dict on the callback handler instance.

```
┌──────────────────────────────────────────────────────┐
│  tracer.span("agent")                                │
│  Sets: _span_id = "agent_s2"                         │
│  Both child tasks inherit this via copy_context()    │
│                                                      │
│  ┌─────────────────────┐  ┌────────────────────────┐ │
│  │ Task 1 (LLM node)   │  │ Task 2 (Tool node)     │ │
│  │                     │  │                         │ │
│  │ on_llm_end:         │  │ @tool function:         │ │
│  │   _last_llm_spans   │  │   agent_id =            │ │
│  │     ["agent_s2"]    │──│     _span_id.get()      │ │
│  │     = "llm_s3"      │  │     → "agent_s2"        │ │
│  │                     │  │   llm_parent =           │ │
│  │   (dict mutation is │  │     handler              │ │
│  │    visible to all)  │  │       .last_llm_span(    │ │
│  │                     │  │         "agent_s2")      │ │
│  │                     │  │     → "llm_s3"           │ │
│  └─────────────────────┘  └────────────────────────┘ │
└──────────────────────────────────────────────────────┘
```

**Why this works:**
1. `_span_id` is set by `tracer.span("agent")` **before** LangGraph spawns any tasks
2. Both tasks inherit the same `_span_id` value via `copy_context()`
3. `_last_llm_spans` is a regular dict on the handler **instance** — shared object state, not a ContextVar
4. Task 1 writes to it; Task 2 reads from it — both reference the same object
5. The `_lock` ensures thread-safe reads/writes

### 3.4 Truncation

Applied before emit in `_truncate_attrs()`. All limits are controlled by `configure(truncation_limit=...)` (default 50,000 chars).

| Field pattern | Treatment |
|---|---|
| `delta`, `response` | Truncate each message's `content` string within the list |
| `result` | Truncate the entire string |
| Any other string | Truncate if over limit |
| Non-string values | Passed through unchanged |

Truncated values get ` ...[truncated]` appended.

### 3.5 Error Semantics

Two distinct error signals:

| Signal | Event emitted | When to use |
|---|---|---|
| **Python exception propagates** out of `span()` | `span.error status:error` | DB timeout, network error, unhandled crash |
| **`span.fail(reason)`** called, no exception | `span.end status:error` | LLM refusal, fixer exhausted retries, no structured response |

Rule: `span.error` means the span did not complete normally. `span.end status:error` means it completed but the outcome was bad.

**Retry semantics:** Retries are sibling spans under the same parent, both with the same `seq` value. First has `status:error`, second has `status:ok`. No explicit retry field needed.

---

## 4. Log Schema Reference

Every trace line is a single JSON object. Fields are split into two tiers:

### Mandatory fields (always present)

| Field | Type | Description |
|---|---|---|
| `ts` | ISO8601 | Timestamp (added by your log formatter) |
| `level` | `"TRACE"` | Always TRACE |
| `event` | enum | `span.start`, `span.end`, `span.error` |
| `span_name` | enum | `run`, `agent`, `llm_call`, `tool_call` |
| `trace_id` | string | One per session |
| `span_id` | string | Unique per span (8-char hex) |
| `parent_span_id` | string \| null | Links child to parent. Null on root span. |

### Conditional fields (on span.end / span.error)

| Field | Type | Description |
|---|---|---|
| `duration_ms` | int | Wall-clock time in milliseconds |
| `status` | `"ok"` \| `"error"` | Outcome |

### Tags (top-level, filterable)

Promoted to top-level JSON fields. Inherited by child spans.

```json
{"user_id": "u123", "session_id": "abc-def"}
```

### `attrs` (detail payload)

Open dict. Content varies by span type.

#### `llm_call` span.start attrs

```json
{
  "agent": "main",
  "model": "anthropic.claude-sonnet-4-6",
  "seq": 3,
  "delta": [{"type": "tool", "tool_call_id": "tc_abc", "content": "[{...}]"}],
  "message_count": 12
}
```

#### `llm_call` span.end attrs

```json
{
  "model": "claude-sonnet-4-6",
  "stop_reason": "tool_use",
  "tokens": {
    "input": 461,
    "output": 197,
    "total": 658,
    "input_cache_read": 15541,
    "input_cache_creation": 0
  },
  "response": [{"type": "ai", "content": [{"type": "tool_use", "name": "search_db", "input": {}}]}]
}
```

#### `tool_call` span.start attrs

```json
{
  "tool": "search_database",
  "input": {"query": "SELECT ...", "intent": "Fetch user details"}
}
```

#### `tool_call` span.end attrs

```json
{
  "tool": "search_database",
  "result": "[{\"col\": \"value\"}]"
}
```

#### `agent` span.end attrs

```json
{
  "agent": "main",
  "output_type": "AnalysisResult"
}
```

---

## 5. API Reference

### `configure(truncation_limit=50_000)`

Configure global tracer settings. Call once at application startup.

```python
from gtracer import configure

configure(truncation_limit=100_000)
```

### `tracer.start_trace(trace_id: str)`

Sets `_trace_id` ContextVar. Call **once per invocation** in the thread/task entrypoint, before any spans.

```python
tracer.start_trace(session_id)
```

### `tracer.span(name, attrs, tags, parent_span_id)` — context manager

The primary API. Updates ContextVars, so nested spans resolve their parent automatically.

```python
with tracer.span("agent", attrs={"agent": "main"}) as span:
    result = await agent.ainvoke(...)
    span.set("output_type", type(result).__name__)
```

| Parameter | Type | Description |
|---|---|---|
| `name` | `str` | One of: `run`, `agent`, `llm_call`, `tool_call` |
| `attrs` | `dict \| None` | Start-time data. Also accumulates end-time data via `span.set()`. |
| `tags` | `dict \| None` | Promoted to top-level JSON. Inherited by children. |
| `parent_span_id` | `str \| None` | Explicit parent override (bypasses ContextVar). |

### `SpanContext.set(key, value)`

Accumulate end-time data onto the span. Flushed on `span.end`.

### `SpanContext.fail(reason="")`

Mark as business-level failure. Emits `span.end status:error`.

### `tracer.open_span(name, attrs, tags, parent_span_id)` → `SpanContext`

Opens a span **without** updating ContextVars. For LangChain callbacks only.

### `tracer.close_span(ctx, end_attrs=None)`

Closes a span opened with `open_span()`.

### `tracer.error_span(ctx, exc)`

Errors a span opened with `open_span()`.

### `serialise_lc_messages(messages) → list[dict]`

Converts LangChain `BaseMessage` instances to JSON-serialisable dicts. Truncates content.

---

## 6. Integration Patterns

### Pattern 1: LangChain `create_agent` with tools

This is the most common pattern. `create_agent` builds a LangGraph `StateGraph` internally, with a ReAct loop: LLM → tools → LLM → tools → ... → structured output.

**Instrumentation pattern:**

```python
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool

from gtracer import tracer, tracing_handler, _span_id


# ── Define tools with tracing ───────────────────────────────

@tool
async def search_database(query: str) -> str:
    """Search the database with a SQL query."""
    llm_parent = tracing_handler.last_llm_span(_span_id.get())
    with tracer.span("tool_call",
                     attrs={"tool": "search_database", "input": {"query": query}},
                     parent_span_id=llm_parent) as span:
        result = await execute_query(query)
        span.set("result", result)
        return result


@tool
async def calculator(expression: str) -> str:
    """Evaluate a math expression."""
    llm_parent = tracing_handler.last_llm_span(_span_id.get())
    with tracer.span("tool_call",
                     attrs={"tool": "calculator", "input": {"expression": expression}},
                     parent_span_id=llm_parent) as span:
        result = str(eval(expression))
        span.set("result", result)
        return result


# ── Build agent ─────────────────────────────────────────────

llm = init_chat_model("anthropic.claude-sonnet-4-6",
                       model_provider="bedrock_converse",
                       temperature=0.2)

agent = create_agent(
    model=llm,
    tools=[search_database, calculator],
    response_format=MyOutputSchema,
)


# ── Run with tracing ────────────────────────────────────────

async def run_analysis(session_id: str, user_query: str):
    tracer.start_trace(session_id)

    with tracer.span("run", tags={"session_id": session_id}):
        with tracer.span("agent", attrs={"agent": "main"}) as agent_span:
            try:
                result = await agent.ainvoke(
                    {"messages": [{"role": "user", "content": user_query}]},
                    config={"callbacks": [tracing_handler]},
                )
            except Exception as exc:
                agent_span.fail(str(exc))
                return None

            structured = result.get("structured_response")
            if structured is None:
                agent_span.fail("no_structured_response")
                return None

            agent_span.set("output_type", type(structured).__name__)
            return structured
```

**Resulting span tree:**

```
run (tags: session_id=...)
└── agent "main"
    ├── llm_call seq:1           ← model decides to call search_database
    │   └── tool_call search_database
    ├── llm_call seq:2           ← model processes result, calls calculator
    │   └── tool_call calculator
    └── llm_call seq:3           ← model produces final answer
```

---

### Pattern 2: LangGraph `StateGraph` with custom nodes

When you build a graph manually instead of using `create_agent`, you have full control over node execution and tracing.

```python
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, AIMessage
from typing import TypedDict

from gtracer import tracer, tracing_handler, _span_id


class GraphState(TypedDict):
    messages: list
    iteration: int


async def research_node(state: GraphState) -> GraphState:
    """LLM calls inside this node are traced automatically via tracing_handler."""
    llm_parent = tracing_handler.last_llm_span(_span_id.get())
    with tracer.span("tool_call",
                     attrs={"tool": "db_lookup", "input": {"query": "..."}},
                     parent_span_id=llm_parent) as span:
        data = await fetch_from_database(...)
        span.set("result", data)

    return {"messages": state["messages"] + [AIMessage(content=data)],
            "iteration": state["iteration"] + 1}


async def synthesize_node(state: GraphState) -> GraphState:
    response = await llm.ainvoke(state["messages"])
    return {"messages": state["messages"] + [response],
            "iteration": state["iteration"] + 1}


def should_continue(state: GraphState) -> str:
    if state["iteration"] >= 3:
        return "synthesize"
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "research"
    return "synthesize"


graph = StateGraph(GraphState)
graph.add_node("research", research_node)
graph.add_node("synthesize", synthesize_node)
graph.add_edge(START, "research")
graph.add_conditional_edges("research", should_continue,
                            {"research": "research", "synthesize": "synthesize"})
graph.add_edge("synthesize", END)
compiled = graph.compile()


async def run_graph(session_id: str, question: str):
    tracer.start_trace(session_id)

    with tracer.span("run", tags={"question": question[:50]}):
        with tracer.span("agent", attrs={"agent": "researcher"}) as span:
            result = await compiled.ainvoke(
                {"messages": [HumanMessage(content=question)], "iteration": 0},
                config={"callbacks": [tracing_handler]},
            )
            span.set("iterations", result["iteration"])
            return result
```

**Resulting span tree:**

```
run (tags: question="What is...")
└── agent "researcher"
    ├── llm_call seq:1        ← research_node's LLM call
    │   └── tool_call db_lookup
    ├── llm_call seq:2        ← research_node iteration 2
    │   └── tool_call db_lookup
    └── llm_call seq:3        ← synthesize_node's LLM call
```

---

### Pattern 3: Nested agents — manual agent-as-a-tool

A tool inside one agent manually spawns another agent internally.

```python
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool

from gtracer import tracer, tracing_handler, _span_id


@tool
async def _inner_tool(query: str) -> str:
    """Tool available to the inner agent only."""
    llm_parent = tracing_handler.last_llm_span(_span_id.get())
    with tracer.span("tool_call",
                     attrs={"tool": "inner_tool", "input": {"query": query}},
                     parent_span_id=llm_parent) as span:
        result = await do_inner_work(query)
        span.set("result", result)
        return result


inner_agent = create_agent(
    model=inner_llm,
    tools=[_inner_tool],
    response_format=InnerResult,
)


@tool
async def smart_search(query: str, intent: str = "") -> str:
    """Search with automatic error recovery via inner agent."""
    llm_parent = tracing_handler.last_llm_span(_span_id.get())
    with tracer.span("tool_call",
                     attrs={"tool": "smart_search", "input": {"query": query}},
                     parent_span_id=llm_parent) as span:
        try:
            result = await execute_search(query)
            span.set("result", result)
            return result
        except SearchError as exc:
            first_error = str(exc)
            span.set("first_error", first_error)

        result = await _run_inner_agent(
            query=query,
            error=first_error,
            tool_span_id=span.span_id,  # explicit parent for sub-agent
        )
        span.set("result", result)
        span.set("retried", True)
        return result


async def _run_inner_agent(
    query: str,
    error: str,
    tool_span_id: str | None = None,
) -> str:
    # CRITICAL: pass tool_span_id so the sub-agent is parented to this
    # tool_call, not to a stale ContextVar. Required for parallel safety.
    with tracer.span("agent",
                     attrs={"agent": "fixer", "error": error},
                     parent_span_id=tool_span_id):
        result = await inner_agent.ainvoke(
            {"messages": [{"role": "user", "content": f"Fix this: {error}"}]},
            config={"callbacks": [tracing_handler]},
        )

    structured = result.get("structured_response")
    if structured:
        return structured.result
    raise RuntimeError("Fixer agent failed")
```

**Resulting span tree (with fixer triggered):**

```
run
└── agent "main"
    ├── llm_call seq:1
    │   └── tool_call "smart_search"
    │       ├── attrs: {first_error: "...", retried: true}
    │       └── agent "fixer"
    │           ├── llm_call seq:1 (agent: "fixer")
    │           │   └── tool_call "inner_tool"
    │           └── llm_call seq:2 (agent: "fixer")
    └── llm_call seq:2
```

**Why `tool_span_id` is critical for parallel execution:**

If two tools run concurrently via `asyncio.gather`, they share ContextVar state. Without explicit `tool_span_id`, sub-agent `tracer.span("agent")` calls clobber each other's `_span_id`.

---

### Pattern 4: LangChain Deep Agents (`create_deep_agent`)

Deep Agents packages planning, sub-agents, filesystem, and context management into `create_deep_agent()`.

```python
from deepagents import create_deep_agent
from gtracer import tracer, tracing_handler


research_subagent = {
    "name": "researcher",
    "description": "Researches in-depth questions using web search",
    "system_prompt": "You are a thorough researcher. Write findings to files.",
    "tools": [web_search],
    "model": "claude-sonnet-4-6",
}

agent = create_deep_agent(
    model="claude-sonnet-4-6",
    tools=[search_database, calculator],
    subagents=[research_subagent],
    response_format=AnalysisResult,
)


async def run_deep_analysis(session_id: str, question: str):
    tracer.start_trace(session_id)

    with tracer.span("run", tags={"session_id": session_id}):
        with tracer.span("agent", attrs={"agent": "deep_main"}) as agent_span:
            try:
                result = await agent.ainvoke(
                    {"messages": [{"role": "user", "content": question}]},
                    config={"callbacks": [tracing_handler]},
                )
            except Exception as exc:
                agent_span.fail(str(exc))
                return None

            structured = result.get("structured_response")
            if structured is None:
                agent_span.fail("no_structured_response")
                return None

            return structured
```

**Resulting span tree:**

```
run
└── agent "deep_main"
    ├── llm_call seq:1
    │   └── tool_call "task"
    │       └── agent "researcher"
    │           ├── llm_call seq:2
    │           │   └── tool_call "web_search"
    │           └── llm_call seq:3
    ├── llm_call seq:4
    │   └── tool_call "search_database"
    └── llm_call seq:5
```

---

### Pattern 5: Parallel tool execution

When the LLM requests multiple tool calls in one response, LangGraph executes them concurrently. The tracing handles this correctly because:

1. **Causal parenting** — all parallel tools share the same `llm_call` parent
2. **Each tool gets its own `SpanContext`** — no shared mutable state
3. **Sub-agents use explicit `parent_span_id`** — bypasses ContextVar collision

```
agent "main"
├── llm_call seq:1              ← model requests 3 tools simultaneously
│   ├── tool_call "search_a"   ← all three are children of seq:1
│   ├── tool_call "search_b"
│   └── tool_call "calculator"
└── llm_call seq:2              ← model processes all 3 results
```

No special code needed for the standard tool pattern. Only sub-agents spawned inside parallel tools need explicit `tool_span_id`.

---

### Pattern 6: Adding a new tool

Minimal template:

```python
from langchain_core.tools import tool
from gtracer import tracer, tracing_handler, _span_id

@tool
async def my_tool(param: str) -> str:
    """Tool description for the LLM."""
    llm_parent = tracing_handler.last_llm_span(_span_id.get())
    with tracer.span("tool_call",
                     attrs={"tool": "my_tool", "input": {"param": param}},
                     parent_span_id=llm_parent) as span:
        try:
            result = await do_work(param)
            span.set("result", result)
            return result
        except SomeError as exc:
            span.fail(str(exc))
            return f"Error: {exc}"
```

**Checklist:**
- [ ] Import `tracer`, `tracing_handler`, `_span_id` from `gtracer`
- [ ] Look up `llm_parent` via `tracing_handler.last_llm_span(_span_id.get())`
- [ ] Open `tool_call` span with `parent_span_id=llm_parent`
- [ ] Set `attrs.tool` to the tool name
- [ ] Set `attrs.input` with the tool's arguments
- [ ] Call `span.set("result", ...)` on success
- [ ] Call `span.fail(reason)` for business errors (no exception raised)
- [ ] Let Python exceptions propagate naturally (the span context manager handles them)
- [ ] If the tool spawns a sub-agent, pass `tool_span_id=span.span_id`

---

## 7. Gotchas

### 1. `start_trace()` must be called before any spans

If `start_trace()` is never called, `trace_id` is `null` on every span. Call it in the **thread/task entrypoint**, not inside a helper function.

### 2. `open_span()` does not update ContextVars

```python
# WRONG — child resolves to outer agent, not this llm_call
ctx = tracer.open_span("llm_call")
with tracer.span("tool_call"):  # parent = agent, not llm_call
    ...

# CORRECT — use span() when children need to see this as parent
with tracer.span("llm_call"):
    with tracer.span("tool_call"):  # parent = llm_call ✓
        ...
```

`open_span()` is for LangChain callbacks only (start and end in separate methods).

### 3. `span.fail()` vs exceptions

| Scenario | Action | Result |
|---|---|---|
| Exception propagates through `span()` | Let it propagate | `span.error` automatic |
| Operation completed, outcome is bad | Call `span.fail(reason)` | `span.end status:error` |
| Catch exception + return error without `fail()` | Bug | Misleading `status:ok` |

### 4. Tags vs attrs

| | Tags | Attrs |
|---|---|---|
| JSON position | Top-level fields | Nested under `"attrs"` |
| Direct filtering | Yes: `df["user_id"]` | Requires `.apply()` |
| Inherited by children | Yes | No |
| Truncated | No | Yes |

### 5. LangGraph overwrites LangChain tags

LangGraph replaces `tags=["agent"]` with internal step tags like `["seq:step:1"]`. **Never read callback `tags` to identify the agent.**

The tracer uses `_agent_name` ContextVar instead. Set via `tracer.span("agent", attrs={"agent": "my_agent"})`. The `attrs.agent` key is **load-bearing** — it must match the logical name you want in traces.

### 6. Hierarchy validation is a warning, not an error

Invalid nesting logs `WARNING` and continues. The span is still emitted. Fix violations when you see them.

### 7. Log level must be TRACE (25)

Spans use `logger.trace()` (level 25). Ensure the `"gtracer"` logger is set to level 25 or below, or span events won't be written.

### 8. seq counter is global per session

`seq` increments across all LLM calls in the session (keyed by `trace_id`), not per agent. This means:

- Main agent: seq 1, 2, 3
- Fixer agent (spawned during main's seq 3): seq 4, 5
- Main agent resumes: seq 6

This is by design — `seq` shows the total number of LLM calls in the session for cost/performance tracking. To see which agent made a call, use the `agent` attr on the `llm_call` span.

### 9. Retries are sibling spans

No explicit retry field. Two consecutive `llm_call` spans with the same `seq` under the same parent = retry. First `status:error`, second `status:ok`.

### 10. Parallel tools share ContextVar state

`asyncio.gather` runs coroutines in the **same Task**. If two tools run concurrently and both open `tracer.span("agent")`, they clobber each other's `_span_id`. Always pass `tool_span_id` explicitly to sub-agent functions.

---

## 8. Configuration

### `configure()`

```python
from gtracer import configure

configure(truncation_limit=50_000)  # default
```

| Parameter | Default | Description |
|---|---|---|
| `truncation_limit` | `50_000` | Max chars for content fields in span attrs (`delta`, `response`, `result`). Longer strings are truncated with `...[truncated]` appended. |

### Log level

```python
import logging
logging.getLogger("gtracer").setLevel(25)   # enable TRACE output
```

Or via environment variable if your app reads it:

```
LOG_LEVEL=TRACE
```
