"""Event schema and the helpers every other module shares.

A run is an ordered list of events. Each event is a flat JSON object:

    v            schema version (1)
    run_id       id of the run it belongs to
    seq          position in the run, starting at 0
    ts           start time, epoch seconds
    kind         run_start | llm_call | tool_call | decision | note | run_end
    name         tool name, model name, decision label, or run name
    input        JSON payload going in  (run_start: the run's input)
    output       JSON payload coming out (run_end: the run's final output)
    error        error text, if the step failed
    duration_ms  how long the step took
    tokens_in / tokens_out / cost_usd   for llm_call (cost may be set on any step)
    status       run_end only: ok | error
    meta         free-form extras (run_start carries tags and replay_of here)

Fields that are None are left out when an event is written. One run is stored
as one JSONL file, one event per line; that is the interchange format used by
the SDK, the collector, the suite and the CLI.
"""
import dataclasses
import hashlib
import json
import os
import time

SCHEMA_VERSION = 1

RUN_START = "run_start"
LLM_CALL = "llm_call"
TOOL_CALL = "tool_call"
DECISION = "decision"
NOTE = "note"
RUN_END = "run_end"
KINDS = (RUN_START, LLM_CALL, TOOL_CALL, DECISION, NOTE, RUN_END)

REDACTED = "[redacted]"
SENSITIVE_KEYS = ("api_key", "apikey", "authorization", "password", "passwd",
                  "secret", "token", "access_key", "private_key", "cookie")
# Keys that contain "token" but are counts, not credentials.
_NOT_SENSITIVE = ("tokens", "token_count", "max_tokens", "tokens_in", "tokens_out")


def new_run_id():
    """Time-sortable id, safe as a file name: 20261116-221501-a1b2c3."""
    return time.strftime("%Y%m%d-%H%M%S", time.gmtime()) + "-" + os.urandom(3).hex()


def to_jsonable(obj, max_chars=200_000, _depth=0):
    """Turn any Python value into plain JSON data, never raising."""
    if obj is None or isinstance(obj, (bool, int, float)):
        if isinstance(obj, float) and (obj != obj or obj in (float("inf"), float("-inf"))):
            return repr(obj)
        return obj
    if isinstance(obj, str):
        if len(obj) > max_chars:
            return obj[:max_chars] + "...[truncated %d chars]" % (len(obj) - max_chars)
        return obj
    if _depth > 40:
        return "<max depth>"
    nxt = _depth + 1
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v, max_chars, nxt) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v, max_chars, nxt) for v in obj]
    if isinstance(obj, (set, frozenset)):
        return sorted((to_jsonable(v, max_chars, nxt) for v in obj), key=canon)
    if isinstance(obj, (bytes, bytearray)):
        return "<%d bytes>" % len(obj)
    try:
        if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
            return to_jsonable(dataclasses.asdict(obj), max_chars, nxt)
        for attr in ("model_dump", "to_dict", "dict"):  # pydantic and friends
            fn = getattr(obj, attr, None)
            if callable(fn):
                return to_jsonable(fn(), max_chars, nxt)
        if hasattr(obj, "__dict__") and vars(obj):
            return to_jsonable({k: v for k, v in vars(obj).items()
                                if not k.startswith("_")}, max_chars, nxt)
    except Exception:
        pass
    try:
        return repr(obj)[:max_chars]
    except Exception:
        return "<unrepresentable %s>" % type(obj).__name__


def redact_keys(obj):
    """Mask values whose key looks like a credential (api_key, password, ...)."""
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            lk = k.lower()
            if any(s in lk for s in SENSITIVE_KEYS) and lk not in _NOT_SENSITIVE \
                    and isinstance(v, (str, bytes)):
                out[k] = REDACTED
            else:
                out[k] = redact_keys(v)
        return out
    if isinstance(obj, list):
        return [redact_keys(v) for v in obj]
    return obj


def canon(obj):
    """Canonical JSON text: equal data always gives equal text."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      default=repr)


def digest(obj):
    return hashlib.sha1(canon(obj).encode("utf-8", "replace")).hexdigest()[:16]


def strip_paths(event, paths):
    """Copy of the event's input/output/error with the given dotted paths removed.

    Paths are relative to the event, e.g. "input.timestamp" or
    "output.items.*.id" ("*" matches every key or list index). Used to keep
    volatile fields (request ids, clocks) out of comparisons.
    """
    view = {"input": event.get("input"), "output": event.get("output"),
            "error": event.get("error")}
    if not paths:
        return view
    view = json.loads(canon(view))  # deep copy
    for path in paths:
        _strip(view, path.split("."))
    return view


def _strip(node, parts):
    head, rest = parts[0], parts[1:]
    if isinstance(node, dict):
        keys = list(node) if head == "*" else ([head] if head in node else [])
        for k in keys:
            if rest:
                _strip(node[k], rest)
            else:
                del node[k]
    elif isinstance(node, list):
        if head == "*":
            idx = range(len(node))
        elif head.isdigit() and int(head) < len(node):
            idx = [int(head)]
        else:
            idx = []
        for i in idx:
            if rest:
                _strip(node[i], rest)
            else:
                node[i] = None


def dump_event(event):
    return json.dumps({k: v for k, v in event.items() if v is not None},
                      separators=(",", ":"), ensure_ascii=False, default=repr)


def read_jsonl(path):
    """Read one run file. A torn last line (run still being written) is skipped."""
    events = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except ValueError:
                continue
            if isinstance(ev, dict) and "run_id" in ev and "seq" in ev:
                events.append(ev)
    events.sort(key=lambda e: e["seq"])
    return events


def write_jsonl(path, events):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for ev in events:
            f.write(dump_event(ev) + "\n")


def summarize(events):
    """Run-level summary computed from a run's events."""
    s = {"run_id": None, "name": None, "status": "running", "started_at": None,
         "ended_at": None, "tags": [], "meta": {}, "input": None, "output": None,
         "error": None, "steps": 0, "llm_calls": 0, "tool_calls": 0, "decisions": 0,
         "errors": 0, "tokens_in": 0, "tokens_out": 0, "cost_usd": 0.0}
    last = None
    for ev in events:
        kind = ev.get("kind")
        s["run_id"] = s["run_id"] or ev.get("run_id")
        ts = ev.get("ts")
        if ts is not None:
            end = ts + (ev.get("duration_ms") or 0) / 1000.0
            if s["started_at"] is None or ts < s["started_at"]:
                s["started_at"] = ts
            if last is None or end > last:
                last = end
        if kind == RUN_START:
            s["name"] = ev.get("name")
            s["input"] = ev.get("input")
            s["meta"] = ev.get("meta") or {}
            s["tags"] = s["meta"].get("tags") or []
        elif kind == RUN_END:
            s["status"] = ev.get("status") or ("error" if ev.get("error") else "ok")
            s["output"] = ev.get("output")
            s["error"] = ev.get("error")
        else:
            s["steps"] += 1
            if kind == LLM_CALL:
                s["llm_calls"] += 1
            elif kind == TOOL_CALL:
                s["tool_calls"] += 1
            elif kind == DECISION:
                s["decisions"] += 1
            if ev.get("error"):
                s["errors"] += 1
        s["tokens_in"] += ev.get("tokens_in") or 0
        s["tokens_out"] += ev.get("tokens_out") or 0
        s["cost_usd"] += ev.get("cost_usd") or 0.0
    s["ended_at"] = last
    s["cost_usd"] = round(s["cost_usd"], 6)
    return s
