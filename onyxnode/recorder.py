"""The SDK: record an agent run, or replay one from a recording.

    import onyxnode

    rec = onyxnode.Recorder("support-agent")
    with rec.start(input={"ticket": 4411}) as run:
        order = run.tool("lookup_order", lookup_order, order_id="A-17")
        reply = run.llm("my-model", call_model, messages)
        run.decision("route", choice="refund", reason="inside the 30 day window")
        run.finish(output=reply)

Recording never raises into the agent: if a sink fails, the event is dropped
with one warning and the agent carries on.
"""
import atexit
import collections
import json
import os
import queue
import threading
import time
import urllib.request
import warnings

from . import events as E

__all__ = ["Recorder", "Run", "LLMResult", "ReplayMiss", "ReplayedError",
           "MemorySink", "JsonlSink", "HttpSink"]


class LLMResult:
    """Return this from the function passed to run.llm() to report usage.

    run.llm() records the usage and hands `output` back to the agent.
    """

    def __init__(self, output, tokens_in=0, tokens_out=0, cost_usd=None):
        self.output = output
        self.tokens_in = tokens_in
        self.tokens_out = tokens_out
        self.cost_usd = cost_usd


class ReplayMiss(Exception):
    """During replay the agent made a call the recording has no answer for."""


class ReplayedError(Exception):
    """During replay, the recorded call had failed; this carries its error text."""


# ---------------------------------------------------------------- sinks

class MemorySink:
    def __init__(self):
        self.events = []

    def emit(self, event):
        self.events.append(event)

    def flush(self):
        pass


class JsonlSink:
    """Appends each event to <directory>/<run_id>.jsonl."""

    def __init__(self, directory):
        self.directory = str(directory)
        self._lock = threading.Lock()

    def emit(self, event):
        with self._lock:
            os.makedirs(self.directory, exist_ok=True)
            path = os.path.join(self.directory, event["run_id"] + ".jsonl")
            with open(path, "a", encoding="utf-8", newline="\n") as f:
                f.write(E.dump_event(event) + "\n")

    def flush(self):
        pass


class HttpSink:
    """Ships events to a collector from a background thread.

    If the collector cannot be reached, events are written to a local JSONL
    spool instead (the collector imports that folder when it is its own runs
    folder, and `onyx import` loads it otherwise), so nothing is lost.
    """

    def __init__(self, url, spool_dir=None, batch=200, timeout=5):
        self.url = url.rstrip("/") + "/api/ingest"
        self.batch = batch
        self.timeout = timeout
        self._spool = JsonlSink(spool_dir or default_runs_dir())
        self._warned = False
        self._q = queue.Queue()
        threading.Thread(target=self._loop, name="onyx-http-sink", daemon=True).start()
        atexit.register(self.flush)

    def emit(self, event):
        self._q.put(event)

    def flush(self, timeout=10):
        done = threading.Event()
        self._q.put(done)
        done.wait(timeout)

    def _loop(self):
        while True:
            items = [self._q.get()]
            try:
                while len(items) < self.batch:
                    items.append(self._q.get_nowait())
            except queue.Empty:
                pass
            batch = [i for i in items if not isinstance(i, threading.Event)]
            if batch:
                self._send(batch)
            for i in items:
                if isinstance(i, threading.Event):
                    i.set()

    def _send(self, batch):
        body = json.dumps({"events": batch}, default=repr).encode("utf-8")
        req = urllib.request.Request(self.url, data=body, method="POST",
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                resp.read()
        except Exception as e:
            if not self._warned:
                self._warned = True
                warnings.warn("onyxnode: collector unreachable (%s); spooling to %s"
                              % (e, self._spool.directory), stacklevel=2)
            for ev in batch:
                try:
                    self._spool.emit(ev)
                except Exception:
                    pass


class _MultiSink:
    def __init__(self, sinks):
        self.sinks = sinks

    def emit(self, event):
        for s in self.sinks:
            s.emit(event)

    def flush(self):
        for s in self.sinks:
            s.flush()


def default_home():
    return os.environ.get("ONYX_DIR") or ".onyx"


def default_runs_dir():
    return os.path.join(default_home(), "runs")


def make_sink(sink=None):
    """None -> $ONYX_URL collector, else JSONL files in $ONYX_DIR/runs (.onyx/runs)."""
    if sink is None:
        url = os.environ.get("ONYX_URL")
        return HttpSink(url) if url else JsonlSink(default_runs_dir())
    if isinstance(sink, (list, tuple)):
        return _MultiSink([make_sink(s) for s in sink])
    if isinstance(sink, (str, os.PathLike)):
        text = os.fspath(sink)
        if text.startswith(("http://", "https://")):
            return HttpSink(text)
        return JsonlSink(text)
    return sink


# ---------------------------------------------------------------- recorder

class Recorder:
    """Creates runs. One Recorder per agent is typical.

    name     default run name
    sink     where events go: None (see make_sink), a collector URL, a folder,
             a sink object, or a list of those
    pricing  {model: (usd_per_million_input_tokens, usd_per_million_output_tokens)}
             used to fill in cost when an LLM call reports tokens but no cost.
             No prices are built in; they change too often to hard-code.
    redact   function applied to every payload before it is stored. The default
             masks values under keys like api_key / password / authorization.
             Pass None to store payloads as-is.
    """

    def __init__(self, name="agent", sink=None, pricing=None,
                 redact=E.redact_keys, max_chars=200_000):
        self.name = name
        self.sink = make_sink(sink)
        self.pricing = dict(pricing or {})
        self.redact = redact
        self.max_chars = max_chars

    def start(self, input=None, name=None, tags=None, meta=None, run_id=None):
        """Begin recording a run. Use the result as a context manager."""
        return Run(self, name or self.name, input, tags, meta, run_id)

    def replay(self, source, on_miss="error", ignore=()):
        """Begin a run that answers tool and LLM calls from a recording.

        source   a list of events, or the path of a run's .jsonl file
        on_miss  "error": raise ReplayMiss when the agent makes a call the
                 recording has no answer for; "live": make the real call
        ignore   dotted paths left out when matching calls, e.g. "input.now"

        The replayed run is recorded like any other (with meta.replay_of set),
        so it can be diffed against the original.
        """
        if on_miss not in ("error", "live"):
            raise ValueError("on_miss must be 'error' or 'live'")
        baseline = E.read_jsonl(source) if isinstance(source, (str, os.PathLike)) else list(source)
        start = next((e for e in baseline if e.get("kind") == E.RUN_START), {})
        meta = {"replay_of": start.get("run_id") or (baseline[0]["run_id"] if baseline else None)}
        run = Run(self, start.get("name") or self.name, start.get("input"),
                  (start.get("meta") or {}).get("tags"), meta, None)
        run._load_tape(baseline, on_miss, tuple(ignore))
        return run

    def flush(self):
        try:
            self.sink.flush()
        except Exception:
            pass


class Run:
    """One recorded run. Create it with Recorder.start() or Recorder.replay()."""

    def __init__(self, recorder, name, input, tags, meta, run_id):
        self._rec = recorder
        self.run_id = run_id or E.new_run_id()
        self.name = name
        self.replaying = False
        self.finished = False
        self._tape = None
        self._on_miss = "error"
        self._ignore = ()
        self._seq = 0
        self._lock = threading.Lock()
        self._warned = False
        self.input = self._prep(input)
        m = dict(meta or {})
        if tags:
            m["tags"] = list(tags)
        self._emit(E.RUN_START, name, input=self.input, meta=m or None)

    # -- context manager

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if not self.finished:
            if exc is not None:
                self.finish(status="error", error=_err_text(exc))
            else:
                self.finish()
        return False

    # -- recording calls

    def tool(self, name, fn, *args, **kwargs):
        """Call fn(*args, **kwargs) and record it as a tool call."""
        return self._call(E.TOOL_CALL, name, fn, args, kwargs)

    def llm(self, model, fn, *args, **kwargs):
        """Call fn(*args, **kwargs) and record it as an LLM call.

        fn may return an LLMResult to report tokens and cost.
        """
        return self._call(E.LLM_CALL, model, fn, args, kwargs)

    async def atool(self, name, fn, *args, **kwargs):
        """Async version of tool(): fn is awaited."""
        return await self._acall(E.TOOL_CALL, name, fn, args, kwargs)

    async def allm(self, model, fn, *args, **kwargs):
        """Async version of llm(): fn is awaited."""
        return await self._acall(E.LLM_CALL, model, fn, args, kwargs)

    def decision(self, name, choice, reason=None, options=None, **data):
        """Record a decision the agent made. Returns `choice`."""
        inp = dict(data)
        if options is not None:
            inp["options"] = options
        out = {"choice": choice}
        if reason is not None:
            out["reason"] = reason
        self._emit(E.DECISION, name, input=self._prep(inp) if inp else None,
                   output=self._prep(out))
        return choice

    def note(self, text, **data):
        """Record a free-form note. Notes are ignored by the diff engine."""
        self._emit(E.NOTE, text, input=self._prep(data) if data else None)

    def record(self, kind, name, input=None, output=None, error=None, duration_ms=None,
               tokens_in=None, tokens_out=None, cost_usd=None, meta=None, ts=None):
        """Record a step by hand, for calls that cannot be wrapped.

        Steps recorded this way are not answered from the recording in replay.
        """
        if kind == E.LLM_CALL and cost_usd is None:
            cost_usd = self._price(name, tokens_in, tokens_out)
        self._emit(kind, name, input=self._prep(input), output=self._prep(output),
                   error=error, duration_ms=duration_ms, tokens_in=tokens_in,
                   tokens_out=tokens_out, cost_usd=cost_usd, meta=meta, ts=ts)

    def finish(self, output=None, status=None, error=None):
        """End the run. Called for you when the `with` block exits."""
        if self.finished:
            return output
        self.finished = True
        status = status or ("error" if error else "ok")
        self._emit(E.RUN_END, self.name, output=self._prep(output), error=error,
                   status=status)
        self._rec.flush()
        return output

    # -- internals

    def _call(self, kind, name, fn, args, kwargs):
        inp = self._prep(_call_input(args, kwargs))
        if self._tape is not None:
            hit, value = self._from_tape(kind, name, inp)
            if hit:
                return value
        ts, t0 = time.time(), time.perf_counter()
        try:
            result = fn(*args, **kwargs)
        except BaseException as e:
            self._done(kind, name, inp, None, e, ts, t0)
            raise
        return self._done(kind, name, inp, result, None, ts, t0)

    async def _acall(self, kind, name, fn, args, kwargs):
        inp = self._prep(_call_input(args, kwargs))
        if self._tape is not None:
            hit, value = self._from_tape(kind, name, inp)
            if hit:
                return value
        ts, t0 = time.time(), time.perf_counter()
        try:
            result = await fn(*args, **kwargs)
        except BaseException as e:
            self._done(kind, name, inp, None, e, ts, t0)
            raise
        return self._done(kind, name, inp, result, None, ts, t0)

    def _done(self, kind, name, inp, result, exc, ts, t0):
        duration = round((time.perf_counter() - t0) * 1000, 3)
        tokens_in = tokens_out = cost = None
        value = result
        if isinstance(result, LLMResult):
            value = result.output
            tokens_in, tokens_out, cost = result.tokens_in, result.tokens_out, result.cost_usd
        if kind == E.LLM_CALL and cost is None:
            cost = self._price(name, tokens_in, tokens_out)
        self._emit(kind, name, input=inp, output=self._prep(value),
                   error=_err_text(exc) if exc is not None else None, duration_ms=duration,
                   tokens_in=tokens_in, tokens_out=tokens_out, cost_usd=cost, ts=ts)
        return value

    def _price(self, model, tokens_in, tokens_out):
        price = self._rec.pricing.get(model)
        if not price or (tokens_in is None and tokens_out is None):
            return None
        return round(((tokens_in or 0) * price[0] + (tokens_out or 0) * price[1]) / 1e6, 8)

    def _prep(self, payload):
        if payload is None:
            return None
        try:
            data = E.to_jsonable(payload, self._rec.max_chars)
            if self._rec.redact:
                data = self._rec.redact(data)
            return data
        except Exception as e:
            return "<unrecordable: %s>" % e

    def _emit(self, kind, name, ts=None, **fields):
        try:
            with self._lock:
                event = {"v": E.SCHEMA_VERSION, "run_id": self.run_id, "seq": self._seq,
                         "ts": round(ts if ts is not None else time.time(), 6),
                         "kind": kind, "name": str(name)}
                event.update(fields)
                self._seq += 1
                self._rec.sink.emit(event)
        except Exception as e:
            if not self._warned:
                self._warned = True
                warnings.warn("onyxnode: failed to record event (%s)" % e, stacklevel=3)

    # -- replay

    def _load_tape(self, baseline, on_miss, ignore):
        self.replaying = True
        self._on_miss = on_miss
        self._ignore = ignore
        self._tape = collections.defaultdict(collections.deque)
        for ev in baseline:
            if ev.get("kind") in (E.TOOL_CALL, E.LLM_CALL):
                self._tape[self._tape_key(ev["kind"], ev.get("name"), ev.get("input"))].append(ev)

    def _tape_key(self, kind, name, inp):
        view = E.strip_paths({"input": inp}, self._ignore)["input"]
        return (kind, str(name), E.digest(view))

    def _from_tape(self, kind, name, inp):
        """Returns (True, recorded_output), raises, or (False, None) to go live."""
        with self._lock:
            waiting = self._tape.get(self._tape_key(kind, name, inp))
            recorded = waiting.popleft() if waiting else None
        if recorded is None:
            if self._on_miss == "live":
                return False, None
            msg = "replay miss: the recording has no %s '%s' with this input" % (kind, name)
            self._emit(kind, name, input=inp, error=msg, meta={"replay_miss": True})
            raise ReplayMiss(msg)
        self._emit(kind, name, input=inp, output=recorded.get("output"),
                   error=recorded.get("error"), duration_ms=0, meta={"replayed": True})
        if recorded.get("error"):
            raise ReplayedError(recorded["error"])
        return True, recorded.get("output")


def _call_input(args, kwargs):
    if not args:
        return dict(kwargs)
    out = {"args": list(args)}
    if kwargs:
        out["kwargs"] = dict(kwargs)
    return out


def _err_text(exc):
    text = str(exc)
    return "%s: %s" % (type(exc).__name__, text) if text else type(exc).__name__
