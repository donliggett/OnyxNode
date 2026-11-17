# OnyxNode

A black-box flight recorder for AI agent runs.

OnyxNode records every tool call, model call, decision and cost of an agent
run. Then it lets you:

- **replay** a run step by step in a timeline UI;
- **diff** two runs and find the **first divergence**: the exact step where a
  good run and a bad run split, and whether the world answered differently or
  the agent acted differently;
- turn recorded runs into a **regression suite**: your agent is re-run against
  the recorded tool and model answers, and any change in what it does fails
  the case.

It is plain Python (3.9+) with **no dependencies** and no build step: a small
SDK, a collector (HTTP + SQLite), a diff engine, and a single-file web UI.

## Try it in two minutes

From this folder (PowerShell or any shell; `onyx` works in place of
`python -m onyxnode` once installed, see [Install](#install)):

```
python -m onyxnode demo
python -m onyxnode diff latest~1 latest
python -m onyxnode serve
```

`demo` records two runs of a built-in, fully offline refund agent: a good one
and a bad one. `diff` prints where they split:

```
FIRST DIVERGENCE at row 3: Same call, different result from tool `lookup_order`
  Both runs called it with identical input and got different answers. The tool,
  or the data behind it, changed. The agent did nothing different up to here.
  3 row(s) before this match exactly; 6 later row(s) also differ.
    changed   output.delivered_days_ago:  12  ->  45
    changed   output.source:  orders-primary  ->  orders-replica-2
```

`serve` starts the collector; open <http://127.0.0.1:8787> for the timeline.
Tick the two runs in the sidebar and press **Find first divergence**.

## Install

```
pip install -e .
```

That gives you the `onyx` command and `import onyxnode`. Without installing,
run everything as `python -m onyxnode ...` from this folder.

## Record your agent

```python
import onyxnode

rec = onyxnode.Recorder("support-agent")

with rec.start(input={"ticket": 4411}, tags=["prod"]) as run:
    order = run.tool("lookup_order", lookup_order, order_id="A-17")
    reply = run.llm("my-model", call_model, messages)
    run.decision("route", choice="refund", reason="inside the 30 day window",
                 options=["refund", "deny"])
    run.finish(output=reply)
```

| Call | Records |
|---|---|
| `run.tool(name, fn, *args, **kwargs)` | calls `fn`, records input, output, error, duration |
| `run.llm(model, fn, *args, **kwargs)` | same, as a model call, plus tokens and cost |
| `run.atool(...)`, `run.allm(...)` | the same for `async` functions (`await` them) |
| `run.decision(name, choice, reason=, options=)` | a choice the agent made |
| `run.note(text, **data)` | a free-form marker (ignored by the diff) |
| `run.record(kind, name, input=, output=, ...)` | a step you time yourself |
| `run.finish(output=)` | the end of the run (automatic when the `with` block exits) |

**Tokens and cost.** Have the function you pass to `run.llm` return an
`onyxnode.LLMResult`:

```python
def call_model(messages):
    resp = client.messages.create(model="my-model", messages=messages, max_tokens=1024)
    return onyxnode.LLMResult(resp.content[0].text,
                              tokens_in=resp.usage.input_tokens,
                              tokens_out=resp.usage.output_tokens)
```

`run.llm` records the usage and returns the `output` to your agent. Cost is
taken from `LLMResult(cost_usd=...)` if you set it, otherwise computed from
the price table you give the recorder. No prices are built in, because they
go stale:

```python
rec = onyxnode.Recorder("support-agent",
                        pricing={"my-model": (3.00, 15.00)})  # USD per million tokens: input, output
```

**Recording never breaks the agent.** If a sink fails, the event is dropped
with one warning. An exception inside a tool is recorded and re-raised
unchanged.

**Secrets.** Values under keys that look like credentials (`api_key`,
`password`, `authorization`, `token`, ...) are stored as `[redacted]`. Pass
`redact=your_function` to change that, or `redact=None` to store payloads
as-is. Everything else your tools see is stored in plain text, so treat the
`.onyx` folder like a log that may contain customer data.

### Where recordings go

| `Recorder(sink=...)` | Result |
|---|---|
| nothing | `$ONYX_URL` if set (a collector), else JSONL files in `$ONYX_DIR/runs` (default `.onyx/runs`) |
| `"http://127.0.0.1:8787"` | sent to a collector from a background thread; spooled to JSONL if it is down |
| `"some/folder"` | one `<run_id>.jsonl` file per run in that folder |
| `onyxnode.MemorySink()` | kept in memory (`sink.events`), for tests |
| a list of the above | all of them |

The collector imports `.onyx/runs` on its own, so the default needs no setup:
run your agent, run `onyx serve`, and the runs are there.

## Find the first divergence

```
onyx diff <good-run> <bad-run>
onyx diff <good-run> <bad-run> --ignore output.request_id --ignore input.now
onyx diff <good-run> <bad-run> --json
```

A run is named by its id, a unique prefix, `latest`, `latest~1` (the one
before), or the path of a `.jsonl` file. The exit code is 0 when the runs
match and 1 when they diverge.

The diff aligns the two runs step by step and gives every row a status:

| Status | Meaning | What it tells you |
|---|---|---|
| `same` | same step, same input, same output | |
| `output_differs` | same step and input, different output | the tool, the data, or the model answered differently |
| `input_differs` | same step, different input | the agent did something different |
| `only_a` / `only_b` | the step exists in one run only | the agent took another path |

The first row that is not `same` is the first divergence; everything after it
is downstream. Timing, tokens and cost are never compared. `--ignore` drops
volatile fields from the comparison; paths are relative to the event
(`input.x`, `output.items.*.id`, `*` matches any key or index).

In Python: `onyxnode.diff_runs(events_a, events_b, ignore=[...])` returns the
same result as a dict; `onyxnode.first_divergence(...)` returns just the
verdict.

## Replay

Two kinds:

- **In the UI**: open a run and use the replay bar (or the arrow keys and
  space) to step through it. Later steps are dimmed and the running token and
  cost totals follow the cursor.
- **In code**: `rec.replay(source)` starts a run in which `run.tool` and
  `run.llm` return the *recorded* outputs instead of calling your functions.
  Calls are matched by kind, name and input. If the agent makes a call the
  recording has no answer for, `ReplayMiss` is raised (or the real call is
  made, with `on_miss="live"`). The replayed run is itself recorded, so it can
  be diffed against the original. This is what the regression suite uses.

## Regression suite

A suite is a folder of recorded runs (default `onyx_suite/`), one `.jsonl`
per case, committed next to your agent.

```
onyx suite add <run> --name refund-damaged-item
onyx suite list
onyx suite run --agent mypackage.agent:run_agent
```

The agent entry point is a function `agent(run, input)` that does its work
through `run` and returns the final output (it may be `async`). For each case
the suite replays the recording against it and diffs the result:

```
PASS  refund-damaged-item  (10 steps)
FAIL  refund-late-order
      row 4: The recording did tool `check_refund_policy` here; the agent now did not

1 passed, 1 failed, 2 total
```

Exit code 1 on any failure, so it drops into CI as is. Add `-v` for the full
diff, `--save` to record the replayed runs into the store and inspect them in
the UI, `--ignore PATH` for volatile fields, `-k TEXT` to pick cases.

You can also save a case from the UI (**Save as regression case** on a run),
or call it from your own tests:

```python
result = onyxnode.run_case("onyx_suite/refund-damaged-item.jsonl", run_agent)
assert result["passed"], result["diff"]["first_divergence"]["headline"]
```

Try a failing case with the demo agent: `onyx suite add latest~1 --name demo`
then run `onyx suite run --agent onyxnode.demo:agent` once normally and once
with the environment variable `ONYX_DEMO_BUG=1`.

## Collector and UI

```
onyx serve [--host 127.0.0.1] [--port 8787] [--suite-dir onyx_suite]
onyx --home some/folder serve
```

The collector keeps its SQLite store in the home folder (`.onyx/onyx.db`),
imports run files from `.onyx/runs`, accepts events over HTTP, and serves the
UI. API, all JSON:

| | |
|---|---|
| `POST /api/ingest` | `{"events": [...]}` |
| `GET /api/runs?q=&limit=` | run summaries, newest first |
| `GET /api/runs/<id>` | one run with its events |
| `POST /api/runs/<id>/label` | `{"label": "good" \| "bad" \| null}` |
| `GET /api/diff?a=<id>&b=<id>&ignore=p1,p2` | the diff, with both runs' events |
| `GET /api/suite`, `POST /api/suite` | list cases; save a run as a case |

**There is no authentication.** It binds to `127.0.0.1` by default; only
change `--host` on a network you trust.

Other commands: `onyx runs`, `onyx show <run>`, `onyx label <run> good|bad|none`,
`onyx export <run> -o file.jsonl`, `onyx import <files or folders>`.

## Event format

One run is a list of events; on disk, one JSON object per line.

```json
{"v":1,"run_id":"20261116-221501-a1b2c3","seq":3,"ts":1794888901.12,"kind":"tool_call",
 "name":"lookup_order","input":{"order_id":"A-17"},"output":{"amount":89.0},"duration_ms":61.2}
```

`kind` is one of `run_start`, `llm_call`, `tool_call`, `decision`, `note`,
`run_end`. Model calls add `tokens_in`, `tokens_out`, `cost_usd`; failed steps
add `error`; `run_end` adds `status` (`ok` or `error`). The full description
is at the top of `onyxnode/events.py`. Anything that can write these lines can
feed OnyxNode, in any language.

## Limits worth knowing

- A step is written when it **finishes**. If the process is killed mid-call,
  that call is not in the recording (everything before it is).
- Steps are a flat sequence. There is no nesting for sub-agents or parallel
  branches; concurrent calls are recorded in the order they finish.
- Replay hands back the recorded JSON form of an output. Return plain data
  (dicts, lists, strings) from the functions you wrap, not SDK response
  objects, so live and replayed runs see the same shape.
- A recorded failure is replayed as `onyxnode.ReplayedError` with the original
  message, not as the original exception class.
- Streaming model output is recorded as the final result, not chunk by chunk.

## Development

```
python -m unittest discover -s tests -v
```

Layout:

```
onyxnode/events.py     event schema, canonical JSON, redaction, run summary
onyxnode/recorder.py   the SDK: Recorder, Run, sinks, replay
onyxnode/diff.py       alignment, first divergence, field diff, text report
onyxnode/suite.py      regression suite
onyxnode/store.py      SQLite store
onyxnode/collector.py  HTTP collector and API
onyxnode/ui/index.html timeline UI (one file, no build)
onyxnode/cli.py        the `onyx` command
onyxnode/demo.py       offline demo agent
tests/                 unittest suite
```
