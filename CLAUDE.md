# OnyxNode

Black-box flight recorder for AI agent runs: a Python SDK that records tool
calls, model calls, decisions and cost; a collector (HTTP + SQLite); a diff
engine with a first-divergence finder; a regression suite that replays
recorded runs; and a single-file timeline UI. README.md is the user guide.

## Deliberate constraints

- **Standard library only.** No runtime dependencies, no frameworks, no build
  step. `pyproject.toml` must keep `dependencies = []`.
- **The UI is one file**, `onyxnode/ui/index.html`: vanilla JS and CSS, nothing
  loaded from the network. Recorded payloads are untrusted, so they are put on
  the page with `textContent` (via the `h()` helper) and never as HTML.
- **Recording must never break the agent.** Sink and serialization failures
  are swallowed with one warning. Exceptions from the user's own tool or model
  function are recorded and re-raised unchanged.
- **The event format is saved user data.** Recordings and suite cases on disk
  use the JSONL schema described at the top of `onyxnode/events.py`
  (`SCHEMA_VERSION`). Changing field names or meanings breaks existing
  recordings and committed suite cases; add fields instead, or bump the
  version and keep reading the old one.
- **No model prices are built in.** Cost comes from the caller
  (`LLMResult.cost_usd` or `Recorder(pricing=...)`).
- **The collector has no auth** and binds to loopback by default. Keep the two
  request guards in `collector.py` (JSON-only POST, loopback Host check).
- CLI output is plain ASCII so it works on any Windows console code page.

## Commands

    python -m unittest discover -s tests -v     run the tests
    python -m onyxnode demo                     record a good and a bad demo run
    python -m onyxnode diff latest~1 latest     first divergence between them
    python -m onyxnode serve                    collector + UI on 127.0.0.1:8787

`onyx` is the same CLI once the package is installed (`pip install -e .`).

## Layout

    onyxnode/events.py     schema, canonical JSON + digests, redaction, summarize()
    onyxnode/recorder.py   Recorder, Run, sinks (memory / JSONL / HTTP), replay tape
    onyxnode/diff.py       weighted-LCS alignment, row statuses, verdict text
    onyxnode/suite.py      suite cases = run files; run_case() replays and diffs
    onyxnode/store.py      SQLite store; ingest is idempotent on (run_id, seq)
    onyxnode/collector.py  HTTP API and static UI
    onyxnode/cli.py        argparse CLI
    onyxnode/demo.py       offline demo agent (also the suite's example agent)
    tests/test_onyxnode.py unittest suite, including a live collector round trip

## How the pieces fit

- Everything exchanges one thing: a run's ordered list of event dicts. The
  SDK emits them, sinks write them, the store indexes them, the diff and the
  suite read them.
- Local state lives in `.onyx/` (git-ignored): `onyx.db` and `runs/*.jsonl`.
  The store re-imports changed files from `runs/` on demand, so the file sink
  and the collector need no wiring.
- `onyx_suite/` holds regression cases and is meant to be committed.
- Diff semantics (what counts as a divergence, what is never compared) are
  documented in the `diff.py` module docstring; the README table mirrors it.
  If the statuses or the verdict wording change, update both, and the
  `STATUS` map in the UI.
