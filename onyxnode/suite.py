"""Recorded runs as a regression suite.

A suite is a folder of run files (default ./onyx_suite), one .jsonl per case,
meant to be committed next to the agent's code. Running a case:

1. starts a replay run from the recording, so every tool and LLM call the
   agent makes is answered from the recording instead of the real world;
2. calls your agent entry point, `agent(run, input)`, which returns the
   final output;
3. diffs what the agent did this time against the recording.

The case passes when there is no divergence. When it fails, the first
divergence says exactly which step the agent now does differently.
"""
import asyncio
import importlib
import inspect
import os
import re
import sys

from . import events as E
from .diff import diff_runs, format_report
from .recorder import MemorySink, Recorder, ReplayMiss

DEFAULT_DIR = "onyx_suite"


def case_name(text):
    name = re.sub(r"[^A-Za-z0-9._-]+", "-", text or "").strip("-.")
    return name or "case"


def add_case(suite_dir, events, name=None, force=False):
    """Save a run's events as a suite case. Returns the file path."""
    if not events:
        raise ValueError("run has no events")
    summary = E.summarize(events)
    name = case_name(name or "%s-%s" % (summary["name"] or "run", summary["run_id"]))
    path = os.path.join(suite_dir, name + ".jsonl")
    if os.path.exists(path) and not force:
        raise FileExistsError("suite case already exists: %s" % path)
    E.write_jsonl(path, events)
    return path


def list_cases(suite_dir):
    """[(name, path)] for every case in the suite folder, sorted by name."""
    if not os.path.isdir(suite_dir):
        return []
    return [(f[:-6], os.path.join(suite_dir, f))
            for f in sorted(os.listdir(suite_dir)) if f.endswith(".jsonl")]


def load_agent(spec):
    """Import an agent entry point given as 'package.module:function'."""
    if ":" not in spec:
        raise ValueError("agent must look like 'module:function', got %r" % spec)
    module, func = spec.split(":", 1)
    if os.getcwd() not in sys.path:
        sys.path.insert(0, os.getcwd())
    return getattr(importlib.import_module(module), func)


def run_case(source, agent, name=None, on_miss="error", ignore=(), sink=None):
    """Replay one case against `agent`. Returns a result dict.

    source  path of a case file, or a list of events
    agent   callable (run, input) -> output; may be async
    sink    optional extra sink that also receives the replayed run
    """
    baseline = E.read_jsonl(source) if isinstance(source, (str, os.PathLike)) else list(source)
    memory = MemorySink()
    rec = Recorder(sink=[memory, sink] if sink is not None else memory)
    run = rec.replay(baseline, on_miss=on_miss, ignore=ignore)
    error = None
    try:
        with run:
            output = agent(run, run.input)
            if inspect.iscoroutine(output):
                output = asyncio.run(output)
            run.finish(output=output)
    except ReplayMiss as e:
        error = str(e)
    except Exception as e:
        error = "%s: %s" % (type(e).__name__, e)
    result = diff_runs(baseline, memory.events, ignore=ignore,
                       names=("the recording", "the agent now"))
    return {
        "name": name or (os.path.basename(str(source))[:-6]
                         if isinstance(source, (str, os.PathLike)) else run.name),
        "passed": result["identical"],
        "error": error,
        "run_id": run.run_id,
        "diff": result,
    }


def run_suite(suite_dir, agent, on_miss="error", ignore=(), sink=None, only=None):
    results = []
    for name, path in list_cases(suite_dir):
        if only and only not in name:
            continue
        results.append(run_case(path, agent, name=name, on_miss=on_miss,
                                ignore=ignore, sink=sink))
    return results


def format_results(results, verbose=False):
    lines = []
    for r in results:
        fd = r["diff"]["first_divergence"]
        if r["passed"]:
            lines.append("PASS  %s  (%d steps)" % (r["name"], len(r["diff"]["rows"])))
            continue
        lines.append("FAIL  %s" % r["name"])
        lines.append("      row %d: %s" % (fd["row"], fd["headline"]))
        for ch in fd["changes"][:6]:
            lines.append("        %s %s: %s -> %s" % (
                ch["op"], ch["path"], _short(ch["a"]), _short(ch["b"])))
        if r["error"]:
            lines.append("      agent stopped: %s" % r["error"])
        if verbose:
            lines += ["      " + ln for ln in format_report(r["diff"]).splitlines()]
    passed = sum(1 for r in results if r["passed"])
    lines.append("")
    lines.append("%d passed, %d failed, %d total" % (passed, len(results) - passed, len(results)))
    return "\n".join(lines)


def _short(value, width=60):
    text = value if isinstance(value, str) else E.canon(value)
    text = text.replace("\n", "\\n")
    return text if len(text) <= width else text[:width - 3] + "..."
