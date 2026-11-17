"""Command line: `onyx ...` (or `python -m onyxnode ...`).

A run can be named by its id, a unique prefix of it, `latest`, `latest~1`
(the one before), or the path of a .jsonl run file. Output is plain ASCII so
it survives any Windows console code page.
"""
import argparse
import json
import os
import sys
import time

from . import __version__, demo, suite
from . import events as E
from .diff import diff_runs, format_report
from .recorder import JsonlSink, default_home
from .store import AmbiguousRun, Store


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="onyx", description="OnyxNode: black-box flight recorder for AI agent runs.")
    parser.add_argument("--home", default=default_home(),
                        help="store folder (default: $ONYX_DIR or .onyx)")
    parser.add_argument("--version", action="version", version="onyxnode " + __version__)
    sub = parser.add_subparsers(dest="cmd", metavar="command")

    p = sub.add_parser("serve", help="run the collector and timeline UI")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8787)
    p.add_argument("--suite-dir", default=suite.DEFAULT_DIR)

    p = sub.add_parser("runs", help="list recorded runs")
    p.add_argument("query", nargs="?", help="filter by name or id")
    p.add_argument("-n", "--limit", type=int, default=30)

    p = sub.add_parser("show", help="print a run's timeline")
    p.add_argument("run")
    p.add_argument("--json", action="store_true", help="print the raw events")

    p = sub.add_parser("diff", help="diff two runs and find the first divergence")
    p.add_argument("a", help="the reference (good) run")
    p.add_argument("b", help="the run to explain (bad)")
    p.add_argument("--ignore", action="append", default=[], metavar="PATH",
                   help="payload path to leave out, e.g. output.request_id (repeatable)")
    p.add_argument("--json", action="store_true")

    p = sub.add_parser("label", help="mark a run good or bad")
    p.add_argument("run")
    p.add_argument("label", choices=["good", "bad", "none"])

    p = sub.add_parser("import", help="load .jsonl run files (or folders of them) into the store")
    p.add_argument("paths", nargs="+")

    p = sub.add_parser("export", help="write a run to a .jsonl file")
    p.add_argument("run")
    p.add_argument("-o", "--out", help="file to write (default: <run_id>.jsonl)")

    p = sub.add_parser("suite", help="regression suite built from recorded runs")
    ssub = p.add_subparsers(dest="suite_cmd", metavar="action")
    p.add_argument("--dir", default=suite.DEFAULT_DIR, help="suite folder (default: onyx_suite)")
    q = ssub.add_parser("add", help="save a run as a regression case")
    q.add_argument("run")
    q.add_argument("--name")
    q.add_argument("--force", action="store_true", help="replace an existing case")
    ssub.add_parser("list", help="list cases")
    q = ssub.add_parser("run", help="replay every case against your agent")
    q.add_argument("--agent", required=True, metavar="MODULE:FUNCTION",
                   help="agent entry point, called as agent(run, input)")
    q.add_argument("-k", dest="only", help="only cases whose name contains this")
    q.add_argument("--ignore", action="append", default=[], metavar="PATH")
    q.add_argument("--live-on-miss", action="store_true",
                   help="make the real call when the recording has no answer")
    q.add_argument("--save", action="store_true",
                   help="also record the replayed runs into the store, to inspect in the UI")
    q.add_argument("-v", "--verbose", action="store_true")

    sub.add_parser("demo", help="record a good and a bad run of the built-in demo agent")

    args = parser.parse_args(argv)
    if not args.cmd:
        parser.print_help()
        return 2
    try:
        return COMMANDS[args.cmd](args) or 0
    except (AmbiguousRun, LookupError, FileExistsError, ValueError) as e:
        print("error: %s" % e, file=sys.stderr)
        return 2
    except BrokenPipeError:
        return 0


# ---------------------------------------------------------------- helpers

def _store(args):
    store = Store(args.home)
    store.sync_dir()
    return store


def _load(store, ref):
    """Events for a run reference (file path or id)."""
    if os.path.isfile(ref):
        return E.read_jsonl(ref)
    run_id = store.resolve(ref)
    if not run_id:
        raise LookupError("no such run: %s" % ref)
    return store.get_events(run_id)


def _when(ts):
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts)) if ts else "-"


def _short(value, width):
    if value is None:
        return ""
    text = value if isinstance(value, str) else E.canon(value)
    text = text.replace("\n", "\\n")
    return text if len(text) <= width else text[:width - 3] + "..."


# ---------------------------------------------------------------- commands

def cmd_serve(args):
    from .collector import serve
    serve(args.home, args.host, args.port, args.suite_dir)


def cmd_runs(args):
    runs = _store(args).list_runs(args.query, args.limit)
    if not runs:
        print("no runs recorded in %s  (try: onyx demo)" % os.path.abspath(args.home))
        return
    print("%-24s %-5s %-20s %-8s %5s %9s  %s" % (
        "RUN", "LABEL", "NAME", "STATUS", "STEPS", "COST", "STARTED"))
    for r in runs:
        print("%-24s %-5s %-20s %-8s %5d %9s  %s" % (
            r["run_id"], r["label"] or "", (r["name"] or "")[:20], r["status"], r["steps"],
            "$%.4f" % r["cost_usd"], _when(r["started_at"])))


def cmd_show(args):
    events = _load(_store(args), args.run)
    if args.json:
        for ev in events:
            print(E.dump_event(ev))
        return
    s = E.summarize(events)
    print("%s  %s  %s" % (s["run_id"], s["name"], s["status"]))
    print("started %s   %d steps   %d llm   %d tool   %d+%d tokens   $%.4f" % (
        _when(s["started_at"]), s["steps"], s["llm_calls"], s["tool_calls"],
        s["tokens_in"], s["tokens_out"], s["cost_usd"]))
    print()
    t0 = s["started_at"] or 0
    for ev in events:
        body = _short(ev.get("input"), 44)
        result = ev.get("error") and "ERROR " + ev["error"] or _short(ev.get("output"), 50)
        print("%4d  +%7.3fs  %-9s %-22s %s%s" % (
            ev["seq"], (ev.get("ts") or t0) - t0, ev.get("kind", "?"),
            (ev.get("name") or "")[:22], body, "  ->  " + result if result else ""))


def cmd_diff(args):
    store = _store(args)
    result = diff_runs(_load(store, args.a), _load(store, args.b), ignore=args.ignore)
    if args.json:
        print(json.dumps(result, indent=2, default=repr))
    else:
        print(format_report(result))
    return 0 if result["identical"] else 1


def cmd_label(args):
    store = _store(args)
    run_id = store.resolve(args.run)
    if not run_id:
        raise LookupError("no such run: %s" % args.run)
    store.set_label(run_id, None if args.label == "none" else args.label)
    print("%s labelled %s" % (run_id, args.label))


def cmd_import(args):
    store = Store(args.home)
    files = []
    for path in args.paths:
        if os.path.isdir(path):
            files += [os.path.join(path, f) for f in sorted(os.listdir(path))
                      if f.endswith(".jsonl")]
        elif os.path.isfile(path):
            files.append(path)
        else:
            raise LookupError("no such file or folder: %s" % path)
    runs = sum(store.import_file(f) for f in files)
    print("imported %d run(s) from %d file(s)" % (runs, len(files)))


def cmd_export(args):
    events = _load(_store(args), args.run)
    out = args.out or (events[0]["run_id"] + ".jsonl")
    E.write_jsonl(out, events)
    print("wrote %s (%d events)" % (out, len(events)))


def cmd_suite(args):
    if args.suite_cmd == "add":
        events = _load(_store(args), args.run)
        path = suite.add_case(args.dir, events, args.name, force=args.force)
        print("added case %s" % path)
    elif args.suite_cmd == "run":
        agent = suite.load_agent(args.agent)
        sink = JsonlSink(os.path.join(args.home, "runs")) if args.save else None
        results = suite.run_suite(args.dir, agent, ignore=args.ignore, sink=sink, only=args.only,
                                  on_miss="live" if args.live_on_miss else "error")
        if not results:
            print("no cases in %s  (add one: onyx suite add <run>)" % os.path.abspath(args.dir))
            return 2
        print(suite.format_results(results, verbose=args.verbose))
        return 0 if all(r["passed"] for r in results) else 1
    else:
        cases = suite.list_cases(args.dir)
        if not cases:
            print("no cases in %s  (add one: onyx suite add <run>)" % os.path.abspath(args.dir))
        for name, path in cases:
            s = E.summarize(E.read_jsonl(path))
            print("%-40s %3d steps  recorded %s" % (name, s["steps"], _when(s["started_at"])))


def cmd_demo(args):
    sink = JsonlSink(os.path.join(args.home, "runs"))
    good = demo.record(sink)
    bad = demo.record(sink, stale_orders=True)
    store = _store(args)
    store.set_label(good, "good")
    store.set_label(bad, "bad")
    print("recorded two runs of the demo refund agent into %s" % os.path.abspath(args.home))
    print("  good  %s" % good)
    print("  bad   %s" % bad)
    print()
    print("next:")
    print("  onyx diff %s %s" % (good, bad))
    print("  onyx serve                 (then open http://127.0.0.1:8787)")
    print("  onyx suite add %s --name refund-damaged-item" % good)
    print("  onyx suite run --agent onyxnode.demo:agent")


COMMANDS = {"serve": cmd_serve, "runs": cmd_runs, "show": cmd_show, "diff": cmd_diff,
            "label": cmd_label, "import": cmd_import, "export": cmd_export,
            "suite": cmd_suite, "demo": cmd_demo}

if __name__ == "__main__":
    sys.exit(main())
