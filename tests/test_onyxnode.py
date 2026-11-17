"""Run with:  python -m unittest discover -s tests -v"""
import asyncio
import json
import os
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import onyxnode  # noqa: E402
from onyxnode import demo, suite  # noqa: E402
from onyxnode import events as E  # noqa: E402
from onyxnode.collector import make_server  # noqa: E402
from onyxnode.diff import changes, diff_runs  # noqa: E402
from onyxnode.store import AmbiguousRun, Store  # noqa: E402


def record(fn, **kw):
    """Record one run in memory; fn(run) does the work. Returns the events."""
    sink = onyxnode.MemorySink()
    rec = onyxnode.Recorder("t", sink=sink, **kw)
    with rec.start(input={"q": 1}) as run:
        run.finish(output=fn(run))
    return sink.events


def steps(*calls):
    """Events for a run that makes tool calls [(name, input, output), ...]."""
    def body(run):
        for name, inp, out in calls:
            run.tool(name, lambda **k: out, **inp)
    return record(body)


class RecorderTests(unittest.TestCase):
    def test_records_sequence_cost_and_summary(self):
        def body(run):
            run.tool("add", lambda a, b: a + b, 2, b=3)
            run.llm("m", lambda p: onyxnode.LLMResult("hi", tokens_in=1000, tokens_out=500), "p")
            run.decision("route", choice="x", reason="because", options=["x", "y"])
            run.note("checkpoint", n=1)
            return "done"
        ev = record(body, pricing={"m": (2.0, 10.0)})
        self.assertEqual([e["kind"] for e in ev], ["run_start", "tool_call", "llm_call",
                                                   "decision", "note", "run_end"])
        self.assertEqual([e["seq"] for e in ev], list(range(6)))
        self.assertEqual(ev[1]["input"], {"args": [2], "kwargs": {"b": 3}})
        self.assertEqual(ev[1]["output"], 5)
        self.assertAlmostEqual(ev[2]["cost_usd"], 0.007)  # 1000*2/1e6 + 500*10/1e6
        self.assertEqual(ev[3]["output"], {"choice": "x", "reason": "because"})
        s = E.summarize(ev)
        self.assertEqual((s["status"], s["steps"], s["llm_calls"], s["tool_calls"]),
                         ("ok", 4, 1, 1))
        self.assertEqual(s["output"], "done")

    def test_tool_error_is_recorded_and_reraised(self):
        sink = onyxnode.MemorySink()
        rec = onyxnode.Recorder(sink=sink)
        with self.assertRaises(ZeroDivisionError):
            with rec.start() as run:
                run.tool("div", lambda: 1 / 0)
        self.assertIn("ZeroDivisionError", sink.events[1]["error"])
        self.assertEqual(sink.events[-1]["status"], "error")

    def test_redacts_credentials_but_not_token_counts(self):
        ev = record(lambda run: run.tool(
            "call", lambda **k: {"access_token": "abc", "max_tokens": 5},
            api_key="sk-secret", headers={"Authorization": "Bearer x"}, q="ok"))
        self.assertEqual(ev[1]["input"], {"api_key": E.REDACTED,
                                          "headers": {"Authorization": E.REDACTED}, "q": "ok"})
        self.assertEqual(ev[1]["output"], {"access_token": E.REDACTED, "max_tokens": 5})

    def test_unserializable_values_and_broken_sink_never_raise(self):
        class Broken:
            def emit(self, event):
                raise OSError("disk full")

            def flush(self):
                raise OSError("disk full")
        rec = onyxnode.Recorder(sink=Broken())
        with self.assertWarns(UserWarning):
            with rec.start() as run:
                self.assertEqual(run.tool("t", lambda x: 7, object()), 7)

    def test_async_tool(self):
        async def fetch(x):
            return x * 2

        async def main(run):
            return await run.atool("fetch", fetch, 21)
        sink = onyxnode.MemorySink()
        with onyxnode.Recorder(sink=sink).start() as run:
            self.assertEqual(asyncio.run(main(run)), 42)
        self.assertEqual(sink.events[1]["output"], 42)

    def test_jsonl_sink_round_trip(self):
        with tempfile.TemporaryDirectory() as d:
            rec = onyxnode.Recorder(sink=d)
            with rec.start(input="x") as run:
                run.tool("t", lambda: "y")
            ev = E.read_jsonl(os.path.join(d, run.run_id + ".jsonl"))
            self.assertEqual([e["kind"] for e in ev], ["run_start", "tool_call", "run_end"])


class DiffTests(unittest.TestCase):
    def test_identical_runs(self):
        a = steps(("s", {"q": 1}, "r1"), ("t", {"x": 1}, "r2"))
        b = steps(("s", {"q": 1}, "r1"), ("t", {"x": 1}, "r2"))
        d = diff_runs(a, b)
        self.assertTrue(d["identical"])
        self.assertIsNone(d["first_divergence"])

    def test_same_input_different_output(self):
        a = steps(("s", {"q": 1}, {"n": 1, "k": "same"}), ("t", {"x": 1}, "r"))
        b = steps(("s", {"q": 1}, {"n": 2, "k": "same"}), ("t", {"x": 2}, "r"))
        fd = diff_runs(a, b)["first_divergence"]
        self.assertEqual((fd["row"], fd["status"], fd["name"]), (1, "output_differs", "s"))
        self.assertEqual(fd["matched_before"], 1)
        self.assertEqual(fd["changes"], [{"path": "output.n", "op": "changed", "a": 1, "b": 2}])
        self.assertEqual(fd["downstream_differences"], 1)

    def test_different_input(self):
        a = steps(("s", {"q": 1}, "r"))
        b = steps(("s", {"q": 2}, "r"))
        fd = diff_runs(a, b)["first_divergence"]
        self.assertEqual((fd["row"], fd["status"]), (1, "input_differs"))

    def test_skipped_step_aligns_on_exact_match(self):
        # B skipped the first search; the second one must pair up, not shift.
        a = steps(("s", {"q": 1}, "r1"), ("s", {"q": 2}, "r2"), ("t", {}, "r3"))
        b = steps(("s", {"q": 2}, "r2"), ("t", {}, "r3"))
        d = diff_runs(a, b)
        self.assertEqual([r["status"] for r in d["rows"]],
                         ["same", "only_a", "same", "same", "same"])
        self.assertEqual(d["first_divergence"]["status"], "only_a")

    def test_extra_step_in_b(self):
        a = steps(("s", {"q": 1}, "r1"), ("t", {}, "r3"))
        b = steps(("s", {"q": 1}, "r1"), ("u", {}, "x"), ("t", {}, "r3"))
        d = diff_runs(a, b)
        self.assertEqual([r["status"] for r in d["rows"]],
                         ["same", "same", "only_b", "same", "same"])

    def test_ignore_paths(self):
        a = steps(("s", {"q": 1, "now": 100}, {"id": "a", "items": [{"v": 1, "ts": 5}]}))
        b = steps(("s", {"q": 1, "now": 200}, {"id": "b", "items": [{"v": 1, "ts": 9}]}))
        self.assertFalse(diff_runs(a, b)["identical"])
        d = diff_runs(a, b, ignore=["input.now", "output.id", "output.items.*.ts"])
        self.assertTrue(d["identical"])

    def test_different_run_input_and_final_output(self):
        a = record(lambda run: "one")
        b = record(lambda run: "two")
        fd = diff_runs(a, b)["first_divergence"]
        self.assertEqual((fd["kind"], fd["status"]), ("run_end", "output_differs"))

    def test_large_runs_use_fallback_alignment(self):
        from onyxnode import diff as diff_mod
        a = steps(*[("s", {"q": i}, i) for i in range(30)])
        b = steps(*[("s", {"q": i + 100}, i) for i in range(30)])
        old, diff_mod.MAX_DP_CELLS = diff_mod.MAX_DP_CELLS, 10
        try:
            d = diff_runs(a, b)
        finally:
            diff_mod.MAX_DP_CELLS = old
        self.assertEqual(d["counts"]["input_differs"], 30)

    def test_changes_lists_added_removed_changed(self):
        got = changes({"a": 1, "b": [1, 2], "c": "x"}, {"a": 1, "b": [1], "d": True})
        self.assertEqual([(c["path"], c["op"]) for c in got],
                         [("b.1", "removed"), ("c", "removed"), ("d", "added")])

    def test_demo_first_divergence_is_the_order_lookup(self):
        good, bad = onyxnode.MemorySink(), onyxnode.MemorySink()
        demo.record(good)
        demo.record(bad, stale_orders=True)
        fd = diff_runs(good.events, bad.events)["first_divergence"]
        self.assertEqual((fd["kind"], fd["name"], fd["status"]),
                         ("tool_call", "lookup_order", "output_differs"))
        self.assertIn("output.delivered_days_ago", [c["path"] for c in fd["changes"]])


class ReplayTests(unittest.TestCase):
    def setUp(self):
        self.calls = []

    def agent(self, run, inp, second="b"):
        def tool(**kw):
            self.calls.append(kw)
            return {"got": kw}
        first = run.tool("one", tool, x="a")
        run.tool("two", tool, x=second, prev=first["got"]["x"])
        return "ok"

    def test_replay_answers_from_recording_without_calling_tools(self):
        baseline = record(lambda run: self.agent(run, None))
        self.calls.clear()
        sink = onyxnode.MemorySink()
        run = onyxnode.Recorder(sink=sink).replay(baseline)
        with run:
            run.finish(output=self.agent(run, run.input))
        self.assertEqual(self.calls, [])
        self.assertTrue(diff_runs(baseline, sink.events)["identical"])
        self.assertEqual(sink.events[0]["meta"]["replay_of"], baseline[0]["run_id"])
        self.assertTrue(sink.events[1]["meta"]["replayed"])

    def test_replay_miss_raises_and_is_recorded(self):
        baseline = record(lambda run: self.agent(run, None))
        sink = onyxnode.MemorySink()
        run = onyxnode.Recorder(sink=sink).replay(baseline)
        with self.assertRaises(onyxnode.ReplayMiss):
            with run:
                self.agent(run, run.input, second="CHANGED")
        fd = diff_runs(baseline, sink.events)["first_divergence"]
        self.assertEqual((fd["name"], fd["status"]), ("two", "input_differs"))

    def test_replay_live_on_miss_calls_the_real_tool(self):
        baseline = record(lambda run: self.agent(run, None))
        self.calls.clear()
        run = onyxnode.Recorder(sink=onyxnode.MemorySink()).replay(baseline, on_miss="live")
        with run:
            self.agent(run, run.input, second="CHANGED")
        self.assertEqual(self.calls, [{"x": "CHANGED", "prev": "a"}])

    def test_replayed_failure_raises(self):
        def body(run):
            try:
                run.tool("boom", lambda: 1 / 0)
            except ZeroDivisionError:
                return "recovered"
        baseline = record(body)
        run = onyxnode.Recorder(sink=onyxnode.MemorySink()).replay(baseline)
        with self.assertRaises(onyxnode.ReplayedError):
            run.tool("boom", lambda: 1)


class SuiteTests(unittest.TestCase):
    def test_case_passes_then_fails_when_agent_changes(self):
        sink = onyxnode.MemorySink()
        demo.record(sink)
        with tempfile.TemporaryDirectory() as d:
            path = suite.add_case(d, sink.events, name="refund: damaged item")
            self.assertTrue(path.endswith("refund-damaged-item.jsonl"))
            with self.assertRaises(FileExistsError):
                suite.add_case(d, sink.events, name="refund: damaged item")
            results = suite.run_suite(d, demo.agent)
            self.assertEqual([r["passed"] for r in results], [True])
            os.environ["ONYX_DEMO_BUG"] = "1"
            try:
                results = suite.run_suite(d, demo.agent)
            finally:
                del os.environ["ONYX_DEMO_BUG"]
            self.assertEqual([r["passed"] for r in results], [False])
            fd = results[0]["diff"]["first_divergence"]
            self.assertEqual((fd["name"], fd["status"]), ("check_refund_policy", "only_a"))
            self.assertIn("FAIL  refund-damaged-item", suite.format_results(results))

    def test_async_agent(self):
        async def agent(run, inp):
            return await run.atool("t", None, x=1)
        baseline = record(lambda run: run.tool("t", lambda x: "v", x=1))
        self.assertTrue(suite.run_case(baseline, agent)["passed"])


class StoreAndCollectorTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.home = os.path.join(self.tmp.name, "home")

    def test_store_ingest_is_idempotent_and_keeps_labels(self):
        store = Store(self.home)
        self.addCleanup(store.close)
        ev = steps(("s", {"q": 1}, "r"))
        store.ingest(ev[:2])
        self.assertEqual(store.list_runs()[0]["status"], "running")
        run_id = ev[0]["run_id"]
        store.set_label(run_id, "good")
        store.ingest(ev)
        store.ingest(ev)
        run = store.get_run(run_id)
        self.assertEqual((run["status"], run["steps"], run["label"]), ("ok", 1, "good"))
        self.assertEqual(len(store.get_events(run_id)), 3)
        self.assertEqual(store.resolve(run_id[:12]), run_id)
        self.assertEqual(store.resolve("latest"), run_id)
        self.assertIsNone(store.resolve("nope"))

    def test_store_picks_up_run_files(self):
        store = Store(self.home)
        self.addCleanup(store.close)
        rec = onyxnode.Recorder(sink=store.runs_dir)
        with rec.start() as run:
            run.tool("t", lambda: 1)
        self.assertEqual(store.sync_dir(), 1)
        self.assertEqual(store.sync_dir(), 0)
        self.assertEqual(store.get_run(run.run_id)["steps"], 1)

    def test_ambiguous_prefix(self):
        store = Store(self.home)
        self.addCleanup(store.close)
        for rid in ("2026-aaa", "2026-bbb"):
            store.ingest([{"run_id": rid, "seq": 0, "ts": 1.0, "kind": "run_start", "name": "n"}])
        with self.assertRaises(AmbiguousRun):
            store.resolve("2026")

    def test_collector_http_round_trip(self):
        server, collector = make_server(self.home, port=0,
                                        suite_dir=os.path.join(self.tmp.name, "suite"))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()

        def cleanup():
            server.shutdown()
            server.server_close()
            collector.store.close()
        self.addCleanup(cleanup)
        base = "http://127.0.0.1:%d" % server.server_address[1]

        def get(path):
            with urllib.request.urlopen(base + path, timeout=5) as r:
                return json.loads(r.read())

        def post(path, body, ctype="application/json", host=None):
            headers = {"Content-Type": ctype}
            if host:
                headers["Host"] = host
            req = urllib.request.Request(base + path, data=json.dumps(body).encode(),
                                         headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=5) as r:
                return json.loads(r.read())

        sink = onyxnode.HttpSink(base, spool_dir=os.path.join(self.tmp.name, "spool"))
        good = demo.record(sink)
        bad = demo.record(sink, stale_orders=True)

        runs = get("/api/runs")["runs"]
        self.assertEqual({r["run_id"] for r in runs}, {good, bad})
        self.assertNotIn("input", runs[0])
        self.assertEqual(len(get("/api/runs/" + good)["events"]), 10)
        d = get("/api/diff?a=%s&b=%s" % (good, bad))
        self.assertEqual(d["first_divergence"]["name"], "lookup_order")
        self.assertEqual(post("/api/runs/%s/label" % good, {"label": "good"})["run"]["label"], "good")
        self.assertEqual(post("/api/suite", {"run_id": good, "name": "case-1"})["name"], "case-1")
        self.assertEqual(get("/api/suite")["cases"], [{"name": "case-1"}])
        self.assertIn(b"OnyxNode", urllib.request.urlopen(base + "/", timeout=5).read())

        for kwargs, code in (({"ctype": "text/plain"}, 415), ({"host": "evil.example"}, 403)):
            with self.assertRaises(urllib.error.HTTPError) as ctx:
                post("/api/ingest", {"events": []}, **kwargs)
            self.assertEqual(ctx.exception.code, code)
            ctx.exception.close()
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            get("/api/runs/does-not-exist")
        self.assertEqual(ctx.exception.code, 404)
        ctx.exception.close()

    def test_http_sink_spools_when_collector_is_down(self):
        spool = os.path.join(self.tmp.name, "spool")
        sink = onyxnode.HttpSink("http://127.0.0.1:9", spool_dir=spool, timeout=1)
        with self.assertWarns(UserWarning):
            run_id = demo.record(sink)
            sink.flush()
        self.assertEqual(len(E.read_jsonl(os.path.join(spool, run_id + ".jsonl"))), 10)


if __name__ == "__main__":
    unittest.main()
