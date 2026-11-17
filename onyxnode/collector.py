"""The collector: receives events over HTTP, serves the API and the timeline UI.

    GET  /                          timeline UI
    GET  /api/health
    POST /api/ingest                {"events": [...]}   (what HttpSink sends)
    GET  /api/runs?q=&limit=        run summaries, newest first
    GET  /api/runs/<id>             {"run": summary, "events": [...]}
    POST /api/runs/<id>/label       {"label": "good" | "bad" | null}
    GET  /api/diff?a=<id>&b=<id>&ignore=path,path
    GET  /api/suite                 suite cases
    POST /api/suite                 {"run_id": ..., "name": ...}  save run as a case

It has no authentication, so it binds to 127.0.0.1 by default. Two cheap
guards keep other web pages in your browser from talking to it: POST bodies
must be sent as application/json (which a cross-site form cannot do), and on a
loopback bind the Host header must be a loopback name (blocks DNS rebinding).
"""
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, unquote, urlparse

from . import suite as suite_mod
from .diff import diff_runs
from .store import AmbiguousRun, Store

UI_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ui", "index.html")
MAX_BODY = 64 * 1024 * 1024
LOOPBACK = ("127.0.0.1", "localhost", "::1", "[::1]")
LIST_DROP = ("input", "output", "meta")


class Collector:
    def __init__(self, home=".onyx", suite_dir=suite_mod.DEFAULT_DIR):
        self.store = Store(home)
        self.suite_dir = suite_dir
        self._last_sync = 0.0
        self._sync_lock = threading.Lock()

    def sync(self):
        """Pick up run files written by the SDK's file sink (at most once a second)."""
        with self._sync_lock:
            if time.time() - self._last_sync < 1.0:
                return
            self.store.sync_dir()
            self._last_sync = time.time()

    # Each handler returns (status, payload).

    def runs(self, query):
        self.sync()
        limit = _int(query.get("limit"), 200)
        runs = self.store.list_runs(query.get("q"), limit)
        return 200, {"runs": [{k: v for k, v in r.items() if k not in LIST_DROP}
                              for r in runs]}

    def run(self, ref):
        self.sync()
        run_id = self.store.resolve(ref)
        if not run_id:
            return 404, {"error": "no such run: %s" % ref}
        return 200, {"run": self.store.get_run(run_id), "events": self.store.get_events(run_id)}

    def diff(self, query):
        self.sync()
        ids = []
        for key in ("a", "b"):
            run_id = self.store.resolve(query.get(key) or "")
            if not run_id:
                return 404, {"error": "no such run for '%s': %s" % (key, query.get(key))}
            ids.append(run_id)
        ignore = [p for p in (query.get("ignore") or "").split(",") if p.strip()]
        ev_a, ev_b = self.store.get_events(ids[0]), self.store.get_events(ids[1])
        result = diff_runs(ev_a, ev_b, ignore=ignore)
        for key, run_id in zip(("a", "b"), ids):
            result[key]["label"] = (self.store.get_run(run_id) or {}).get("label")
        result["events_a"], result["events_b"] = ev_a, ev_b
        return 200, result

    def ingest(self, body):
        events = body.get("events") if isinstance(body, dict) else body
        if not isinstance(events, list):
            return 400, {"error": "expected {\"events\": [...]}"}
        return 200, {"ok": True, "runs": self.store.ingest(events)}

    def label(self, ref, body):
        run_id = self.store.resolve(ref)
        if not run_id:
            return 404, {"error": "no such run: %s" % ref}
        self.store.set_label(run_id, (body or {}).get("label"))
        return 200, {"run": self.store.get_run(run_id)}

    def suite_list(self):
        cases = [{"name": name} for name, _ in suite_mod.list_cases(self.suite_dir)]
        return 200, {"dir": os.path.abspath(self.suite_dir), "cases": cases}

    def suite_add(self, body):
        body = body or {}
        run_id = self.store.resolve(str(body.get("run_id") or ""))
        if not run_id:
            return 404, {"error": "no such run"}
        try:
            path = suite_mod.add_case(self.suite_dir, self.store.get_events(run_id),
                                      body.get("name"), force=bool(body.get("force")))
        except FileExistsError:
            return 409, {"error": "a case with that name already exists"}
        return 200, {"ok": True, "name": os.path.basename(path)[:-6],
                     "path": os.path.abspath(path)}


def make_handler(collector, loopback_only):
    class Handler(BaseHTTPRequestHandler):
        server_version = "OnyxNode"
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):
            pass

        def do_GET(self):
            self._route("GET")

        def do_POST(self):
            self._route("POST")

        def _route(self, method):
            try:
                if loopback_only and not self._host_ok():
                    return self._send(403, {"error": "bad Host header"})
                url = urlparse(self.path)
                parts = [unquote(p) for p in url.path.split("/") if p]
                query = {k: v[-1] for k, v in parse_qs(url.query).items()}
                body = None
                if method == "POST":
                    ctype = (self.headers.get("Content-Type") or "").split(";")[0].strip()
                    if ctype != "application/json":
                        return self._send(415, {"error": "send application/json"})
                    length = _int(self.headers.get("Content-Length"), 0)
                    if length > MAX_BODY:
                        return self._send(413, {"error": "body too large"})
                    try:
                        body = json.loads(self.rfile.read(length) or b"null")
                    except ValueError:
                        return self._send(400, {"error": "invalid JSON"})
                status, payload = self._dispatch(method, parts, query, body)
                self._send(status, payload)
            except AmbiguousRun as e:
                self._send(400, {"error": str(e)})
            except (BrokenPipeError, ConnectionError):
                pass
            except Exception as e:  # keep the collector up whatever a request does
                self._send(500, {"error": "%s: %s" % (type(e).__name__, e)})

        def _dispatch(self, method, parts, query, body):
            c = collector
            if method == "GET":
                if not parts or parts == ["index.html"]:
                    return 200, _ui()
                if parts == ["api", "health"]:
                    return 200, {"ok": True}
                if parts == ["api", "runs"]:
                    return c.runs(query)
                if len(parts) == 3 and parts[:2] == ["api", "runs"]:
                    return c.run(parts[2])
                if parts == ["api", "diff"]:
                    return c.diff(query)
                if parts == ["api", "suite"]:
                    return c.suite_list()
            else:
                if parts == ["api", "ingest"]:
                    return c.ingest(body)
                if len(parts) == 4 and parts[:2] == ["api", "runs"] and parts[3] == "label":
                    return c.label(parts[2], body)
                if parts == ["api", "suite"]:
                    return c.suite_add(body)
            return 404, {"error": "not found"}

        def _host_ok(self):
            host = (self.headers.get("Host") or "").strip().lower()
            if host.startswith("["):
                host = host[:host.find("]") + 1]
            else:
                host = host.rsplit(":", 1)[0] if ":" in host else host
            return host in LOOPBACK

        def _send(self, status, payload):
            if isinstance(payload, bytes):
                data, ctype = payload, "text/html; charset=utf-8"
            else:
                data = json.dumps(payload, default=repr).encode("utf-8")
                ctype = "application/json"
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            if status >= 400:  # the request body may be unread; don't reuse the socket
                self.close_connection = True
                self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(data)

    return Handler


def _ui():
    with open(UI_FILE, "rb") as f:
        return f.read()


def _int(value, default):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def make_server(home=".onyx", host="127.0.0.1", port=8787, suite_dir=suite_mod.DEFAULT_DIR):
    """Build the HTTP server without starting it. Returns (server, collector)."""
    collector = Collector(home, suite_dir)
    server = ThreadingHTTPServer((host, port), make_handler(collector, host in LOOPBACK))
    server.daemon_threads = True
    return server, collector


def serve(home=".onyx", host="127.0.0.1", port=8787, suite_dir=suite_mod.DEFAULT_DIR):
    server, collector = make_server(home, host, port, suite_dir)
    collector.sync()
    print("OnyxNode collector on http://%s:%d   (store: %s)"
          % (host, server.server_address[1], os.path.abspath(home)))
    if host not in LOOPBACK:
        print("WARNING: bound to %s with no authentication; anyone who can reach "
              "this port can read and write recordings." % host)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        server.server_close()
        collector.store.close()
