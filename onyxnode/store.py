"""SQLite store behind the collector and the CLI.

Layout of an OnyxNode home folder (default ./.onyx):

    .onyx/onyx.db      the store
    .onyx/runs/*.jsonl run files written by the SDK's file sink; picked up
                       automatically by sync_dir()

Ingest is idempotent: an event is keyed by (run_id, seq), so importing the
same file twice, or receiving a run over HTTP and from a file, is harmless.
"""
import json
import os
import sqlite3
import threading

from . import events as E

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY, name TEXT, status TEXT, started_at REAL, ended_at REAL,
    label TEXT, summary TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS events (
    run_id TEXT NOT NULL, seq INTEGER NOT NULL, data TEXT NOT NULL,
    PRIMARY KEY (run_id, seq)
);
CREATE TABLE IF NOT EXISTS files (path TEXT PRIMARY KEY, size INTEGER, mtime REAL);
CREATE INDEX IF NOT EXISTS runs_started ON runs (started_at);
"""

LABELS = ("good", "bad")


class AmbiguousRun(LookupError):
    pass


class Store:
    def __init__(self, home=".onyx"):
        self.home = str(home)
        self.runs_dir = os.path.join(self.home, "runs")
        os.makedirs(self.runs_dir, exist_ok=True)
        self._lock = threading.RLock()
        self._db = sqlite3.connect(os.path.join(self.home, "onyx.db"),
                                   check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        with self._lock:
            self._db.executescript(SCHEMA)

    def close(self):
        with self._lock:
            self._db.close()

    # -- writing

    def ingest(self, events):
        """Store events and refresh the summary of every run they touch."""
        touched = set()
        with self._lock, self._db:
            for ev in events:
                if not isinstance(ev, dict):
                    continue
                run_id, seq = ev.get("run_id"), ev.get("seq")
                if not isinstance(run_id, str) or not run_id or not isinstance(seq, int):
                    continue
                self._db.execute("INSERT OR REPLACE INTO events VALUES (?, ?, ?)",
                                 (run_id, seq, E.dump_event(ev)))
                touched.add(run_id)
            for run_id in touched:
                s = E.summarize(self._events(run_id))
                self._db.execute(
                    "INSERT INTO runs (run_id, name, status, started_at, ended_at, summary) "
                    "VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT (run_id) DO UPDATE SET "
                    "name=excluded.name, status=excluded.status, started_at=excluded.started_at, "
                    "ended_at=excluded.ended_at, summary=excluded.summary",
                    (run_id, s["name"], s["status"], s["started_at"], s["ended_at"],
                     json.dumps(s, default=repr)))
        return len(touched)

    def import_file(self, path):
        return self.ingest(E.read_jsonl(path))

    def sync_dir(self, directory=None):
        """Import new or changed .jsonl files from the runs folder."""
        directory = directory or self.runs_dir
        if not os.path.isdir(directory):
            return 0
        changed = 0
        with self._lock:
            known = {r["path"]: (r["size"], r["mtime"])
                     for r in self._db.execute("SELECT * FROM files")}
            for entry in os.scandir(directory):
                if not entry.name.endswith(".jsonl") or not entry.is_file():
                    continue
                st = entry.stat()
                path = os.path.abspath(entry.path)
                if known.get(path) == (st.st_size, st.st_mtime):
                    continue
                try:
                    self.import_file(path)
                except OSError:
                    continue
                with self._db:
                    self._db.execute("INSERT OR REPLACE INTO files VALUES (?, ?, ?)",
                                     (path, st.st_size, st.st_mtime))
                changed += 1
        return changed

    def set_label(self, run_id, label):
        if label not in LABELS:
            label = None
        with self._lock, self._db:
            cur = self._db.execute("UPDATE runs SET label=? WHERE run_id=?", (label, run_id))
        return cur.rowcount > 0

    # -- reading

    def list_runs(self, query=None, limit=200):
        sql, args = "SELECT * FROM runs", []
        if query:
            sql += " WHERE run_id LIKE ? OR name LIKE ?"
            args += ["%" + query + "%"] * 2
        sql += " ORDER BY started_at DESC, run_id DESC LIMIT ?"
        args.append(int(limit))
        with self._lock:
            return [self._summary(r) for r in self._db.execute(sql, args)]

    def get_run(self, run_id):
        with self._lock:
            row = self._db.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
        return self._summary(row) if row else None

    def get_events(self, run_id):
        with self._lock:
            return self._events(run_id)

    def resolve(self, ref):
        """Full run id from an id, a unique id prefix, or 'latest' / 'latest~N'."""
        with self._lock:
            if ref.startswith("latest"):
                back = int(ref[7:]) if ref[6:7] == "~" and ref[7:].isdigit() else 0
                rows = self._db.execute(
                    "SELECT run_id FROM runs ORDER BY started_at DESC, run_id DESC "
                    "LIMIT 1 OFFSET ?", (back,)).fetchall()
            else:
                rows = self._db.execute(
                    "SELECT run_id FROM runs WHERE run_id=? OR run_id LIKE ? LIMIT 3",
                    (ref, ref.replace("%", "").replace("_", "") + "%")).fetchall()
        ids = sorted({r["run_id"] for r in rows})
        if ref in ids:
            return ref
        if len(ids) > 1:
            raise AmbiguousRun("'%s' matches several runs: %s" % (ref, ", ".join(ids)))
        return ids[0] if ids else None

    def _events(self, run_id):
        return [json.loads(r["data"]) for r in self._db.execute(
            "SELECT data FROM events WHERE run_id=? ORDER BY seq", (run_id,))]

    @staticmethod
    def _summary(row):
        s = json.loads(row["summary"])
        s["label"] = row["label"]
        return s
