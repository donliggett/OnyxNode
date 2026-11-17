"""Diff two runs and find the first step where they split.

How it works:

1. Each comparable event gets a key: its shape (kind, name) plus a digest of
   its input, and a second digest of its output and error.
2. The two key lists are aligned with a weighted longest-common-subsequence:
   a pair with the same shape and the same input scores 3, same shape with a
   different input scores 1. Common head and tail are peeled off first, so the
   quadratic part only runs on the stretch that actually differs.
3. Every aligned row gets a status:
       same            same step, same input, same output
       output_differs  same step, same input, different output or error
       input_differs   same step, different input
       only_a/only_b   the step exists in one run only
4. The first row that is not "same" is the first divergence. Everything after
   it is downstream of it.

Timing, token counts and cost are never compared: they vary run to run and are
consequences, not causes. Notes are skipped. Use `ignore` to drop volatile
fields inside payloads (e.g. "output.request_id").
"""
import difflib

from . import events as E

SAME = "same"
OUTPUT_DIFFERS = "output_differs"
INPUT_DIFFERS = "input_differs"
ONLY_A = "only_a"
ONLY_B = "only_b"

DEFAULT_KINDS = (E.RUN_START, E.LLM_CALL, E.TOOL_CALL, E.DECISION, E.RUN_END)
MAX_DP_CELLS = 4_000_000
MAX_CHANGES = 60


def diff_runs(events_a, events_b, ignore=(), kinds=DEFAULT_KINDS, names=("run A", "run B")):
    """Compare two event lists. Returns a JSON-ready dict (see module docstring).

    names  how the two runs are called in the first-divergence wording.
    """
    ignore = tuple(ignore or ())
    a = [_Item(ev, ignore) for ev in events_a if ev.get("kind") in kinds]
    b = [_Item(ev, ignore) for ev in events_b if ev.get("kind") in kinds]

    rows = []
    counts = {SAME: 0, OUTPUT_DIFFERS: 0, INPUT_DIFFERS: 0, ONLY_A: 0, ONLY_B: 0}
    for ia, ib in _align(a, b):
        xa = a[ia] if ia is not None else None
        xb = b[ib] if ib is not None else None
        if xa is None:
            status = ONLY_B
        elif xb is None:
            status = ONLY_A
        elif xa.in_hash != xb.in_hash:
            status = INPUT_DIFFERS
        elif xa.out_hash != xb.out_hash:
            status = OUTPUT_DIFFERS
        else:
            status = SAME
        ref = xa or xb
        row = {"i": len(rows), "status": status, "kind": ref.kind, "name": ref.name,
               "a": xa.seq if xa else None, "b": xb.seq if xb else None}
        if status in (INPUT_DIFFERS, OUTPUT_DIFFERS):
            row["changes"] = changes(xa.view, xb.view)
        rows.append(row)
        counts[status] += 1

    first = next((r for r in rows if r["status"] != SAME), None)
    sa, sb = E.summarize(events_a), E.summarize(events_b)
    return {
        "a": sa, "b": sb, "rows": rows, "counts": counts,
        "identical": first is None,
        "first_divergence": _describe(first, rows, names) if first else None,
        "delta": {
            "steps": sb["steps"] - sa["steps"],
            "cost_usd": round(sb["cost_usd"] - sa["cost_usd"], 6),
            "tokens": (sb["tokens_in"] + sb["tokens_out"]) - (sa["tokens_in"] + sa["tokens_out"]),
            "duration_s": _dur(sb) - _dur(sa),
        },
    }


def first_divergence(events_a, events_b, ignore=(), kinds=DEFAULT_KINDS):
    """Just the first divergence (None when the runs match)."""
    return diff_runs(events_a, events_b, ignore, kinds)["first_divergence"]


class _Item:
    __slots__ = ("seq", "kind", "name", "shape", "in_hash", "out_hash", "view")

    def __init__(self, ev, ignore):
        self.seq = ev.get("seq")
        self.kind = ev.get("kind")
        self.name = ev.get("name")
        view = E.strip_paths(ev, ignore)
        if self.kind == E.RUN_END:
            view["status"] = ev.get("status")
        self.view = view
        # Run boundaries pair up regardless of the run's name.
        self.shape = (self.kind,) if self.kind in (E.RUN_START, E.RUN_END) \
            else (self.kind, self.name)
        self.in_hash = E.digest(view["input"])
        self.out_hash = E.digest([view["output"], view["error"], view.get("status")])


# ---------------------------------------------------------------- alignment

def _align(a, b):
    """List of (index_in_a | None, index_in_b | None), in order."""
    n, m = len(a), len(b)
    head = 0
    while head < n and head < m and _exact(a[head], b[head]):
        head += 1
    tail = 0
    while tail < n - head and tail < m - head and _exact(a[n - 1 - tail], b[m - 1 - tail]):
        tail += 1
    pairs = [(i, i) for i in range(head)]
    mid_a, mid_b = a[head:n - tail], b[head:m - tail]
    if len(mid_a) * len(mid_b) <= MAX_DP_CELLS:
        mid = _align_dp(mid_a, mid_b)
    else:
        mid = _align_fast(mid_a, mid_b)
    pairs += [(None if i is None else i + head, None if j is None else j + head)
              for i, j in mid]
    pairs += [(n - tail + k, m - tail + k) for k in range(tail)]
    return pairs


def _exact(x, y):
    return x.shape == y.shape and x.in_hash == y.in_hash


def _align_dp(a, b):
    n, m = len(a), len(b)
    if not n or not m:
        return [(i, None) for i in range(n)] + [(None, j) for j in range(m)]
    # best[i][j] = best score aligning a[i:] with b[j:]
    best = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        row, below = best[i], best[i + 1]
        shape, in_hash = a[i].shape, a[i].in_hash
        for j in range(m - 1, -1, -1):
            score = below[j] if below[j] >= row[j + 1] else row[j + 1]
            y = b[j]
            if y.shape == shape:
                paired = below[j + 1] + (3 if y.in_hash == in_hash else 1)
                if paired > score:
                    score = paired
            row[j] = score
    out, i, j = [], 0, 0
    while i < n and j < m:
        x, y = a[i], b[j]
        if x.shape == y.shape and \
                best[i][j] == best[i + 1][j + 1] + (3 if x.in_hash == y.in_hash else 1):
            out.append((i, j))
            i += 1
            j += 1
        elif best[i + 1][j] >= best[i][j + 1]:
            out.append((i, None))
            i += 1
        else:
            out.append((None, j))
            j += 1
    out += [(k, None) for k in range(i, n)]
    out += [(None, k) for k in range(j, m)]
    return out


def _align_fast(a, b):
    """Fallback for very long differing stretches: align on shape only."""
    sm = difflib.SequenceMatcher(None, [x.shape for x in a], [y.shape for y in b],
                                 autojunk=False)
    out = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            out += [(i1 + k, j1 + k) for k in range(i2 - i1)]
        else:
            out += [(i, None) for i in range(i1, i2)]
            out += [(None, j) for j in range(j1, j2)]
    return out


# ---------------------------------------------------------------- field diff

def changes(a, b, limit=MAX_CHANGES):
    """Field-level differences between two JSON values.

    Returns [{"path": "output.days", "op": "changed|added|removed", "a": .., "b": ..}].
    """
    out = []
    _walk(a, b, "", out, limit)
    return out


def _walk(a, b, path, out, limit):
    if len(out) >= limit:
        return
    if isinstance(a, dict) and isinstance(b, dict):
        for k in list(a) + [k for k in b if k not in a]:
            sub = path + "." + k if path else k
            if k not in b:
                if a[k] is not None:
                    out.append({"path": sub, "op": "removed", "a": a[k], "b": None})
            elif k not in a:
                if b[k] is not None:
                    out.append({"path": sub, "op": "added", "a": None, "b": b[k]})
            else:
                _walk(a[k], b[k], sub, out, limit)
            if len(out) >= limit:
                return
    elif isinstance(a, list) and isinstance(b, list):
        for i in range(max(len(a), len(b))):
            sub = "%s.%d" % (path, i) if path else str(i)
            if i >= len(b):
                out.append({"path": sub, "op": "removed", "a": a[i], "b": None})
            elif i >= len(a):
                out.append({"path": sub, "op": "added", "a": None, "b": b[i]})
            else:
                _walk(a[i], b[i], sub, out, limit)
            if len(out) >= limit:
                return
    elif E.canon(a) != E.canon(b):
        op = "added" if a is None else "removed" if b is None else "changed"
        out.append({"path": path, "op": op, "a": a, "b": b})


# ---------------------------------------------------------------- verdict

def _describe(first, rows, names):
    status, kind, name = first["status"], first["kind"], first["name"]
    before = first["i"]
    what = _label(kind, name)
    if kind == E.RUN_START:
        headline = "The runs started with different inputs"
        why = ("They were given different work to do, so every later difference "
               "may simply follow from that.")
    elif kind == E.RUN_END:
        headline = "Same steps, different ending"
        why = ("Every recorded step matches; only the final output or status differs. "
               "The cause is in code that runs after the last recorded step.")
    elif status == OUTPUT_DIFFERS and kind == E.TOOL_CALL:
        headline = "Same call, different result from %s" % what
        why = ("Both runs called it with identical input and got different answers. "
               "The tool, or the data behind it, changed. The agent did nothing "
               "different up to here.")
    elif status == OUTPUT_DIFFERS and kind == E.LLM_CALL:
        headline = "Same prompt, different completion from %s" % what
        why = ("The model saw identical input in both runs and answered differently. "
               "This is model non-determinism or a model/version change, not a change "
               "in what the agent sent.")
    elif status == OUTPUT_DIFFERS and kind == E.DECISION:
        headline = "The agent decided differently at %s" % what
        why = ("Everything recorded before this point matches, so the decision "
               "depends on something that was not recorded, or the agent's code changed.")
    elif status == OUTPUT_DIFFERS:
        headline = "Same input, different output at %s" % what
        why = "Both runs reached this step identically and it produced different results."
    elif status == INPUT_DIFFERS:
        headline = "The agent sent different input to %s" % what
        why = ("Every earlier step matches, yet the agent called this one differently. "
               "The cause is in the agent itself: changed code or config, or state "
               "that is not being recorded.")
    elif status == ONLY_A:
        headline = "%s did %s here; %s did not" % (_cap(names[0]), what, names[1])
        why = ("Every earlier step matches. %s skipped this step or took another path."
               % _cap(names[1]))
    else:
        headline = "%s did %s here; %s did not" % (_cap(names[1]), what, names[0])
        why = ("Every earlier step matches. %s added this step or took another path."
               % _cap(names[1]))
    downstream = sum(1 for r in rows[first["i"] + 1:] if r["status"] != SAME)
    return {
        "row": first["i"], "status": status, "kind": kind, "name": name,
        "a": first["a"], "b": first["b"],
        "matched_before": before,
        "downstream_differences": downstream,
        "headline": headline, "why": why,
        "changes": first.get("changes", []),
    }


def _cap(text):
    return text[:1].upper() + text[1:]


def _label(kind, name):
    return "%s `%s`" % ({E.TOOL_CALL: "tool", E.LLM_CALL: "model", E.DECISION: "decision"}
                        .get(kind, kind), name)


def _dur(summary):
    if summary["started_at"] is None or summary["ended_at"] is None:
        return 0.0
    return round(summary["ended_at"] - summary["started_at"], 3)


# ---------------------------------------------------------------- text report

def format_report(result, max_rows=200):
    """Plain-text (ASCII) report of a diff_runs() result."""
    lines = []
    for tag in ("a", "b"):
        s = result[tag]
        lines.append("%s  %-24s %-20s %-7s %3d steps  $%.4f" % (
            tag.upper(), s["run_id"] or "?", (s["name"] or "")[:20], s["status"],
            s["steps"], s["cost_usd"]))
    lines.append("")
    fd = result["first_divergence"]
    if fd is None:
        lines.append("IDENTICAL: no divergence in %d compared steps." % len(result["rows"]))
        return "\n".join(lines)
    lines.append("FIRST DIVERGENCE at row %d: %s" % (fd["row"], fd["headline"]))
    lines.append("  %s" % fd["why"])
    lines.append("  %d row(s) before this match exactly; %d later row(s) also differ."
                 % (fd["matched_before"], fd["downstream_differences"]))
    for ch in fd["changes"][:12]:
        lines.append("    %-9s %s:  %s  ->  %s" % (ch["op"], ch["path"],
                                                 _short(ch["a"]), _short(ch["b"])))
    if len(fd["changes"]) > 12:
        lines.append("    ... %d more field(s)" % (len(fd["changes"]) - 12))
    lines.append("")
    marks = {SAME: "  ", OUTPUT_DIFFERS: "!=", INPUT_DIFFERS: "!=", ONLY_A: "A ", ONLY_B: " B"}
    notes = {SAME: "", OUTPUT_DIFFERS: "output differs", INPUT_DIFFERS: "input differs",
             ONLY_A: "only in A", ONLY_B: "only in B"}
    lines.append(" row  A#   B#       step")
    for r in result["rows"][:max_rows]:
        lines.append("%4d  %-4s %-4s %s  %-10s %-28s %s%s" % (
            r["i"], "-" if r["a"] is None else r["a"], "-" if r["b"] is None else r["b"],
            marks[r["status"]], r["kind"], (r["name"] or "")[:28], notes[r["status"]],
            "   <-- first divergence" if r["i"] == fd["row"] else ""))
    if len(result["rows"]) > max_rows:
        lines.append("  ... %d more rows" % (len(result["rows"]) - max_rows))
    return "\n".join(lines)


def _short(value, width=70):
    text = value if isinstance(value, str) else E.canon(value)
    text = text.replace("\n", "\\n")
    return text if len(text) <= width else text[:width - 3] + "..."
