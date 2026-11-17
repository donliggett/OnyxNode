"""A small, fully offline demo agent, so there is something to record and diff.

It is a customer-support refund agent with a rule-based stand-in for the LLM
and fake tools. Nothing here calls a real model or service, and the prices in
PRICING are invented for the demo.

    onyx demo                                   record a good run and a bad run
    onyx suite run --agent onyxnode.demo:agent  replay recorded cases against it

The bad run differs from the good one in exactly one place: the order lookup
returns a stale delivery date. Everything after that (policy check, decision,
reply) goes wrong as a consequence, which is what the first-divergence finder
is for.

Set ONYX_DEMO_BUG=1 to make the agent skip the policy check, to see a
regression case fail.
"""
import os
import time

from .recorder import LLMResult, Recorder

MODEL = "demo-model"
PRICING = {MODEL: (3.0, 15.0)}  # invented: USD per million input / output tokens

TICKET = {
    "ticket_id": "T-4411",
    "customer": "dana@example.com",
    "order_id": "A-1017",
    "message": "My headphones arrived with a cracked earcup. I would like a refund.",
}


class World:
    """The outside world the tools talk to."""

    def __init__(self, stale_orders=False):
        self.stale_orders = stale_orders
        self.sent = []

    def lookup_order(self, order_id):
        time.sleep(0.06)
        return {"order_id": order_id, "item": "Studio headphones", "category": "electronics",
                "amount": 89.0, "currency": "USD",
                "delivered_days_ago": 45 if self.stale_orders else 12,
                "source": "orders-replica-2" if self.stale_orders else "orders-primary"}

    def check_refund_policy(self, category, delivered_days_ago, reason):
        time.sleep(0.03)
        window = 30
        eligible = delivered_days_ago <= window
        return {"eligible": eligible, "window_days": window,
                "rule": "damaged-on-arrival" if reason == "damaged" else "standard",
                "detail": "inside the return window" if eligible
                else "delivered %d days ago, window is %d" % (delivered_days_ago, window)}

    def issue_refund(self, order_id, amount):
        time.sleep(0.09)
        return {"refund_id": "R-" + order_id[2:], "amount": amount, "status": "issued"}

    def send_email(self, to, subject, body):
        time.sleep(0.04)
        self.sent.append((to, subject, body))
        return {"delivered": True}


def fake_llm(messages):
    """Rule-based stand-in for a model call. Deterministic on purpose."""
    time.sleep(0.12)
    task, text = messages[0]["content"], messages[-1]["content"]
    if task.startswith("Classify"):
        damaged = any(w in text.lower() for w in ("cracked", "broken", "damaged"))
        out = {"intent": "refund_request" if "refund" in text.lower() else "other",
               "reason": "damaged" if damaged else "unspecified"}
    elif "resolution: refund" in text:
        out = ("Hi Dana, I'm sorry your headphones arrived damaged. I've issued a full "
               "refund of $89.00; it should reach you within 5 business days.")
    else:
        out = ("Hi Dana, I'm sorry about the trouble with your headphones. Unfortunately "
               "this order is outside our 30 day return window, so I can't offer a refund.")
    tokens_in = sum(len(m["content"]) for m in messages) // 4
    return LLMResult(out, tokens_in=tokens_in, tokens_out=len(str(out)) // 4)


def agent(run, ticket, world=None):
    """The agent entry point: (run, input) -> final output."""
    world = world or World()

    intent = run.llm(MODEL, fake_llm, [
        {"role": "system", "content": "Classify the customer's intent and reason."},
        {"role": "user", "content": ticket["message"]}])
    run.decision("intent", choice=intent["intent"], options=["refund_request", "other"])
    if intent["intent"] != "refund_request":
        return {"resolution": "handoff"}

    order = run.tool("lookup_order", world.lookup_order, order_id=ticket["order_id"])

    if os.environ.get("ONYX_DEMO_BUG"):
        eligible, why = True, "policy check skipped"
    else:
        policy = run.tool("check_refund_policy", world.check_refund_policy,
                          category=order["category"],
                          delivered_days_ago=order["delivered_days_ago"],
                          reason=intent["reason"])
        eligible, why = policy["eligible"], policy["detail"]

    resolution = run.decision("resolution", choice="refund" if eligible else "deny",
                              reason=why, options=["refund", "deny"])
    refund = None
    if resolution == "refund":
        refund = run.tool("issue_refund", world.issue_refund,
                          order_id=order["order_id"], amount=order["amount"])

    reply = run.llm(MODEL, fake_llm, [
        {"role": "system", "content": "Write a short, kind reply to the customer."},
        {"role": "user", "content": "%s\nresolution: %s (%s)"
                                    % (ticket["message"], resolution, why)}])
    run.tool("send_email", world.send_email, to=ticket["customer"],
             subject="Your order %s" % order["order_id"], body=reply)
    return {"resolution": resolution, "refund": refund, "reply": reply}


def record(sink=None, stale_orders=False):
    """Record one demo run. Returns its run id."""
    rec = Recorder("refund-agent", sink=sink, pricing=PRICING)
    with rec.start(input=TICKET, tags=["demo"]) as run:
        run.finish(output=agent(run, TICKET, World(stale_orders=stale_orders)))
    return run.run_id
