"""serve/test_response_store.py - stored and background Responses (serve/response_store.py, serve/response_jobs.py):
the store's own tests (ported from Project Maya, MIT) and the HTTP routes against the mock engine (no GPU).

    python -m unittest serve.test_response_store -v
"""
from __future__ import annotations

import json
import sys
import threading
import time
import unittest
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve import helpers as H  # noqa: E402
from serve.frontend import ChatTemplate  # noqa: E402
from serve.response_store import ResponseStore, StoreBusy, input_items, input_page, limits_from_env  # noqa: E402
from serve.server import ByteTokenizer, MockEngine, Service, serve  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def item(index, content="x" * 2048):
    return {"id": f"msg_{index}", "type": "message", "role": "user", "content": content}


class Store(unittest.TestCase):
    """Project Maya's serve/test_response_store.py (MIT), the parts that do not depend on Maya's config names."""

    def test_shared_blocks_survive_ancestor_deletion_and_owner_scopes_are_distinct(self):
        store = ResponseStore()
        history = [item(1)]
        first = {"id": "r1", "status": "completed", "output": [item(2)]}
        second = {"id": "r2", "status": "completed", "output": [item(4)]}
        store.put("a", first, history)
        store.put("a", second, history + first["output"] + [item(3)])
        self.assertEqual(store.stats()["blocks"], 4)
        store.delete("a", "r1")
        self.assertEqual(store.get("a", "r2")["input"], [item(1), item(2), item(3)])
        store.put("b", second, [item(1), item(2), item(3)])
        self.assertEqual(store.stats()["blocks"], 8)
        with self.assertRaises(KeyError):
            store.get("c", "r2")
        store.delete("a", "r2")
        self.assertEqual(store.stats()["blocks"], 4)
        store.delete("b", "r2")
        self.assertEqual((store.stats()["blocks"], store.stats()["bytes"]), (0, 0))

    def test_failed_capacity_planning_does_not_evict_good_records(self):
        store = ResponseStore(max_bytes=8192, owner_bytes=1024)
        store.put("good", {"id": "old", "output": []}, [])
        before = store.stats()["bytes"]
        with self.assertRaises(ValueError):
            store.put("large", {"id": "new", "output": [item(1)]}, [])
        self.assertEqual(store.stats()["bytes"], before)
        self.assertEqual(store.get("good", "old")["response"]["id"], "old")

    def test_ttl_and_concurrent_readers(self):
        now = [0]
        store = ResponseStore(ttl=10, clock=lambda: now[0])
        store.put("a", {"id": "r", "output": []}, [item(1)])
        now[0] = 10
        with self.assertRaises(KeyError):
            store.get("a", "r")
        store = ResponseStore(max_entries=16)

        def work(index):
            for revision in range(10):
                store.put(str(index), {"id": "r", "output": [item(revision)]}, [item(100)])
                self.assertEqual(store.get(str(index), "r")["input"][0]["id"], "msg_100")
            store.delete(str(index), "r")
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(work, range(8)))
        self.assertEqual((store.stats()["bytes"], store.stats()["blocks"]), (0, 0))

    def test_assembly_limit_and_limits_from_env(self):
        store = ResponseStore(assembly_bytes=128)
        store.put("a", {"id": "r", "output": []}, [item(1)])
        with self.assertRaises(StoreBusy):
            store.get("a", "r")
        self.assertEqual(limits_from_env({"STRATA_RESPONSES_OWNER_ENTRIES": "2"})["OWNER_ENTRIES"], 2)
        for value in ("0", "-1", "bad"):
            with self.assertRaises(ValueError):
                limits_from_env({"STRATA_RESPONSES_CACHE_BYTES": value})

    def test_input_items_and_pages(self):
        items = input_items("hello")
        self.assertEqual(items[0]["content"], [{"type": "input_text", "text": "hello"}])
        self.assertTrue(items[0]["id"].startswith("item_"))
        many = input_items([{"role": "user", "content": str(i)} for i in range(5)])
        page = input_page(many, {"limit": ["2"], "order": ["asc"]})
        self.assertEqual(([x["content"][0]["text"] for x in page["data"]], page["has_more"]), (["0", "1"], True))
        page = input_page(many, {"after": [page["last_id"]], "order": ["asc"]})
        self.assertEqual([x["content"][0]["text"] for x in page["data"]], ["2", "3", "4"])
        with self.assertRaises(ValueError):
            input_page(many, {"limit": ["0"]})


class Order(unittest.TestCase):
    def test_a_background_response_waits_but_is_not_stopped(self):
        o = H.MainFirst()
        c = threading.Event()
        e = o.enter(c, True, stoppable=False)
        m = o.enter(threading.Event(), False)
        self.assertFalse(c.is_set())
        self.assertFalse(o.acquire(e, threading.Lock(), 0.3))       # but it does not take the turn first
        o.leave(m)
        o.leave(e)


class Http(unittest.TestCase):
    def start(self, script="</think>\n\nThe answer.", delay=0.0, key=""):
        tok = ByteTokenizer()
        self.engine = MockEngine(tok, script, max_context=16384, delay_s=delay)
        self.svc = Service(self.engine, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        self.svc.api_key = key
        self.httpd = serve(self.svc, port=0)
        self.addCleanup(self.httpd.server_close)
        self.addCleanup(self.httpd.shutdown)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def call(self, method, path, body=None, key=None, raw=False):
        headers = {"Content-Type": "application/json"}
        if key:
            headers["Authorization"] = f"Bearer {key}"
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode() if body is not None else None,
                                     headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                data = r.read()
                return r.status, data.decode() if raw else json.loads(data)
        except urllib.error.HTTPError as e:
            with e:
                return e.code, json.loads(e.read())

    def prompt(self):
        return bytes(t for t in self.engine.last_prompt if t < 256).decode("utf-8", "replace")

    def test_store_retrieve_chain_and_delete(self):
        self.start()
        s, r1 = self.call("POST", "/v1/responses", {"model": "m", "input": "first question", "store": True})
        self.assertEqual(s, 200, r1)
        self.assertTrue(r1["store"])
        s, got = self.call("GET", f"/v1/responses/{r1['id']}")
        self.assertEqual((s, got["id"], got["output"][-1]["content"][0]["text"]), (200, r1["id"], "The answer."))
        s, items = self.call("GET", f"/v1/responses/{r1['id']}/input_items")
        self.assertEqual((s, items["data"][0]["content"][0]["text"]), (200, "first question"))
        s, r2 = self.call("POST", "/v1/responses", {"model": "m", "input": "second question", "store": True,
                                                    "previous_response_id": r1["id"]})
        self.assertEqual(s, 200, r2)
        self.assertEqual(r2["previous_response_id"], r1["id"])
        p = self.prompt()
        self.assertLess(p.index("first question"), p.index("The answer."))
        self.assertLess(p.index("The answer."), p.index("second question"))
        s, items = self.call("GET", f"/v1/responses/{r2['id']}/input_items?order=asc")
        self.assertEqual(len(items["data"]), 3)              # question, answer, question
        s, d = self.call("DELETE", f"/v1/responses/{r1['id']}")
        self.assertEqual((s, d), (200, {"id": r1["id"], "object": "response.deleted", "deleted": True}))
        s, _ = self.call("GET", f"/v1/responses/{r1['id']}")
        self.assertEqual(s, 404)
        s, _ = self.call("GET", f"/v1/responses/{r2['id']}")    # its own copy of the history stays
        self.assertEqual(s, 200)

    def test_not_stored_unless_asked(self):
        self.start()
        s, r = self.call("POST", "/v1/responses", {"model": "m", "input": "hi"})
        self.assertEqual((s, r["store"]), (200, False))
        s, _ = self.call("GET", f"/v1/responses/{r['id']}")
        self.assertEqual(s, 404)
        s, e = self.call("POST", "/v1/responses", {"model": "m", "input": "x", "previous_response_id": r["id"]})
        self.assertEqual((s, e["error"]["param"]), (400, "previous_response_id"))
        self.svc.responses_store_default = True              # "responses_store_default": true
        s, r = self.call("POST", "/v1/responses", {"model": "m", "input": "hi"})
        self.assertEqual((s, r["store"]), (200, True))
        s, e = self.call("POST", "/v1/responses", {"model": "m", "input": "x", "background": True, "store": False})
        self.assertEqual((s, e["error"]["param"]), (400, "background"))

    def test_responses_belong_to_their_key(self):
        self.start(key=["alpha", "beta"])
        s, r = self.call("POST", "/v1/responses", {"model": "m", "input": "hi", "store": True}, key="alpha")
        self.assertEqual(s, 200, r)
        self.assertEqual(self.call("GET", f"/v1/responses/{r['id']}", key="alpha")[0], 200)
        self.assertEqual(self.call("GET", f"/v1/responses/{r['id']}", key="beta")[0], 404)
        self.assertEqual(self.call("DELETE", f"/v1/responses/{r['id']}", key="beta")[0], 404)
        self.assertEqual(self.call("GET", f"/v1/responses/{r['id']}")[0], 401)

    def wait(self, rid, want=("completed",), limit=30):
        deadline = time.monotonic() + limit
        while time.monotonic() < deadline:
            s, r = self.call("GET", f"/v1/responses/{rid}")
            if r.get("status") in want:
                return r
            time.sleep(0.1)
        self.fail(f"{rid} stayed {r.get('status')}")

    def test_background_poll_and_replay(self):
        self.start(delay=0.01)
        s, r = self.call("POST", "/v1/responses", {"model": "m", "input": "hi", "background": True, "store": True})
        self.assertEqual(s, 200, r)
        self.assertIn(r["status"], ("queued", "in_progress"))
        self.assertTrue(r["background"])
        done = self.wait(r["id"])
        self.assertEqual(done["output"][-1]["content"][0]["text"], "The answer.")
        s, text = self.call("GET", f"/v1/responses/{r['id']}?stream=true", raw=True)
        names = [line[7:] for line in text.splitlines() if line.startswith("event: ")]
        self.assertEqual((names[0], names[-1]), ("response.queued", "response.completed"))
        s, text = self.call("GET", f"/v1/responses/{r['id']}?stream=true&starting_after=2", raw=True)
        seqs = [json.loads(line[6:])["sequence_number"] for line in text.splitlines() if line.startswith("data: ")]
        self.assertEqual(seqs[0], 3)
        s, chained = self.call("POST", "/v1/responses", {"model": "m", "input": "more", "store": True,
                                                         "previous_response_id": r["id"]})
        self.assertEqual(s, 200, chained)
        self.assertIn("The answer.", self.prompt())

    def test_background_cancel(self):
        self.start(script="</think>\n\n" + "z" * 400, delay=0.02)
        s, r = self.call("POST", "/v1/responses", {"model": "m", "input": "hi", "background": True, "store": True})
        self.wait(r["id"], ("in_progress",))
        s, c = self.call("POST", f"/v1/responses/{r['id']}/cancel", {})
        self.assertEqual((s, c["status"]), (200, "cancelled"))
        s, missing = self.call("POST", "/v1/responses/resp_nope/cancel", {})
        self.assertEqual((s, missing["error"]["code"]), (404, "not_found"))


if __name__ == "__main__":
    unittest.main()
