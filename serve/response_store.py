"""Stored Responses (0.94): `store`, `previous_response_id`, retrieve / delete / input_items.

Ported from Project Maya (serve/response_store.py, MIT License, Copyright (c) 2026 Peasant Smith and the Project Maya
contributors, Copyright (c) 2026 Niko1221 and the Strata contributors).  Changes: the limits live here (LIMITS, from
STRATA_RESPONSES_* environment variables), input_items / input_page come along from Maya's serve/responses.py, and
`expand` keeps the stored instructions out (OpenAI's rule: instructions do not carry over).

Credential-scoped snapshots sharing immutable JSON item blocks without parent dependencies: everything is in memory,
bounded by entries, bytes and a time to live, and keyed by the hash of the API key that wrote it.
"""

from collections import Counter, OrderedDict
from contextlib import contextmanager
import hashlib
import json
import threading
import os
import time
import uuid


def encoded(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


class StoreBusy(ValueError):
    pass


class ResponseStore:
    def __init__(self, max_entries=128, max_bytes=64 * 1024 * 1024, ttl=3600, clock=time.monotonic,
                 max_entry_bytes=None, owner_entries=0, owner_bytes=0, assembly_bytes=None):
        self.max_entries, self.max_bytes, self.ttl, self.clock = max_entries, max_bytes, ttl, clock
        self.max_entry_bytes = max_bytes if max_entry_bytes is None else max_entry_bytes
        self.owner_entries, self.owner_bytes = owner_entries, owner_bytes
        self.assembly_limit = max_bytes if assembly_bytes is None else assembly_bytes
        self._entries, self._blocks = OrderedDict(), {}
        self._bytes, self._inflight, self._inflight_peak = 0, 0, 0
        self._owners, self._lock = {}, threading.RLock()
        self._metrics = Counter({name: 0 for name in ("hits", "misses", "writes", "expired", "evicted", "deleted",
                                                     "rejected", "assembly_rejected", "write_seconds", "read_seconds",
                                                     "lock_wait_seconds", "lock_hold_seconds")})

    @contextmanager
    def _guard(self):
        started = time.perf_counter()
        with self._lock:
            acquired = time.perf_counter()
            self._metrics["lock_wait_seconds"] += acquired - started
            try:
                yield
            finally:
                self._metrics["lock_hold_seconds"] += time.perf_counter() - acquired

    @classmethod
    def from_limits(cls, limits):
        return cls(max_entries=limits["CACHE_ENTRIES"], max_bytes=limits["CACHE_BYTES"], ttl=limits["CACHE_TTL_S"],
                   max_entry_bytes=limits["ENTRY_BYTES"], owner_entries=limits["OWNER_ENTRIES"],
                   owner_bytes=limits["OWNER_BYTES"], assembly_bytes=limits["ASSEMBLY_BYTES"])

    def _drop(self, key, reason=None):
        entry = self._entries.pop(key)
        freed = entry["size"]
        for block in entry["input"] + entry["output"]:
            self._blocks[block][1] -= 1
            if self._blocks[block][1] == 0:
                freed += len(self._blocks.pop(block)[0])
        self._bytes -= freed
        self._owners[key[0]] -= freed
        if self._owners[key[0]] == 0:
            del self._owners[key[0]]
        if reason:
            self._metrics[reason] += 1

    def _expire(self):
        now = self.clock()
        for key, entry in list(self._entries.items()):
            if entry["deadline"] <= now:
                self._drop(key, "expired")

    def put(self, owner, response, items):
        started = time.perf_counter()
        output = response.get("output", [])
        metadata = encoded({key: value for key, value in response.items() if key != "output"})
        blocks, refs = {}, []
        for group in (items, output):
            group_refs = []
            for item in group:
                data = encoded(item)
                block = owner, hashlib.sha256(data).hexdigest()
                blocks[block] = data
                group_refs.append(block)
            refs.append(tuple(group_refs))
        descriptor = len(metadata) + sum(len(encoded([key[1] for key in group])) for group in refs)
        logical_size = len(metadata) + sum(len(blocks[key]) for group in refs for key in group)
        key = owner, response["id"]
        with self._guard():
            self._expire()
            if logical_size > self.max_entry_bytes or self.max_entries < 1:
                self._metrics["rejected"] += 1
                raise ValueError("stored response exceeds the in-memory Responses cache capacity; use store=false")
            # Plan all evictions first so an impossible write leaves existing snapshots intact.
            counts = {block: record[1] for block, record in self._blocks.items()}
            size, owner_size = self._bytes, self._owners.get(owner, 0)
            remaining = list(self._entries)
            removed = []

            def discard(victim):
                nonlocal size, owner_size
                entry = self._entries[victim]
                freed = entry["size"]
                for block in entry["input"] + entry["output"]:
                    counts[block] -= 1
                    if counts[block] == 0:
                        freed += len(self._blocks[block][0])
                size -= freed
                if victim[0] == owner:
                    owner_size -= freed
                remaining.remove(victim)
                removed.append(victim)

            if key in self._entries:
                discard(key)
            added = descriptor
            for group in refs:
                for block in group:
                    if counts.get(block, 0) == 0:
                        added += len(blocks[block])
                    counts[block] = counts.get(block, 0) + 1
            size += added
            owner_size += added
            while True:
                quota = ((self.owner_entries and sum(k[0] == owner for k in remaining) + 1 > self.owner_entries) or
                         (self.owner_bytes and owner_size > self.owner_bytes))
                if not quota and len(remaining) + 1 <= self.max_entries and size <= self.max_bytes:
                    break
                victim = next((k for k in remaining if not quota or k[0] == owner), None)
                if victim is None:
                    self._metrics["rejected"] += 1
                    raise ValueError("stored response exceeds the in-memory Responses cache capacity; use store=false")
                discard(victim)
            for victim in removed:
                self._drop(victim, "evicted" if victim != key else None)
            actual = descriptor
            for group in refs:
                for block in group:
                    if block not in self._blocks:
                        self._blocks[block] = [blocks[block], 0]
                        actual += len(blocks[block])
                    self._blocks[block][1] += 1
            self._entries[key] = {"metadata": metadata, "input": refs[0], "output": refs[1],
                                  "size": descriptor, "deadline": self.clock() + self.ttl}
            self._bytes += actual
            self._owners[owner] = self._owners.get(owner, 0) + actual
            self._metrics["writes"] += 1
            self._metrics["write_seconds"] += time.perf_counter() - started

    def get(self, owner, response_id):
        started = time.perf_counter()
        with self._guard():
            self._expire()
            key = owner, response_id
            if key not in self._entries:
                self._metrics["misses"] += 1
                raise KeyError(response_id)
            entry = self._entries[key]
            groups = [tuple(self._blocks[block][0] for block in entry[group]) for group in ("input", "output")]
            metadata = entry["metadata"]
            assembly = len(metadata) + sum(len(data) for group in groups for data in group)
            if self._inflight + assembly > self.assembly_limit:
                self._metrics["assembly_rejected"] += 1
                raise StoreBusy("Responses snapshot assembly is at capacity; retry later")
            self._inflight += assembly
            self._inflight_peak = max(self._inflight_peak, self._inflight)
            self._entries.move_to_end(key)
            self._metrics["hits"] += 1
        try:
            response = json.loads(metadata)
            response["output"] = [json.loads(data) for data in groups[1]]
            return {"response": response, "input": [json.loads(data) for data in groups[0]]}
        finally:
            with self._guard():
                self._inflight -= assembly
                self._metrics["read_seconds"] += time.perf_counter() - started

    def delete(self, owner, response_id):
        with self._guard():
            self._expire()
            key = owner, response_id
            if key not in self._entries:
                raise KeyError(response_id)
            self._drop(key, "deleted")

    def expand(self, owner, req):
        previous = req.get("previous_response_id")
        if previous is None:
            return req
        if not isinstance(previous, str) or not previous:
            raise ValueError("previous_response_id must be a nonempty string")
        stored = self.get(owner, previous)
        added = req.get("input", [])
        if isinstance(added, str):
            added = [{"role": "user", "content": added}]
        if not isinstance(added, list):
            raise ValueError("input must be a string or array of input items")
        return {**req, "input": stored["input"] + stored["response"]["output"] + added}

    def stats(self):
        with self._guard():
            self._expire()
            return {**dict(self._metrics), "entries": len(self._entries), "blocks": len(self._blocks),
                    "bytes": self._bytes, "inflight_bytes": self._inflight, "peak_inflight_bytes": self._inflight_peak,
                    "max_entries": self.max_entries, "max_bytes": self.max_bytes, "ttl_s": self.ttl}


# ------------------------------------------------------------------------------------------------ limits (Maya's
# serve/response_config.py; STRATA_RESPONSES_<NAME> overrides one)
LIMITS_DEFAULT = {
    "CACHE_ENTRIES": 128, "CACHE_BYTES": 64 * 1024 * 1024, "CACHE_TTL_S": 3600,
    "ENTRY_BYTES": 64 * 1024 * 1024, "OWNER_ENTRIES": 0, "OWNER_BYTES": 0,
    "ASSEMBLY_BYTES": 64 * 1024 * 1024,
    "JOB_WORKERS": 2, "JOB_ENTRIES": 16, "JOB_OWNER_ENTRIES": 4,
    "JOB_TIMEOUT_S": 3600, "JOB_TTL_S": 900, "JOB_INPUT_BYTES": 16 * 1024 * 1024,
    "EVENT_BYTES": 8 * 1024 * 1024, "EVENT_COUNT": 65536,
    "STREAM_READERS": 32, "STREAM_OWNER_READERS": 8, "STREAM_TIMEOUT_S": 5,
}
ENV_PREFIX = "STRATA_RESPONSES_"


def limits_from_env(env=None):
    env = os.environ if env is None else env
    limits = {}
    for name, default in LIMITS_DEFAULT.items():
        raw = env.get(ENV_PREFIX + name, str(default))
        try:
            value = int(raw)
        except (TypeError, ValueError):
            raise ValueError(f"{ENV_PREFIX}{name} must be an integer") from None
        if value < 0 or (value == 0 and name not in ("OWNER_ENTRIES", "OWNER_BYTES")):
            raise ValueError(f"{ENV_PREFIX}{name} must be positive (owner quotas may be zero to disable)")
        limits[name] = value
    if limits["JOB_WORKERS"] > 32 or limits["JOB_ENTRIES"] > 4096:
        raise ValueError("Responses supports at most 32 background workers and 4096 job records")
    if limits["EVENT_BYTES"] < 4096:
        raise ValueError("Responses event buffers must be at least 4096 bytes")
    return limits


# ------------------------------------------------------------------------------------------------ input items (Maya's
# serve/responses.py)
def input_items(items):
    """Give input history stable item IDs and the content shape used by the SDK's item list."""
    if isinstance(items, str):
        items = [{"role": "user", "content": items}]
    out = []
    seen = set()
    for item in json.loads(json.dumps(items)):
        if not isinstance(item, dict):
            raise ValueError("expected an input item object")
        kind = item.setdefault("type", "message")
        item.setdefault("status", "completed")
        if kind == "message" and "content" in item:
            content = item["content"]
            text_type = "output_text" if item.get("role") == "assistant" else "input_text"
            if isinstance(content, str):
                content = [{"type": text_type, "text": content}]
            for part in content if isinstance(content, list) else []:
                if isinstance(part, dict) and part.get("type") in ("input_text", "output_text", "text"):
                    part["type"] = text_type
                    if text_type == "output_text":
                        part.setdefault("annotations", [])
            item["content"] = content
        item.setdefault("id", "item_" + uuid.uuid4().hex[:24])
        if not isinstance(item["id"], str) or not item["id"]:
            raise ValueError("input item ids must be nonempty strings")
        if item["id"] in seen:
            raise ValueError(f"duplicate input item id: {item['id']}")
        seen.add(item["id"])
        out.append(item)
    return out


def input_page(items, query):
    if set(query) - {"after", "limit", "order", "include"} or any(len(v) != 1 for k, v in query.items()
                                                                  if k != "include"):
        raise ValueError("input_items accepts only after, limit and order, once each")
    order = query.get("order", ["desc"])[0]
    if order not in ("asc", "desc"):
        raise ValueError("order must be asc or desc")
    try:
        limit = int(query.get("limit", ["20"])[0])
    except ValueError:
        raise ValueError("limit must be an integer between 1 and 100") from None
    if not 1 <= limit <= 100:
        raise ValueError("limit must be between 1 and 100")
    ordered = items if order == "asc" else list(reversed(items))
    after = query.get("after", [None])[0]
    if after is not None:
        index = next((i for i, item in enumerate(ordered) if item.get("id") == after), None)
        if index is None:
            raise ValueError("after must name an input item of this response")
        ordered = ordered[index + 1:]
    data = ordered[:limit]
    return {"object": "list", "data": data, "has_more": len(ordered) > limit,
            "first_id": data[0]["id"] if data else None, "last_id": data[-1]["id"] if data else None}
