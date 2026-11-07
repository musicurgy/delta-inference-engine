"""Web search and web fetch as server tools (0.94, off until configured).

Anthropic's `web_search_*` and `web_fetch_*` tools (Claude Code's WebSearch sends `web_search_20250305`) and the
Responses API's hosted `web_search` (Codex) run here, inside the answer, through the same loop as the other server
tools (serve/server_tools.py, server.run_with_mcp).  The run config's `"web_tools"` block turns them on and picks the
search backend:

    "web_tools": {
      "search": "searxng",                    # or "brave", "tavily", "mcp"
      "searxng_url": "http://127.0.0.1:8888", # your own SearXNG (JSON output enabled)
      "brave_api_key": "...",                 # "search": "brave"
      "tavily_api_key": "...",                # "search": "tavily"
      "mcp_tool": "server__tool",             # "search": "mcp": a configured MCP server's search tool
      "fetch": true,                          # web fetch (a page the conversation already names)
      "max_uses": 5, "max_results": 5, "snippet_chars": 500, "fetch_max_chars": 20000, "timeout_s": 15
    }

Privacy: a search sends the query (written by the model from your conversation) to the backend you configured; a fetch
downloads the page from its site.  Nothing is on by default.  A fetch goes only to a public address (the server's
public_url check, after DNS and after every redirect), only to a URL that appears in the conversation already (as on
Anthropic's API: a page cannot make the model send your data to an address it made up), and returns text only.
"""
from __future__ import annotations

import base64
import collections
import html
import html.parser
import json
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

from serve import documents
from serve import netguard
from serve import server_tools as ST

DEFAULTS = {"max_uses": 5, "max_results": 5, "snippet_chars": 500, "fetch_max_chars": 20000, "timeout_s": 15.0,
            "fetch_max_bytes": 2 << 20}
BACKENDS = ("searxng", "brave", "tavily", "mcp")
ENC_PREFIX = "strata.ws1:"
USER_AGENT = "Mozilla/5.0 (compatible; Strata web tools)"
MAX_QUERY = 400
MAX_URL = 2000

CONFIG: dict = {}                       # the "web_tools" block, cleaned (empty: web tools off)
_CHECK = None                           # server.public_url: raises ValueError for an address this must not fetch
_PEER = None                            # server.public_peer: the same for the address a connection reached (netguard)
_HUB = None                             # the MCP hub, for "search": "mcp"


class WebError(Exception):
    def __init__(self, code: str, message: str = ""):
        super().__init__(message or code)
        self.code = code


def configure(block, public_url=None, mcp_hub=None, public_peer=None) -> dict:
    """Read the config's "web_tools" block (None: off).  ValueError names a bad setting."""
    global CONFIG, _CHECK, _HUB, _PEER
    _CHECK, _HUB, _PEER = public_url, mcp_hub, public_peer
    if not block:
        CONFIG = {}
        return CONFIG
    if not isinstance(block, dict):
        raise ValueError('"web_tools" must be an object')
    cfg = {**DEFAULTS, **block}
    search = cfg.get("search")
    if search not in (None, "", *BACKENDS):
        raise ValueError(f'web_tools.search must be one of {", ".join(BACKENDS)}, not {search!r}')
    need = {"searxng": "searxng_url", "brave": "brave_api_key", "tavily": "tavily_api_key", "mcp": "mcp_tool"}
    if search and not cfg.get(need[search]):
        raise ValueError(f'web_tools.search "{search}" needs web_tools.{need[search]}')
    if search == "searxng" and not str(cfg["searxng_url"]).startswith(("http://", "https://")):
        raise ValueError("web_tools.searxng_url must be an http(s) address")
    for k in ("max_uses", "max_results", "snippet_chars", "fetch_max_chars", "fetch_max_bytes"):
        if isinstance(cfg[k], bool) or not isinstance(cfg[k], int) or cfg[k] < 1:
            raise ValueError(f"web_tools.{k} must be a whole number of at least 1")
    cfg["fetch"] = cfg.get("fetch") is True
    cfg["search"] = search or None
    CONFIG = cfg
    return CONFIG


def searching() -> bool:
    return bool(CONFIG.get("search"))


def fetching() -> bool:
    return bool(CONFIG.get("fetch"))


def describe() -> str:
    parts = []
    if searching():
        parts.append(f"web search via {CONFIG['search']}")
    if fetching():
        parts.append("web fetch")
    return ", ".join(parts)


# ------------------------------------------------------------------------------------------------ helpers
def _domain_ok(url: str, allowed, blocked) -> bool:
    host = (urllib.parse.urlsplit(url).hostname or "").lower().rstrip(".")

    def match(d):
        d = str(d).lower().strip().lstrip(".")
        d = d.split("/", 1)[0]
        return bool(d) and (host == d or host.endswith("." + d))
    if blocked and any(match(d) for d in blocked):
        return False
    return not allowed or any(match(d) for d in allowed)


def _get(url: str, headers=None, data=None, timeout=None, opener=None) -> tuple[bytes, str, str]:
    """-> (body, content type, final URL), at most fetch_max_bytes."""
    req = urllib.request.Request(url, data=data, headers={"User-Agent": USER_AGENT, **(headers or {})})
    with (opener or urllib.request.build_opener()).open(req, timeout=timeout or CONFIG.get("timeout_s", 15)) as r:
        body = r.read(int(CONFIG.get("fetch_max_bytes", DEFAULTS["fetch_max_bytes"])) + 1)
        return body, r.headers.get("Content-Type", ""), r.geturl()


def _json(url, headers=None, data=None) -> dict:
    body, _, _ = _get(url, headers, data)
    return json.loads(body.decode("utf-8", "replace"))


# ------------------------------------------------------------------------------------------------ search backends
def _searxng(query: str, n: int) -> list[dict]:
    base = str(CONFIG["searxng_url"]).rstrip("/")
    d = _json(f"{base}/search?" + urllib.parse.urlencode({"q": query, "format": "json"}))
    return [{"url": r.get("url"), "title": r.get("title"), "snippet": r.get("content"),
             "page_age": r.get("publishedDate")} for r in d.get("results") or []][:n]


def _brave(query: str, n: int) -> list[dict]:
    d = _json("https://api.search.brave.com/res/v1/web/search?" + urllib.parse.urlencode({"q": query, "count": n}),
              {"Accept": "application/json", "X-Subscription-Token": str(CONFIG["brave_api_key"])})
    return [{"url": r.get("url"), "title": r.get("title"), "snippet": r.get("description"),
             "page_age": r.get("page_age") or r.get("age")} for r in (d.get("web") or {}).get("results") or []][:n]


def _tavily(query: str, n: int) -> list[dict]:
    key = str(CONFIG["tavily_api_key"])
    d = _json("https://api.tavily.com/search", {"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
              json.dumps({"query": query, "max_results": n}).encode())
    return [{"url": r.get("url"), "title": r.get("title"), "snippet": r.get("content"),
             "page_age": r.get("published_date")} for r in d.get("results") or []][:n]


def _mcp(query: str, n: int, cancel=None) -> list[dict]:
    if _HUB is None:
        raise WebError("unavailable", "no MCP servers are configured")
    _HUB.routes()
    r = _HUB.call(str(CONFIG["mcp_tool"]), {"query": query}, cancel)
    if not r.get("ok"):
        raise WebError("unavailable", r.get("text") or "the MCP search failed")
    text = r.get("text") or ""
    try:                                          # a tool that answers with a JSON list of results
        data = json.loads(text)
        rows = data.get("results") if isinstance(data, dict) else data
        if isinstance(rows, list) and all(isinstance(x, dict) for x in rows):
            return [{"url": x.get("url") or x.get("link"), "title": x.get("title") or x.get("name"),
                     "snippet": x.get("snippet") or x.get("content") or x.get("description"),
                     "page_age": x.get("page_age") or x.get("date")} for x in rows][:n]
    except ValueError:
        pass
    return [{"url": "", "title": f"{CONFIG['mcp_tool']} results", "snippet": text, "page_age": None}]


SEARCH = {"searxng": _searxng, "brave": _brave, "tavily": _tavily}


def search(query, allowed=None, blocked=None, cancel=None) -> list[dict]:
    """The backend's results for `query`, filtered by domain and trimmed: [{url, title, snippet, page_age}]."""
    if not searching():
        raise WebError("unavailable", "web search is not configured on this server")
    if not isinstance(query, str) or not query.strip():
        raise WebError("invalid_input", "the query is empty")
    if len(query) > MAX_QUERY:
        raise WebError("query_too_long")
    n = int(CONFIG["max_results"])
    backend = CONFIG["search"]
    try:
        rows = _mcp(query, n, cancel) if backend == "mcp" else SEARCH[backend](query, n * 2 if allowed or blocked
                                                                               else n)
    except WebError:
        raise
    except urllib.error.HTTPError as e:
        raise WebError("too_many_requests" if e.code == 429 else "unavailable", f"{backend}: HTTP {e.code}") from None
    except (urllib.error.URLError, OSError, ValueError) as e:
        raise WebError("unavailable", f"{backend}: {e}") from None
    cap = int(CONFIG["snippet_chars"])
    out = []
    for r in rows:
        url = r.get("url") or ""
        if url and (allowed or blocked) and not _domain_ok(url, allowed, blocked):
            continue
        snippet = re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", "", str(r.get("snippet") or "")))).strip()
        if len(snippet) > cap:
            snippet = snippet[:cap].rstrip() + " ..."
        out.append({"url": url, "title": str(r.get("title") or url)[:300], "snippet": snippet,
                    "page_age": str(r["page_age"]) if r.get("page_age") else None})
        if len(out) >= n:
            break
    return out


def results_text(results: list[dict]) -> str:
    """What the model reads for a search (rebuilt the same from the result blocks on replay)."""
    if not results:
        return "Web search: no results."
    lines = ["Web search results:"]
    for i, r in enumerate(results, 1):
        lines.append("")
        lines.append(f"[{i}] {r['title']}")
        if r.get("url"):
            lines.append(f"URL: {r['url']}")
        if r.get("page_age"):
            lines.append(f"Date: {r['page_age']}")
        if r.get("snippet"):
            lines.append(r["snippet"])
    return "\n".join(lines)


def _enc(snippet: str) -> str:
    return ENC_PREFIX + base64.b64encode(json.dumps({"s": snippet}, ensure_ascii=False).encode()).decode("ascii")


def _dec(value) -> str:
    if isinstance(value, str) and value.startswith(ENC_PREFIX):
        try:
            return json.loads(base64.b64decode(value[len(ENC_PREFIX):])).get("s") or ""
        except (ValueError, TypeError):
            return ""
    return ""


def search_blocks(results: list[dict]) -> list[dict]:
    """Anthropic's web_search_result blocks; encrypted_content carries the snippet the model read (this server's own
    encoding, as for reasoning), so a replayed turn is rendered as it was read."""
    out = []
    for r in results:
        b = {"type": "web_search_result", "url": r["url"], "title": r["title"], "encrypted_content": _enc(r["snippet"])}
        if r.get("page_age"):
            b["page_age"] = r["page_age"]
        out.append(b)
    return out


def results_of_blocks(blocks) -> list[dict]:
    return [{"url": b.get("url") or "", "title": b.get("title") or "", "snippet": _dec(b.get("encrypted_content")),
             "page_age": b.get("page_age")} for b in blocks if isinstance(b, dict)]


# ------------------------------------------------------------------------------------------------ fetch
class _Text(html.parser.HTMLParser):
    SKIP = {"script", "style", "noscript", "svg", "template", "iframe", "head"}
    BLOCK = {"p", "div", "br", "li", "ul", "ol", "tr", "table", "section", "article", "header", "footer", "h1", "h2",
             "h3", "h4", "h5", "h6", "pre", "blockquote", "hr", "main", "nav", "dd", "dt"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.out, self.skip, self.title, self.in_title = [], 0, "", False

    def handle_starttag(self, tag, attrs):
        if tag == "title":
            self.in_title = True
        if tag in self.SKIP:
            self.skip += 1
        elif tag in self.BLOCK:
            self.out.append("\n")

    def handle_endtag(self, tag):
        if tag == "title":
            self.in_title = False
        if tag in self.SKIP:
            self.skip = max(0, self.skip - 1)
        elif tag in self.BLOCK:
            self.out.append("\n")

    def handle_data(self, data):
        if self.in_title:
            self.title += data
        if not self.skip:
            self.out.append(data)


def html_text(page: str) -> tuple[str, str]:
    """(title, text) of an HTML page: scripts and styles dropped, blocks on their own lines, spaces collapsed."""
    p = _Text()
    try:
        p.feed(page)
        p.close()
    except Exception:  # noqa: BLE001 - a broken page still gives what was read
        pass
    text = "".join(p.out)
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n\s*\n\s*(\n\s*)+", "\n\n", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    return re.sub(r"\s+", " ", p.title).strip(), text.strip()


class _Redirects(urllib.request.HTTPRedirectHandler):
    max_redirections = 3

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if _CHECK is not None:
            _CHECK(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def fetch(url, allowed=None, blocked=None, seen: str = "") -> dict:
    """A page as text: {url, title, text} (text trimmed to fetch_max_chars, with a note).  WebError otherwise."""
    if not fetching():
        raise WebError("unavailable", "web fetch is not configured on this server")
    if not isinstance(url, str) or not url.strip():
        raise WebError("invalid_input", "no URL")
    url = url.strip()
    if len(url) > MAX_URL:
        raise WebError("url_too_long")
    if not url.startswith(("http://", "https://")):
        raise WebError("invalid_input", "only http(s) pages are fetched")
    if url not in seen and url.rstrip("/") not in seen:
        raise WebError("url_not_allowed", "only a URL that already appears in the conversation can be fetched")
    if not _domain_ok(url, allowed, blocked):
        raise WebError("url_not_allowed", "the domain is not allowed for this request")
    try:
        if _CHECK is not None:
            _CHECK(url)
    except ValueError as e:
        raise WebError("url_not_allowed", str(e)) from None
    # 0.94: each connection (the first and every redirect's) also checks the address it reached, so a name that
    # answers a public address to _CHECK and 127.0.0.1 to the connection (DNS rebinding) is refused
    handlers = netguard.handlers(_PEER) if _PEER is not None else []
    try:
        body, ctype, final = _get(url, {"Accept": "text/html,text/plain,application/json;q=0.9,*/*;q=0.5"},
                                  opener=urllib.request.build_opener(_Redirects, *handlers))
    except ValueError as e:                          # a redirect into this PC's network (_CHECK), or a peer (_PEER)
        raise WebError("url_not_allowed", str(e)) from None
    except urllib.error.HTTPError as e:
        raise WebError("too_many_requests" if e.code == 429 else "url_not_accessible", f"HTTP {e.code}") from None
    except (urllib.error.URLError, OSError) as e:
        raise WebError("url_not_accessible", str(e)) from None
    ctype = ctype.split(";")[0].strip().lower()
    charset = "utf-8"
    m = re.search(r"charset=([\w-]+)", ctype)
    if m:
        charset = m.group(1)
    over = len(body) > int(CONFIG["fetch_max_bytes"])
    body = body[:int(CONFIG["fetch_max_bytes"])]
    if ctype in ("text/html", "application/xhtml+xml") or (not ctype and body.lstrip()[:1] == b"<"):
        title, text = html_text(body.decode(charset, "replace"))
    elif ctype.startswith("text/") or ctype in ("application/json", "application/xml", "application/x-yaml", ""):
        title, text = "", body.decode(charset, "replace")
    elif ctype == "application/pdf":
        # 0.94: read like any other document - in serve/documents.py's reader process, with its time and memory
        # limits - never parsed in the server (a hostile PDF could hang or crash it there)
        name = urllib.parse.urlsplit(final or url).path.rsplit("/", 1)[-1][:100] or "page.pdf"
        doc = documents.extract(body, name, "application/pdf")
        if doc.error:
            raise WebError("unsupported_content_type", f"the PDF could not be read: {doc.error}")
        title, text = "", doc.text + (f"\n\n[{doc.note}]" if doc.note else "")
    else:
        raise WebError("unsupported_content_type", ctype)
    cap = int(CONFIG["fetch_max_chars"])
    if len(text) > cap or over:
        text = text[:cap].rstrip() + f"\n\n[cut at {cap:,} characters]"
    return {"url": final or url, "title": title[:300], "text": text}


def page_text(url: str, title: str, text: str) -> str:
    return f"Page {url}" + (f" ({title})" if title else "") + ":\n\n" + text


# ------------------------------------------------------------------------------------------------ server tools
def _entry_list(entry, key):
    v = (entry or {}).get(key)
    return [str(x) for x in v] if isinstance(v, list) else None


def _max_uses(entry) -> int:
    v = (entry or {}).get("max_uses")
    return int(v) if isinstance(v, int) and not isinstance(v, bool) and v > 0 else int(CONFIG.get("max_uses", 5))


def _search_definition(entry, ctx):
    return {"name": entry["name"], "description": "Search the web. Returns the top results with their title, address "
            "and a short extract; cite the addresses you use.", "parameters": {
                "type": "object", "properties": {"query": {"type": "string", "description": "the search query"}},
                "required": ["query"]}}


def _fetch_definition(entry, ctx):
    return {"name": entry["name"], "description": "Fetch a web page or document as text. Only an address that already "
            "appears in the conversation (from the user or a search result) can be fetched.", "parameters": {
                "type": "object", "properties": {"url": {"type": "string", "description": "the page's address"}},
                "required": ["url"]}}


def _seen(ctx) -> str:
    return getattr(ctx, "seen", "") or ""


def _note_seen(ctx, text: str) -> None:
    try:
        ctx.seen = _seen(ctx) + "\n" + text
    except AttributeError:
        pass


def _run_search(arguments, ctx, cancel=None, entry=None, use=1):
    if use > _max_uses(entry):
        return "error: max uses exceeded", {"type": "web_search_tool_result_error", "error_code": "max_uses_exceeded"}
    q = arguments.get("query")
    try:
        results = search(q, _entry_list(entry, "allowed_domains"), _entry_list(entry, "blocked_domains"), cancel)
    except WebError as e:
        print(f"[strata] web search failed: {e}", flush=True)
        return f"error: {e.code.replace('_', ' ')}", {"type": "web_search_tool_result_error", "error_code": e.code}
    print(f"[strata] web search ({CONFIG['search']}): {len(results)} results", flush=True)
    text = results_text(results)
    _note_seen(ctx, text)
    return text, search_blocks(results)


def _replay_search(block, ctx):
    c = block.get("content")
    if isinstance(c, dict):
        return f"error: {str(c.get('error_code') or 'unavailable').replace('_', ' ')}"
    return results_text(results_of_blocks(c if isinstance(c, list) else []))


def _run_fetch(arguments, ctx, cancel=None, entry=None, use=1):
    if use > _max_uses(entry):
        return "error: max uses exceeded", {"type": "web_fetch_tool_error", "error_code": "max_uses_exceeded"}
    try:
        page = fetch(arguments.get("url"), _entry_list(entry, "allowed_domains"), _entry_list(entry, "blocked_domains"),
                     _seen(ctx))
    except WebError as e:
        print(f"[strata] web fetch refused or failed: {e}", flush=True)
        return f"error: {e.code.replace('_', ' ')}", {"type": "web_fetch_tool_error", "error_code": e.code}
    print(f"[strata] web fetch: {len(page['text']):,} characters", flush=True)
    text = page_text(page["url"], page["title"], page["text"])
    _note_seen(ctx, text)
    doc = {"type": "document", "source": {"type": "text", "media_type": "text/plain", "data": page["text"]}}
    if page["title"]:
        doc["title"] = page["title"]
    return text, {"type": "web_fetch_result", "url": page["url"], "content": doc,
                  "retrieved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


def _replay_fetch(block, ctx):
    c = block.get("content") if isinstance(block.get("content"), dict) else {}
    if c.get("type") != "web_fetch_result":
        return f"error: {str(c.get('error_code') or 'unavailable').replace('_', ' ')}"
    doc = c.get("content") if isinstance(c.get("content"), dict) else {}
    src = doc.get("source") if isinstance(doc.get("source"), dict) else {}
    return page_text(c.get("url") or "", doc.get("title") or "", src.get("data") or "")


ST.register(ST.ServerKind(name="web_search", prefixes=("web_search_",), result_type="web_search_tool_result",
                          definition=_search_definition, run=_run_search, replay=_replay_search, available=searching,
                          usage_key="web_search_requests"))
ST.register(ST.ServerKind(name="web_fetch", prefixes=("web_fetch_",), result_type="web_fetch_tool_result",
                          definition=_fetch_definition, run=_run_fetch, replay=_replay_fetch, available=fetching,
                          usage_key="web_fetch_requests"))


# ------------------------------------------------------------------------------------------------ Responses (Codex)
_CODEX_TEXT: "collections.OrderedDict[str, str]" = collections.OrderedDict()   # web_search_call id -> what was read
_CODEX_LOCK = threading.Lock()
_CODEX_MAX = 256


def remember_codex(item_id: str, text: str) -> None:
    """A Responses web_search_call carries no results, so the text the model read is kept here (bounded) for the turn
    Codex sends back: the replayed prompt is then the one the server read."""
    with _CODEX_LOCK:
        _CODEX_TEXT[item_id] = text
        _CODEX_TEXT.move_to_end(item_id)
        while len(_CODEX_TEXT) > _CODEX_MAX:
            _CODEX_TEXT.popitem(last=False)


def codex_text(item_id) -> str:
    with _CODEX_LOCK:
        return _CODEX_TEXT.get(item_id) or "Web search: the results are no longer available."


def codex_context(entry: dict) -> ST.Context:
    """A Context with the Responses request's hosted web_search as the server tool `web_search`."""
    kind = next(k for k in ST.KINDS if k.name == "web_search")
    filters = (entry or {}).get("filters") if isinstance((entry or {}).get("filters"), dict) else {}
    own = {"name": "web_search"}
    if isinstance(filters.get("allowed_domains"), list):
        own["allowed_domains"] = filters["allowed_domains"]
    return ST.Context(server={"web_search": (kind, own)})
