"""serve/netguard.py - urllib handlers that check the address a connection really reached (0.94, DNS rebinding).

server.public_url checks every address a URL's host name resolves to before a picture or a web page is fetched.  But
http.client resolves the name again when it connects, and a name with a zero TTL can answer a public address to the
check and 127.0.0.1 (or the router, or 169.254.169.254) to the connection.  The handlers here give every connection a
`_create_connection` that passes the connected socket's peer address to `check(ip)` (a ValueError refuses it) and closes
the socket before a byte is sent - neither the TLS hello nor the request.  Each redirect hop opens its connection through
the same handler, so redirects are checked the same way.  Through a proxy the proxy resolves the name and the peer is the
proxy, so a proxied request is not checked here (as before).

Self-contained (no serve.server import): server.py and websearch.py both build their openers with `handlers(check)`.
"""
from __future__ import annotations

import http.client
import urllib.request


def _checked(cls, check):
    """A connection class factory for urllib's do_open: `cls`, whose socket is refused unless check(peer ip) passes."""
    def make(host, **kw):
        conn = cls(host, **kw)
        create = conn._create_connection

        def connect(address, *a, **k):
            sock = create(address, *a, **k)
            try:
                check(sock.getpeername()[0])
            except BaseException:
                sock.close()
                raise
            return sock
        conn._create_connection = connect
        return conn
    return make


def _proxied(req) -> bool:
    return bool(getattr(req, "_tunnel_host", None)) or req.has_proxy()


class PeerCheckedHTTP(urllib.request.HTTPHandler):
    def __init__(self, check):
        super().__init__()
        self._check = check

    def http_open(self, req):
        cls = http.client.HTTPConnection
        return self.do_open(cls if _proxied(req) else _checked(cls, self._check), req)


class PeerCheckedHTTPS(urllib.request.HTTPSHandler):
    def __init__(self, check):
        super().__init__()
        self._check = check

    def https_open(self, req):
        cls = http.client.HTTPSConnection
        return self.do_open(cls if _proxied(req) else _checked(cls, self._check), req, context=self._context)


def handlers(check) -> list:
    """The http and https handlers for urllib.request.build_opener; `check(ip)` raises ValueError to refuse a peer."""
    return [PeerCheckedHTTP(check), PeerCheckedHTTPS(check)]
