"""0.94 (SSRF): an image URL is fetched only when every address its host resolves to is a global one, and a redirect
is checked like the first URL.    python -m unittest serve.test_image_url_public"""
import http.server
import os
import socket
import threading
import unittest
import urllib.request
from unittest import mock

from serve import netguard, server


def resolver(*ips):
    def resolve(host, port, proto=0):
        return [(socket.AF_INET6 if ":" in ip else socket.AF_INET, socket.SOCK_STREAM, proto, "", (ip, port)) for ip in ips]
    return resolve


class PublicImageUrls(unittest.TestCase):
    def setUp(self):
        p = mock.patch.dict(os.environ, {}, clear=False)
        p.start()
        self.addCleanup(p.stop)
        os.environ.pop("STRATA_IMAGE_FETCH_PRIVATE", None)

    def test_inward_addresses_refused(self):
        for ip in ("127.0.0.1", "10.0.0.5", "192.168.1.1", "172.16.0.1", "169.254.169.254", "0.0.0.0", "::1",
                   "fe80::1", "fc00::1", "::ffff:127.0.0.1", "100.64.0.1"):
            with self.assertRaises(ValueError, msg=ip):
                server.public_url("http://pics.example/x.png", resolve=resolver(ip))

    def test_any_inward_address_refuses(self):
        with self.assertRaises(ValueError):
            server.public_url("https://pics.example/x.png", resolve=resolver("93.184.216.34", "10.0.0.1"))

    def test_public_allowed(self):
        server.public_url("https://pics.example/x.png", resolve=resolver("93.184.216.34"))
        server.public_url("http://pics.example:8080/x.png", resolve=resolver("2606:2800:220:1:248:1893:25c8:1946"))

    def test_bad_urls(self):
        for url in ("ftp://pics.example/x.png", "http:///x.png", "http://[::1/x.png"):
            with self.assertRaises(ValueError, msg=url):
                server.public_url(url, resolve=resolver("93.184.216.34"))

    def test_opt_out(self):
        with mock.patch.dict(os.environ, {"STRATA_IMAGE_FETCH_PRIVATE": "1"}):
            server.public_url("http://pics.example/x.png", resolve=resolver("127.0.0.1"))

    def test_redirect_checked(self):
        h = server._PublicRedirects()
        with mock.patch.object(server, "public_url", side_effect=ValueError("inward")) as pu:
            with self.assertRaises(ValueError):
                h.redirect_request(None, None, 302, "Found", {}, "http://127.0.0.1/admin")
            pu.assert_called_once_with("http://127.0.0.1/admin")

    def test_download_checks_first(self):
        with mock.patch.object(server, "public_url", side_effect=ValueError("inward")), \
                mock.patch.object(server._PUBLIC_OPENER, "open") as op:
            with self.assertRaises(ValueError):
                server.Vision.download("http://127.0.0.1/x.png")
            op.assert_not_called()


class _Site(http.server.BaseHTTPRequestHandler):
    hits: list = []

    def log_message(self, *a):
        pass

    def do_GET(self):
        _Site.hits.append(self.path)
        if self.path == "/hop":
            self.send_response(302)
            self.send_header("Location", "/x.png")
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Length", "3")
        self.end_headers()
        self.wfile.write(b"PNG")


class Rebinding(unittest.TestCase):
    """0.94 (DNS rebinding): the name passed public_url, then the connection reached 127.0.0.1 - refused before a byte
    is sent, on the first connection and on a redirect's."""

    @classmethod
    def setUpClass(cls):
        cls.site = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Site)
        threading.Thread(target=cls.site.serve_forever, daemon=True).start()
        cls.root = f"http://127.0.0.1:{cls.site.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.site.shutdown()
        cls.site.server_close()

    def setUp(self):
        _Site.hits = []
        p = mock.patch.dict(os.environ, {}, clear=False)
        p.start()
        self.addCleanup(p.stop)
        os.environ.pop("STRATA_IMAGE_FETCH_PRIVATE", None)

    def test_peer_checked(self):
        with self.assertRaises(ValueError):
            server.public_peer("127.0.0.1")
        with self.assertRaises(ValueError):
            server.public_peer("::ffff:192.168.1.1")
        server.public_peer("93.184.216.34")

    def test_connection_to_this_pc_refused_after_a_passing_check(self):
        with mock.patch.object(server, "public_url"):          # the name's first answer was public
            with self.assertRaises(ValueError) as e:
                server.Vision.download(self.root + "/x.png")
        self.assertIn("127.0.0.1", str(e.exception))
        self.assertEqual(_Site.hits, [])                        # nothing reached the local service

    def test_each_redirect_hop_is_checked(self):
        seen = []

        def first_only(ip):                                     # the first connection passes, the redirect's not
            seen.append(ip)
            if len(seen) > 1:
                raise ValueError("inward")
        opener = urllib.request.build_opener(*netguard.handlers(first_only))
        with self.assertRaises(ValueError):
            opener.open(self.root + "/hop", timeout=10)
        self.assertEqual((seen, _Site.hits), (["127.0.0.1", "127.0.0.1"], ["/hop"]))

    def test_opt_out_still_fetches(self):
        with mock.patch.dict(os.environ, {"STRATA_IMAGE_FETCH_PRIVATE": "1"}):
            self.assertEqual(server.Vision.download(self.root + "/hop"), b"PNG")
        self.assertEqual(_Site.hits, ["/hop", "/x.png"])

    def test_web_fetch_peer_checked(self):
        from serve import websearch as W
        W.configure({"fetch": True}, public_url=lambda url: None, public_peer=server.public_peer)
        self.addCleanup(W.configure, None)
        url = self.root + "/x.png"
        with self.assertRaises(W.WebError) as e:
            W.fetch(url, seen=url)
        self.assertEqual(e.exception.code, "url_not_allowed")
        self.assertEqual(_Site.hits, [])


if __name__ == "__main__":
    unittest.main()
