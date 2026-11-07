"""0.94 (SSRF): an image URL is fetched only when every address its host resolves to is a global one, and a redirect
is checked like the first URL.    python -m unittest serve.test_image_url_public"""
import os
import socket
import unittest
from unittest import mock

from serve import server


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


if __name__ == "__main__":
    unittest.main()
