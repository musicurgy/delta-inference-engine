"""0.94 (B1): Pillow opens a client's picture only as JPEG, PNG, GIF, BMP or WebP.  Pillow 12.3's JPEG 2000 decoder
(libopenjp2, GHSA-78g4-xqwv-q3g8) and its TIFF, MSP and Spider readers carry published advisories, and Image.open with
no `formats` tries every reader Pillow has: a picture in any other format is a 400 before a decoder sees its bytes.
DeepSeek's encoder (tools/v41/vision, a separate process) opens what the server passes on with the same list.

    python -m unittest serve.test_picture_formats
"""
import base64
import importlib.util
import io
import json
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest import mock

from serve.frontend import ChatTemplate
from serve.server import PICTURE_FORMATS, ByteTokenizer, MockEngine, Service, Vision, serve

try:
    from PIL import Image
except ImportError:                 # Pillow is optional: without it only JPEG, PNG, BMP and GIF are read at all
    Image = None

ROOT = Path(__file__).resolve().parents[1]
V41_VISION = ROOT / "tools" / "v41" / "vision"


def picture(fmt: str) -> bytes:
    out = io.BytesIO()
    Image.new("RGB", (16, 8), (200, 30, 30)).save(out, format=fmt)
    return out.getvalue()


class Readers:
    """Image.OPEN with every reader outside PICTURE_FORMATS replaced by one that records the call (and fails)."""

    def __init__(self):
        Image.init()
        self.reached = []

        def recorder(fmt):
            def factory(fp, filename=None):
                self.reached.append(fmt)
                raise SyntaxError(f"{fmt} reader reached")
            return factory

        self.patch = mock.patch.dict(Image.OPEN, {fmt: (recorder(fmt), accept) for fmt, (_, accept) in
                                                  Image.OPEN.items() if fmt not in PICTURE_FORMATS})

    def __enter__(self):
        self.patch.start()
        return self

    def __exit__(self, *exc):
        self.patch.stop()


@unittest.skipIf(Image is None, "Pillow is not installed")
class PictureFormats(unittest.TestCase):
    OTHER = ("JPEG2000", "TIFF", "SPIDER", "TGA", "ICO", "PPM")

    def test_other_formats_are_refused_before_any_reader(self):
        for fmt in self.OTHER:
            data = picture(fmt)
            with self.subTest(fmt=fmt), Readers() as readers:
                with self.assertRaisesRegex(ValueError, "JPEG, PNG, GIF, WebP or BMP"):
                    Vision.normalize(data)
                self.assertEqual(readers.reached, [])

    def test_the_recorder_would_see_an_unrestricted_open(self):
        # the check above means something: Image.open without `formats` reaches the JPEG 2000 reader
        with Readers() as readers, self.assertRaises(Exception):
            Image.open(io.BytesIO(picture("JPEG2000")))
        self.assertIn("JPEG2000", readers.reached)

    def test_a_jpeg_start_that_does_not_parse_reaches_no_other_reader(self):
        # JPEG's first bytes, then something else: before 0.94 Image.open went on through every other reader
        with Readers() as readers:
            data = b"\xff\xd8\xff" + picture("TGA")
            self.assertEqual(Vision.normalize(data), data)          # left as it was sent (the encoder says so)
            self.assertEqual(readers.reached, [])

    def test_the_picture_formats_still_work(self):
        out = Vision.normalize(picture("WEBP"))                     # converted for strata-vision's decoder
        self.assertEqual(out[:8], b"\x89PNG\r\n\x1a\n")
        self.assertEqual(Image.open(io.BytesIO(out)).size, (16, 8))
        for fmt in ("JPEG", "PNG", "GIF", "BMP"):                   # passed through as they are
            data = picture(fmt)
            self.assertIs(Vision.normalize(data), data)

    def test_a_jpeg2000_data_url_is_a_400(self):
        jp2 = base64.b64encode(picture("JPEG2000")).decode()
        tok = ByteTokenizer()
        with tempfile.TemporaryDirectory() as d:
            v = Vision.__new__(Vision)
            v.dir, v.lock, v.cache = Path(d), threading.Lock(), {}
            v.proc, v.stopped = mock.Mock(**{"poll.return_value": None}), False   # running; never asked
            svc = Service(MockEngine(tok, "</think>\n\nok", max_context=4096), tok,
                          ChatTemplate(ROOT / "serve/chat_template.jinja"), vision=v)
            httpd = serve(svc, port=0)
            base = f"http://127.0.0.1:{httpd.server_address[1]}"

            def post(path, image):
                body = {"model": "m", "max_tokens": 8, "messages": [{"role": "user", "content": [
                    {"type": "text", "text": "what is it?"}, image]}]}
                req = urllib.request.Request(base + path, data=json.dumps(body).encode(),
                                             headers={"Content-Type": "application/json"})
                try:
                    with urllib.request.urlopen(req, timeout=30) as r:
                        return r.status, json.loads(r.read())
                except urllib.error.HTTPError as e:
                    with e:
                        return e.code, json.loads(e.read())

            try:
                with Readers() as readers:
                    for path, image in (
                            ("/v1/chat/completions",
                             {"type": "image_url", "image_url": {"url": f"data:image/jp2;base64,{jp2}"}}),
                            ("/v1/messages",
                             {"type": "image", "source": {"type": "base64", "media_type": "image/jp2", "data": jp2}})):
                        with self.subTest(path=path):
                            status, body = post(path, image)
                            self.assertEqual(status, 400, body)
                            self.assertIn("JPEG, PNG, GIF, WebP or BMP", json.dumps(body))
                    self.assertEqual(readers.reached, [])
                v.proc.stdin.write.assert_not_called()
                self.assertEqual(list(Path(d).iterdir()), [])
            finally:
                httpd.shutdown()
                httpd.server_close()


@unittest.skipUnless(Image is not None and importlib.util.find_spec("torch") and importlib.util.find_spec("numpy"),
                     "needs Pillow and PyTorch")
class EncoderFormats(unittest.TestCase):
    """DeepSeek's load_image (its file unchanged) opens what the server passed on with the same list."""

    def test_reference_open_is_restricted(self):
        sys.path.insert(0, str(V41_VISION))
        import ds41_vision as DV
        self.assertEqual(DV.PICTURE_FORMATS, PICTURE_FORMATS)
        DV._load_reference(DV.DEFAULT_REFERENCE)
        DV._load_reference(DV.DEFAULT_REFERENCE)                    # a second load does not wrap it twice
        import image_processor as IP
        self.assertIsInstance(IP.Image, DV._PictureFormatsOnly)
        self.assertIs(IP.Image._image, Image)
        with Readers() as readers:
            for fmt in ("JPEG2000", "TIFF"):
                with self.subTest(fmt=fmt), self.assertRaises(Exception):
                    IP.load_image({"data": picture(fmt)}, None)
            data = b"BM" + picture("TGA")[2:]                       # BMP's first bytes, the rest not a BMP
            with self.assertRaises(Exception):
                IP.load_image({"data": data}, None)
            self.assertEqual(readers.reached, [])
        with IP.Image.open(io.BytesIO(picture("PNG"))) as im:       # the picture formats still open
            self.assertEqual(im.size, (16, 8))


if __name__ == "__main__":
    unittest.main()
