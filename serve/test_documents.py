"""serve/test_documents.py - documents in the prompt (serve/documents.py): PDF, Word, Excel, PowerPoint, HTML, CSV and
text become text between a header and a footer, in the web chat (POST /v1/extract) and in the APIs (Anthropic
`document`, OpenAI `file`, Responses `input_file`).  Fixtures are made here: a one-page PDF written by hand, Office files
made with the libraries themselves (those tests are skipped where the library is not installed).

    python -m unittest serve.test_documents -v
"""
from __future__ import annotations

import base64
import contextlib
import http.client
import importlib.util
import io
import json
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve import documents as D  # noqa: E402
from serve.frontend import ChatTemplate, anthropic_to_messages, openai_to_messages  # noqa: E402
from serve.responses import ResponsesError, input_messages  # noqa: E402
from serve.server import ByteTokenizer, MockEngine, Service, serve  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
HAVE = {m: importlib.util.find_spec(m) is not None for m in ("pypdfium2", "docx", "openpyxl", "pptx")}


def make_pdf(pages: list[str]) -> bytes:
    """A valid PDF with one line of Helvetica text per page ("" = a page with no text), xref offsets computed."""
    objs: list = ["<< /Type /Catalog /Pages 2 0 R >>", None, "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    kids = []
    for t in pages:
        stream = f"BT /F1 12 Tf 72 720 Td ({t}) Tj ET".encode() if t else b""
        objs.append(f"<< /Length {len(stream)} >>\nstream\n".encode() + stream + b"\nendstream")
        objs.append(f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 3 0 R >> >> "
                    f"/Contents {len(objs)} 0 R >>")
        kids.append(len(objs))
    objs[1] = f"<< /Type /Pages /Kids [{' '.join(f'{k} 0 R' for k in kids)}] /Count {len(kids)} >>"
    out, offs = bytearray(b"%PDF-1.4\n"), []
    for i, o in enumerate(objs, 1):
        offs.append(len(out))
        out += f"{i} 0 obj\n".encode() + (o if isinstance(o, bytes) else o.encode()) + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    out += b"".join(f"{o:010d} 00000 n \n".encode() for o in offs)
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return bytes(out)


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


class Quiet(unittest.TestCase):
    """Each test starts with an empty cache and without the server-window lines."""

    def setUp(self):
        D._cache.clear()
        p = mock.patch("builtins.print")
        p.start()
        self.addCleanup(p.stop)


class TextKinds(Quiet):
    def test_sniff(self):
        self.assertEqual(D.sniff(b"%PDF-1.7 ..."), "pdf")
        self.assertEqual(D.sniff(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1xx", "a.doc"), "legacy")
        self.assertEqual(D.sniff(b"a,b\n1,2", "x.CSV"), "csv")
        self.assertEqual(D.sniff(b"a\tb", "", "text/tab-separated-values"), "tsv")
        self.assertEqual(D.sniff(b"<p>x</p>", "page.htm"), "html")
        self.assertEqual(D.sniff(b"# notes", "n.md"), "text")
        self.assertEqual(D.sniff(b"\x00\x01\x02binary", "x.bin"), "binary")
        self.assertEqual(D.sniff(b"not a pdf", "fake.pdf"), "binary")      # the name says PDF, the bytes do not
        self.assertEqual(D.sniff(b"PK\x03\x04garbage", "x.zip"), "binary")

    def test_decode_text(self):
        self.assertEqual(D.decode_text("café".encode()), "café")
        self.assertEqual(D.decode_text(b"\xef\xbb\xbfhi"), "hi")
        self.assertEqual(D.decode_text("hi".encode("utf-16")), "hi")
        self.assertEqual(D.decode_text(b"caf\xe9"), "café")            # Windows-1252

    def test_html(self):
        doc = D.extract(b"<html><head><title>T</title><style>p{}</style><script>x()</script></head><body>"
                        b"<h2>Head</h2><p>One &amp; two</p><ul><li>item</li></ul><table><tr><td>a</td><td>b</td>"
                        b"</tr></table></body></html>", "page.html")
        self.assertEqual(doc.kind, "html")
        for want in ("Title: T", "## Head", "One & two", "- item", "a", "b"):
            self.assertIn(want, doc.text)
        for gone in ("p{}", "x()", "<"):
            self.assertNotIn(gone, doc.text)

    def test_csv_becomes_a_table(self):
        doc = D.extract(b"name,qty\nwidget,3\n\"a|b\",\"x\ny\"\n", "stock.csv")
        self.assertEqual(doc.text.splitlines()[:3], ["| name | qty |", "|---|---|", "| widget | 3 |"])
        self.assertIn("a\\|b", doc.text)

    def test_render_has_a_header_and_a_footer(self):
        text = D.extract(b"hello", "notes.txt").render()
        self.assertEqual(text, '<document name="notes.txt" type="text">\nhello\n</document>')
        self.assertIn('name="ab"', D.Doc('a"<b>\n', "text", "x").render())     # the name cannot break the header

    def test_unreadable_files_are_one_line_saying_why(self):
        self.assertIn("Word 97-2003", D.extract(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1rest", "old.doc").render())
        self.assertIn("not a PDF, Word", D.extract(b"\x00\x01\x02", "x.bin").render())
        with mock.patch.object(D, "MAX_BYTES", 10):
            self.assertIn("up to", D.extract(b"x" * 11, "big.txt").render())
        self.assertIn("no text was found", D.extract(b"  \n", "empty.txt").render())

    def test_long_documents_are_cut_with_a_note(self):
        with mock.patch.object(D, "MAX_CHARS", 100):
            doc = D.extract(b"x" * 250, "long.txt")
        self.assertEqual(len(doc.text), 100)
        self.assertIn("cut at 100 of 250 characters", doc.render())

    def test_cache_reads_a_document_once(self):
        with mock.patch.object(D, "_run_child", return_value={"text": "page text", "pages": 1}) as child:
            pdf = b"%PDF-1.4 whatever"
            a = D.extract(pdf, "a.pdf")
            b = D.extract(pdf, "a.pdf")
        self.assertIs(a, b)
        self.assertEqual(child.call_count, 1)

    def test_a_timeout_is_a_note_and_not_cached(self):
        with mock.patch.object(D, "_run_child", return_value={"error": "reading it took longer than 120 s"}) as child:
            D.extract(b"%PDF-1.4 slow", "slow.pdf")
            doc = D.extract(b"%PDF-1.4 slow", "slow.pdf")
        self.assertEqual(child.call_count, 2)
        self.assertIn("longer than", doc.render())

    def test_child_reports_a_missing_package(self):
        out = io.BytesIO()

        def missing(_data):
            raise ImportError("No module named 'pypdfium2'")
        with mock.patch.dict(D.READERS, {"pdf": missing}), \
                mock.patch.object(sys, "stdin", mock.Mock(buffer=io.BytesIO(b"%PDF"))), \
                mock.patch.object(sys, "stdout", mock.Mock(buffer=out)):
            D._child_main("pdf")
        self.assertIn("pypdfium2 package is not installed", json.loads(out.getvalue())["error"])

    def test_estimate(self):
        self.assertAlmostEqual(D.estimate_seconds(6500), 10.0)
        self.assertAlmostEqual(D.estimate_seconds(1000, 500), 2.0)


@unittest.skipUnless(HAVE["pypdfium2"], "pypdfium2 is not installed")
class Pdf(Quiet):
    def test_pages_and_a_page_without_text(self):
        doc = D.extract(make_pdf(["Hello PDF world", ""]), "report.pdf")
        self.assertEqual((doc.kind, doc.pages, doc.error), ("pdf", 2, ""))
        text = doc.render()
        self.assertTrue(text.startswith('<document name="report.pdf" type="PDF" pages="2">\n--- Page 1 ---\n'
                                        'Hello PDF world'), text)
        self.assertIn("--- Page 2 ---\n(no text on this page)", text)
        self.assertIn("1 of 2 pages have no text layer", text)
        self.assertTrue(text.endswith("</document>"))

    def test_a_scan_says_so(self):
        self.assertIn("no text layer (a scan", D.extract(make_pdf([""]), "scan.pdf").render())

    def test_a_broken_pdf_is_a_note(self):
        text = D.extract(b"%PDF-1.4\nthis is not really a pdf", "bad.pdf").render()
        self.assertTrue(text.startswith('[Document "bad.pdf" could not be read: '), text)


class Office(Quiet):
    @unittest.skipUnless(HAVE["docx"], "python-docx is not installed")
    def test_word(self):
        import docx
        d = docx.Document()
        d.add_heading("Plan", 1)
        d.add_paragraph("First paragraph.")
        t = d.add_table(rows=2, cols=2)
        t.cell(0, 0).text, t.cell(0, 1).text, t.cell(1, 0).text, t.cell(1, 1).text = "k", "v", "a", "1"
        d.add_paragraph("After the table.")
        buf = io.BytesIO()
        d.save(buf)
        doc = D.extract(buf.getvalue(), "plan.docx")
        self.assertEqual(doc.kind, "docx")
        self.assertEqual(doc.text, "# Plan\n\nFirst paragraph.\n\n| k | v |\n|---|---|\n| a | 1 |\n\nAfter the table.")

    @unittest.skipUnless(HAVE["openpyxl"], "openpyxl is not installed")
    def test_excel(self):
        import openpyxl
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "Sales"
        ws.append(["month", "total"])
        ws.append(["Jan", 12.5])
        wb.create_sheet("Empty")
        buf = io.BytesIO()
        wb.save(buf)
        doc = D.extract(buf.getvalue(), "sales.xlsx")
        self.assertEqual((doc.kind, doc.pages), ("xlsx", 2))
        self.assertIn("## Sheet: Sales\n\n| month | total |\n|---|---|\n| Jan | 12.5 |", doc.text)
        self.assertIn("## Sheet: Empty\n\n(empty)", doc.text)
        self.assertIn('sheets="2"', doc.render())

    @unittest.skipUnless(HAVE["pptx"], "python-pptx is not installed")
    def test_powerpoint(self):
        import pptx
        p = pptx.Presentation()
        s = p.slides.add_slide(p.slide_layouts[1])
        s.shapes.title.text = "Roadmap"
        s.placeholders[1].text = "Ship 0.94"
        s.notes_slide.notes_text_frame.text = "say it slowly"
        buf = io.BytesIO()
        p.save(buf)
        doc = D.extract(buf.getvalue(), "deck.pptx")
        self.assertEqual((doc.kind, doc.pages), ("pptx", 1))
        self.assertEqual(doc.text, "## Slide 1\n\nRoadmap\n\nShip 0.94\n\nNotes: say it slowly")


class ApiParts(Quiet):
    """The three APIs' document parts become text parts in place; a URL or a file id is a note, never fetched."""
    TEXT = b"The secret word is pelican."

    def test_anthropic_document_blocks(self):
        for source in ({"type": "base64", "media_type": "text/plain", "data": b64(self.TEXT)},
                       {"type": "text", "media_type": "text/plain", "data": self.TEXT.decode()},
                       {"type": "content", "content": [{"type": "text", "text": self.TEXT.decode()}]}):
            with self.subTest(source=source["type"]):
                messages, _, _ = anthropic_to_messages({"messages": [{"role": "user", "content": [
                    {"type": "document", "source": source, "title": "memo.txt", "context": "from HR"},
                    {"type": "text", "text": "What is the word?"}]}]})
                self.assertEqual(messages[-1]["content"],
                                 '<document name="memo.txt" type="text">\nContext: from HR\n'
                                 'The secret word is pelican.\n</document>\n\nWhat is the word?')

    @unittest.skipUnless(HAVE["pypdfium2"], "pypdfium2 is not installed")
    def test_anthropic_pdf_inside_a_tool_result(self):
        # Claude Code's Read of a PDF: a document block in a tool_result
        messages, _, _ = anthropic_to_messages({"messages": [
            {"role": "user", "content": "read it"},
            {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "Read", "input": {}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": [
                {"type": "document", "source": {"type": "base64", "media_type": "application/pdf",
                                                "data": b64(make_pdf(["Quarterly numbers"]))}}]}]}]})
        tool = next(m for m in messages if m["role"] == "tool")
        self.assertIn('type="PDF" pages="1"', tool["content"])
        self.assertIn("Quarterly numbers", tool["content"])

    def test_anthropic_url_and_file_sources_are_notes(self):
        for source, why in (({"type": "url", "url": "http://10.0.0.1/x.pdf"}, "does not fetch URLs"),
                            ({"type": "file", "file_id": "file_1"}, "keeps no uploaded files")):
            with mock.patch("urllib.request.urlopen") as fetch:
                messages, _, _ = anthropic_to_messages({"messages": [{"role": "user", "content": [
                    {"type": "document", "source": source}, {"type": "text", "text": "q"}]}]})
            fetch.assert_not_called()
            self.assertIn(why, messages[-1]["content"])

    def test_openai_file_part(self):
        for data in ("data:text/plain;base64," + b64(self.TEXT), b64(self.TEXT)):
            messages, _, _ = openai_to_messages({"messages": [{"role": "user", "content": [
                {"type": "text", "text": "Summarize:"},
                {"type": "file", "file": {"filename": "memo.txt", "file_data": data}}]}]})
            self.assertEqual(messages[-1]["content"],
                             'Summarize:\n\n<document name="memo.txt" type="text">\nThe secret word is pelican.\n'
                             '</document>')
        messages, _, _ = openai_to_messages({"messages": [{"role": "user", "content": [
            {"type": "file", "file": {"file_id": "file-abc"}}]}]})
        self.assertIn("keeps no uploaded files", messages[-1]["content"])

    def test_responses_input_file(self):
        msgs = input_messages({"input": [{"role": "user", "content": [
            {"type": "input_file", "filename": "memo.txt", "file_data": "data:text/plain;base64," + b64(self.TEXT)},
            {"type": "input_text", "text": "word?"}]}]})
        self.assertIn("The secret word is pelican.\n</document>\n\nword?", msgs[-1]["content"])
        for part in ({"type": "input_file", "file_id": "f"}, {"type": "input_file", "file_url": "http://x/y.pdf"}):
            with self.assertRaises(ResponsesError):
                input_messages({"input": [{"role": "user", "content": [part]}]})

    def test_same_document_same_prompt(self):
        # a client re-sends the document with every turn: the text must be identical (the conversation cache)
        part = {"type": "file", "file": {"filename": "m.txt", "file_data": b64(self.TEXT)}}
        a = openai_to_messages({"messages": [{"role": "user", "content": [part]}]})
        b = openai_to_messages({"messages": [{"role": "user", "content": [part]}]})
        self.assertEqual(a, b)

    def test_bad_base64_is_a_note(self):
        messages, _, _ = openai_to_messages({"messages": [{"role": "user", "content": [
            {"type": "file", "file": {"filename": "x.pdf", "file_data": "data:application/pdf;base64,%%%"}}]}]})
        self.assertIn('[Document "x.pdf"', messages[-1]["content"])


class ExtractEndpoint(Quiet):
    """POST /v1/extract: what the web chat shows before sending (text, tokens, first-read time)."""

    def setUp(self):
        super().setUp()
        tok = ByteTokenizer()
        self.svc = Service(MockEngine(tok, "</think>\n\nok", max_context=4096), tok,
                           ChatTemplate(ROOT / "serve/chat_template.jinja"))
        self.httpd = serve(self.svc, port=0)
        self.addCleanup(self.httpd.server_close)
        self.addCleanup(self.httpd.shutdown)

    def post(self, body, ctype="application/json"):
        c = http.client.HTTPConnection("127.0.0.1", self.httpd.server_address[1], timeout=60)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                c.request("POST", "/v1/extract", body=json.dumps(body).encode(), headers={"Content-Type": ctype})
                r = c.getresponse()
                return r.status, json.loads(r.read())
        finally:
            c.close()

    def test_text_document(self):
        status, out = self.post({"name": "notes.md", "data": "data:text/markdown;base64," + b64(b"# Hi\n\nthere")})
        self.assertEqual(status, 200)
        self.assertEqual(out["text"], '<document name="notes.md" type="text">\n# Hi\n\nthere\n</document>')
        self.assertEqual(out["tokens"], len(out["text"].encode()))          # the byte tokenizer: one per byte
        self.assertEqual(out["read_seconds"], round(out["tokens"] / D.READ_TOK_S, 1))
        self.assertIsNone(out["error"])

    @unittest.skipUnless(HAVE["pypdfium2"], "pypdfium2 is not installed")
    def test_pdf(self):
        status, out = self.post({"name": "r.pdf", "data": b64(make_pdf(["Hello"]))})
        self.assertEqual((status, out["type"], out["pages"]), (200, "PDF", 1))
        self.assertIn("Hello", out["text"])

    def test_unreadable_and_bad_requests(self):
        status, out = self.post({"name": "old.xls", "data": b64(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1x")})
        self.assertEqual(status, 200)
        self.assertIn("Excel 97-2003", out["error"])
        self.assertEqual(self.post({"name": "x"})[0], 400)
        self.assertEqual(self.post({"name": "x", "data": "%%%"})[0], 400)


if __name__ == "__main__":
    unittest.main()
