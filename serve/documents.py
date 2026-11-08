"""Documents in the prompt: PDF, Word, Excel, PowerPoint, HTML, CSV and plain text become text the model reads.

The model reads text and pictures only, so a document is turned into text on the CPU before the turn and put into the
prompt between a clear header and footer:

    <document name="report.pdf" type="PDF" pages="12">
    --- Page 1 ---
    ...
    </document>

Where documents come from:
  * the web chat's Attach button (POST /v1/extract: the text, its token count and the first-read time are shown before
    sending);
  * Anthropic `document` blocks (base64 PDF, plain text, or a list of text blocks), also inside a tool_result
    (Claude Code's Read of a PDF);
  * OpenAI chat `file` parts and Responses `input_file` parts (file_data as a data: URL or base64).
URLs and file ids are not fetched (this server reads no URLs and keeps no files): such a part becomes a one-line note.

PDF, Word, Excel and PowerPoint are read in a separate Python process with a time limit, so a broken or hostile file
cannot hang or crash the server: pypdfium2 (Apache-2.0 / BSD-3) for PDF, python-docx, openpyxl and python-pptx (MIT)
for Office files.  Text, Markdown, CSV and HTML are read here with the standard library.  The result is cached by the
file's SHA-256, so a client that sends the same PDF with every turn gets the same text (the conversation cache keeps
working) and it is read once.

Run as a script (`python documents.py <kind>`), it reads one file's bytes from stdin and prints the extraction as JSON:
that is the child process.  It imports nothing from the server.
"""
from __future__ import annotations

import base64
import binascii
import csv
import hashlib
import html.parser
import io
import json
import os
import re
import subprocess
import sys
import threading
import time
import zipfile
from collections import OrderedDict

MAX_BYTES = 64 << 20            # the largest document read (bytes); larger ones become a note
MAX_CHARS = 2_000_000           # the most text one document puts into the prompt (~500K tokens); the rest is cut
CHILD_TIMEOUT_S = 120.0         # a PDF / Office file that takes longer is given up (a note says so)
# 0.94 (decompression bombs): MAX_BYTES bounds only the packed file.  An Office file (a zip) whose parts unpack to more
# than MAX_UNZIPPED, or that has a part over RATIO_FROM unpacking to more than MAX_RATIO times its packed size, is
# refused before the reader starts; the reader may not commit more than CHILD_MEMORY; and its answer is read up to
# 12 bytes a character of MAX_CHARS (ensure_ascii's escapes, a surrogate pair at worst) plus OUTPUT_SLACK
MAX_UNZIPPED = 256 << 20
MAX_RATIO = 200
RATIO_FROM = 4 << 20
CHILD_MEMORY = 2 << 30
OUTPUT_SLACK = 64 << 10
READ_TOK_S = 650.0              # first-read speed for the estimate (README: 600-740 tok/s on long prompts, RTX 3090)

KIND_NAMES = {"pdf": "PDF", "docx": "Word", "xlsx": "Excel", "pptx": "PowerPoint", "html": "HTML", "csv": "CSV",
              "tsv": "TSV", "text": "text"}
BINARY_KINDS = ("pdf", "docx", "xlsx", "pptx")
PACKAGES = {"pdf": "pypdfium2", "docx": "python-docx", "xlsx": "openpyxl", "pptx": "python-pptx"}
MEDIA_KINDS = {
    "application/pdf": "pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "xlsx",
    "application/vnd.ms-excel.sheet.macroenabled.12": "xlsx",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": "pptx",
    "text/html": "html", "application/xhtml+xml": "html",
    "text/csv": "csv", "text/tab-separated-values": "tsv",
}
EXT_KINDS = {".pdf": "pdf", ".docx": "docx", ".docm": "docx", ".xlsx": "xlsx", ".xlsm": "xlsx", ".pptx": "pptx",
             ".pptm": "pptx", ".html": "html", ".htm": "html", ".xhtml": "html", ".csv": "csv", ".tsv": "tsv"}
LEGACY = {".doc": "Word 97-2003 (.doc)", ".xls": "Excel 97-2003 (.xls)", ".ppt": "PowerPoint 97-2003 (.ppt)"}


class Doc:
    """One extracted document: its name, kind, text, page / sheet / slide count, and a note when it was cut or failed."""

    def __init__(self, name: str, kind: str, text: str = "", pages: int | None = None, note: str = "",
                 error: str = ""):
        self.name, self.kind, self.text, self.pages, self.note, self.error = name, kind, text, pages, note, error

    def render(self) -> str:
        """The prompt text: header, body, footer.  A document that could not be read is one line saying why."""
        name = _attr(self.name)
        if self.error:
            return f"[Document \"{name}\" could not be read: {self.error}]"
        attrs = f'name="{name}" type="{KIND_NAMES.get(self.kind, self.kind)}"'
        if self.pages is not None:
            unit = {"xlsx": "sheets", "pptx": "slides"}.get(self.kind, "pages")
            attrs += f' {unit}="{self.pages}"'
        body = self.text.strip("\n")
        note = f"\n[{self.note}]" if self.note else ""
        return f"<document {attrs}>\n{body}{note}\n</document>"


def _attr(s: str) -> str:
    return re.sub(r"[\"\r\n<>]", "", str(s or "document"))[:200]


# ------------------------------------------------------------------------------------------------ which kind
def sniff(data: bytes, name: str = "", media_type: str = "") -> str:
    """The document's kind from its first bytes, then its media type, then its file name.  'legacy' for a .doc / .xls /
    .ppt (an OLE file), 'binary' for anything else that is not text."""
    if data[:5] == b"%PDF-":
        return "pdf"
    if data[:4] == b"PK\x03\x04":
        try:
            names = set(zipfile.ZipFile(io.BytesIO(data)).namelist())
        except (zipfile.BadZipFile, ValueError):
            names = set()
        for part, kind in (("word/document.xml", "docx"), ("xl/workbook.xml", "xlsx"),
                           ("ppt/presentation.xml", "pptx")):
            if part in names:
                return kind
        return "binary"
    if data[:8] == b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1":
        return "legacy"
    mt = (media_type or "").split(";")[0].strip().lower()
    ext = os.path.splitext(name or "")[1].lower()
    kind = MEDIA_KINDS.get(mt) or EXT_KINDS.get(ext)
    if kind in BINARY_KINDS:              # said to be a PDF / Office file, but the bytes are not one
        return "binary"
    if b"\x00" in data[:8192] and not data.startswith((b"\xff\xfe", b"\xfe\xff")):
        return "binary"
    return kind or "text"


def decode_text(data: bytes) -> str:
    """Bytes of a text file -> str: a BOM wins, then UTF-8, then Windows-1252 (what an old Windows editor saved)."""
    for bom, enc in ((b"\xef\xbb\xbf", "utf-8"), (b"\xff\xfe", "utf-16-le"), (b"\xfe\xff", "utf-16-be")):
        if data.startswith(bom):
            return data[len(bom):].decode(enc, errors="replace")
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("cp1252", errors="replace")


# ------------------------------------------------------------------------------------------------ text formats (here)
class _HtmlText(html.parser.HTMLParser):
    BLOCK = {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "section", "article", "header",
             "footer", "pre", "blockquote", "table", "ul", "ol", "dt", "dd", "hr", "title"}
    SKIP = {"script", "style", "noscript", "template", "svg", "head"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.out, self.skip, self.title = [], 0, ""
        self._in_title = False

    def handle_starttag(self, tag, attrs):
        if tag == "title":
            self._in_title = True
        elif tag in self.SKIP:
            self.skip += 1
        if tag in self.BLOCK:
            self.out.append("\n")
        if tag in ("h1", "h2", "h3", "h4", "h5", "h6") and not self.skip:
            self.out.append("#" * int(tag[1]) + " ")
        elif tag == "li" and not self.skip:
            self.out.append("- ")
        elif tag in ("td", "th"):
            self.out.append(" | ")

    def handle_endtag(self, tag):
        if tag == "title":
            self._in_title = False
        elif tag in self.SKIP and self.skip:
            self.skip -= 1
        if tag in self.BLOCK:
            self.out.append("\n")

    def handle_data(self, data):
        if self._in_title:
            self.title += data
        elif not self.skip:
            self.out.append(data)


def html_text(source: str) -> str:
    p = _HtmlText()
    p.feed(source)
    p.close()
    text = "".join(p.out)
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    title = p.title.strip()
    return f"Title: {title}\n\n{text}" if title else text


def md_table(rows: list[list]) -> str:
    """Rows of cells -> a Markdown table (the first row is the header).  Pipes and newlines in a cell are escaped."""
    rows = [[("" if c is None else str(c)).replace("|", "\\|").replace("\r", "").replace("\n", " ") for c in r]
            for r in rows]
    rows = [r for r in rows if any(c.strip() for c in r)]
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    lines = ["| " + " | ".join(rows[0]) + " |", "|" + "---|" * width]
    lines += ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return "\n".join(lines)


def csv_text(source: str, delimiter: str) -> str:
    rows = list(csv.reader(io.StringIO(source), delimiter=delimiter))
    return md_table(rows)


# ------------------------------------------------------------------------------------------------ binary formats (child)
def _pdf(data: bytes) -> dict:
    import pypdfium2 as pdfium   # Apache-2.0 / BSD-3 (PDFium)
    pdf = pdfium.PdfDocument(data)
    try:
        pages, empty = [], []
        for i in range(len(pdf)):
            page = pdf[i]
            tp = page.get_textpage()
            text = tp.get_text_range().replace("\r\n", "\n").replace("\r", "\n").strip()
            tp.close()
            page.close()
            if not text:
                empty.append(i + 1)
            pages.append(f"--- Page {i + 1} ---\n{text}" if text else f"--- Page {i + 1} ---\n(no text on this page)")
        note = ""
        if empty and len(empty) == len(pages):
            note = "this PDF has no text layer (a scan or pictures only); its pages were not read as pictures"
        elif empty:
            note = f"{len(empty)} of {len(pages)} pages have no text layer (scans or pictures; not read)"
        return {"text": "\n\n".join(pages), "pages": len(pages), "note": note}
    finally:
        pdf.close()


def _docx(data: bytes) -> dict:
    import docx   # python-docx, MIT
    from docx.table import Table
    d = docx.Document(io.BytesIO(data))
    out = []
    for block in d.iter_inner_content():
        if isinstance(block, Table):
            out.append(md_table([[cell.text for cell in row.cells] for row in block.rows]))
            continue
        text = block.text.strip()
        if not text:
            continue
        style = (block.style.name if block.style is not None else "") or ""
        m = re.match(r"Heading (\d)", style)
        if m:
            out.append("#" * min(6, int(m.group(1))) + " " + text)
        elif style == "Title":
            out.append("# " + text)
        elif style.startswith("List"):
            out.append("- " + text)
        else:
            out.append(text)
    return {"text": "\n\n".join(out), "pages": None}


def _xlsx(data: bytes) -> dict:
    import openpyxl   # MIT
    wb = openpyxl.load_workbook(io.BytesIO(data), read_only=True, data_only=True)
    try:
        out = []
        for ws in wb.worksheets:
            rows = [["" if v is None else v for v in row] for row in ws.iter_rows(values_only=True)]
            while rows and not any(str(c).strip() for c in rows[-1]):
                rows.pop()
            table = md_table(rows)
            out.append(f"## Sheet: {ws.title}\n\n{table if table else '(empty)'}")
        return {"text": "\n\n".join(out), "pages": len(wb.worksheets)}
    finally:
        wb.close()


def _pptx(data: bytes) -> dict:
    import pptx   # python-pptx, MIT
    prs = pptx.Presentation(io.BytesIO(data))
    out = []

    def shape_text(shape, lines):
        if shape.shape_type == 6 and hasattr(shape, "shapes"):          # a group
            for s in shape.shapes:
                shape_text(s, lines)
            return
        if getattr(shape, "has_table", False) and shape.has_table:
            lines.append(md_table([[c.text for c in r.cells] for r in shape.table.rows]))
        elif getattr(shape, "has_text_frame", False) and shape.has_text_frame:
            t = shape.text_frame.text.strip()
            if t:
                lines.append(t)

    for n, slide in enumerate(prs.slides, 1):
        lines = []
        for shape in slide.shapes:
            shape_text(shape, lines)
        if slide.has_notes_slide:
            notes = slide.notes_slide.notes_text_frame.text.strip() if slide.notes_slide.notes_text_frame else ""
            if notes:
                lines.append("Notes: " + notes)
        out.append(f"## Slide {n}\n\n" + ("\n\n".join(lines) if lines else "(no text)"))
    return {"text": "\n\n".join(out), "pages": len(prs.slides)}


READERS = {"pdf": _pdf, "docx": _docx, "xlsx": _xlsx, "pptx": _pptx}


def _child_main(kind: str) -> int:
    data = sys.stdin.buffer.read()
    try:
        out = READERS[kind](data)
    except ImportError as e:
        out = {"error": f"the {PACKAGES[kind]} package is not installed ({e}); run the first-start step again"}
    except MemoryError:
        out = {"error": f"reading it needed more than {CHILD_MEMORY >> 20} MB of memory"}
    except Exception as e:   # noqa: BLE001 - any failure of a parser on a broken file is the answer
        out = {"error": f"{type(e).__name__}: {e}"[:400]}
    text = out.get("text")
    if isinstance(text, str) and len(text) > MAX_CHARS:     # only what the prompt can take goes back (and its length)
        out["text"], out["chars"] = text[:MAX_CHARS], len(text)
    sys.stdout.buffer.write(json.dumps(out, ensure_ascii=True).encode("ascii"))
    return 0


def zip_bomb(data: bytes) -> str:
    """Why an Office file (a zip) is not read - its parts unpack to too much - or "".  The sizes are the zip's own, and
    Python's zipfile (which python-docx, openpyxl and python-pptx read with) never unpacks a part past its stated size."""
    try:
        infos = zipfile.ZipFile(io.BytesIO(data)).infolist()
    except (zipfile.BadZipFile, ValueError, OSError):
        return ""                                            # the reader says what is wrong with it
    total = 0
    for info in infos:
        total += info.file_size
        if info.file_size > RATIO_FROM and info.file_size > MAX_RATIO * max(info.compress_size, 1):
            return (f"its part {info.filename[:80]!r} unpacks to {info.file_size / 1048576:.0f} MB, "
                    f"{info.file_size // max(info.compress_size, 1):,} times its packed size (a decompression bomb?)")
    if total > MAX_UNZIPPED:
        return f"it unpacks to {total / 1048576:.0f} MB; Office files up to {MAX_UNZIPPED >> 20} MB unpacked are read"
    return ""


def _limit_memory(limit: int) -> bool:
    """The reader process (this one) may not commit more than `limit` bytes: a Windows job object with a process memory
    limit, RLIMIT_AS elsewhere.  A larger allocation fails (MemoryError) instead of taking the RAM the model needs.
    Best effort: False when it could not be set."""
    try:
        if os.name == "nt":
            import ctypes
            from ctypes import wintypes

            class Basic(ctypes.Structure):
                _fields_ = [("PerProcessUserTimeLimit", ctypes.c_longlong), ("PerJobUserTimeLimit", ctypes.c_longlong),
                            ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                            ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                            ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD),
                            ("SchedulingClass", wintypes.DWORD)]

            class Extended(ctypes.Structure):
                _fields_ = [("BasicLimitInformation", Basic), ("IoInfo", ctypes.c_ulonglong * 6),
                            ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                            ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]

            k32 = ctypes.WinDLL("kernel32", use_last_error=True)
            k32.CreateJobObjectW.restype = wintypes.HANDLE
            k32.CreateJobObjectW.argtypes = (wintypes.LPVOID, wintypes.LPCWSTR)
            k32.SetInformationJobObject.argtypes = (wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD)
            k32.AssignProcessToJobObject.argtypes = (wintypes.HANDLE, wintypes.HANDLE)
            k32.GetCurrentProcess.restype = wintypes.HANDLE
            job = k32.CreateJobObjectW(None, None)          # left open: the limit lasts as long as this process
            if not job:
                return False
            info = Extended()
            info.BasicLimitInformation.LimitFlags = 0x100    # JOB_OBJECT_LIMIT_PROCESS_MEMORY
            info.ProcessMemoryLimit = limit
            return bool(k32.SetInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info))   # extended
                        and k32.AssignProcessToJobObject(job, k32.GetCurrentProcess()))
        import resource
        hard = resource.getrlimit(resource.RLIMIT_AS)[1]
        resource.setrlimit(resource.RLIMIT_AS, (limit if hard == resource.RLIM_INFINITY else min(limit, hard), hard))
        return True
    except (OSError, ValueError, AttributeError, ImportError):
        return False


def _run_child(kind: str, data: bytes, timeout: float = CHILD_TIMEOUT_S) -> dict:
    """Read a PDF / Office file in a separate Python with a time limit (a hostile file must not hang the server), a
    memory limit (the child sets it on itself) and a limit on how much of its answer is read."""
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    env = {k: v for k, v in os.environ.items() if not k.startswith("PYTHON")}
    cap = 12 * MAX_CHARS + OUTPUT_SLACK
    try:
        p = subprocess.Popen([sys.executable, "-I", os.path.abspath(__file__), kind],   # noqa: S603
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             creationflags=flags, env=env)
    except OSError as e:
        return {"error": f"the reader could not start ({e})"}
    late = threading.Event()

    def stop(timed_out=False):
        if timed_out:
            late.set()
        try:
            p.kill()
        except OSError:
            pass

    def feed():
        try:
            p.stdin.write(data)
        except (OSError, ValueError):                       # the child ended (or was stopped) before reading it all
            pass
        finally:
            try:
                p.stdin.close()
            except OSError:
                pass

    err = [b""]

    def drain():                                            # stderr: only its end is kept
        for chunk in iter(lambda: p.stderr.read(65536), b""):
            err[0] = (err[0] + chunk)[-65536:]

    timer = threading.Timer(timeout, stop, (True,))
    timer.daemon = True
    threads = [threading.Thread(target=feed, daemon=True), threading.Thread(target=drain, daemon=True)]
    timer.start()
    for t in threads:
        t.start()
    try:
        raw = p.stdout.read(cap + 1)
        over = len(raw) > cap
        if over:
            stop()
        p.wait()
    finally:
        timer.cancel()
        for t in threads:
            t.join(timeout=5)
        for f in (p.stdout, p.stderr):
            try:
                f.close()
            except OSError:
                pass
    if late.is_set():
        return {"error": f"reading it took longer than {timeout:.0f} s"}
    if over:
        return {"error": f"the reader's answer was over {cap:,} bytes"}
    try:
        out = json.loads(raw.decode("ascii") or "{}")
    except ValueError:
        out = {}
    if not isinstance(out, dict) or ("text" not in out and "error" not in out):
        tail = err[0].decode("utf-8", "replace").strip().splitlines()[-1:] or [f"exit code {p.returncode}"]
        return {"error": f"the reader stopped ({tail[0][:300]})"}
    return out


# ------------------------------------------------------------------------------------------------ the entry point
_cache: OrderedDict = OrderedDict()
_cache_lock = threading.Lock()
CACHE_ENTRIES = 32


def extract(data: bytes, name: str = "", media_type: str = "") -> Doc:
    """One document's bytes -> a Doc (never raises: a failure is Doc.error).  Cached by content, kind and name."""
    name = name or "document"
    if len(data) > MAX_BYTES:
        return Doc(name, "", error=f"it is {len(data) / 1048576:.0f} MB; documents up to {MAX_BYTES >> 20} MB are read")
    kind = sniff(data, name, media_type)
    key = hashlib.sha256(data).hexdigest() + "|" + kind + "|" + name
    with _cache_lock:
        if key in _cache:
            _cache.move_to_end(key)
            return _cache[key]
    t0 = time.monotonic()
    doc = _extract(data, name, kind)
    took = time.monotonic() - t0
    if doc.error:
        print(f"[strata] document {name!r}: not read ({doc.error})", flush=True)
    else:
        unit = {"xlsx": "sheet", "pptx": "slide"}.get(kind, "page")
        count = f", {doc.pages} {unit}{'s' * (doc.pages != 1)}" if doc.pages is not None else ""
        print(f"[strata] document {name!r} ({KIND_NAMES.get(kind, kind)}{count}): {len(doc.text):,} characters "
              f"in {took:.1f} s", flush=True)
    if not doc.error or "longer than" not in doc.error:     # a timeout may pass on a quieter machine: not cached
        with _cache_lock:
            _cache[key] = doc
            while len(_cache) > CACHE_ENTRIES:
                _cache.popitem(last=False)
    return doc


def _extract(data: bytes, name: str, kind: str) -> Doc:
    if kind == "legacy":
        ext = os.path.splitext(name)[1].lower()
        what = LEGACY.get(ext, "an old Office 97-2003 file")
        return Doc(name, kind, error=f"{what} is not supported; save it as .docx / .xlsx / .pptx or PDF")
    if kind == "binary":
        return Doc(name, kind, error="it is not a PDF, Word, Excel, PowerPoint, HTML or text file")
    full = 0                                                # the text's length before the reader cut it
    if kind in BINARY_KINDS:
        bomb = zip_bomb(data) if kind != "pdf" else ""
        if bomb:
            return Doc(name, kind, error=bomb)
        out = _run_child(kind, data)
        if out.get("error"):
            return Doc(name, kind, error=out["error"])
        doc = Doc(name, kind, out.get("text") or "", out.get("pages"), out.get("note") or "")
        full = out.get("chars") if isinstance(out.get("chars"), int) else 0
    else:
        text = decode_text(data)
        if kind == "html":
            text = html_text(text)
        elif kind in ("csv", "tsv"):
            text = csv_text(text, "," if kind == "csv" else "\t")
        doc = Doc(name, kind, text.replace("\r\n", "\n"))
    if not doc.text.strip() and not doc.note:
        doc.note = "no text was found in it"
    if max(len(doc.text), full) > MAX_CHARS:
        cut = max(len(doc.text), full)
        doc.text = doc.text[:MAX_CHARS]
        doc.note = (doc.note + "; " if doc.note else "") + \
            f"cut at {MAX_CHARS:,} of {cut:,} characters (the rest was not read)"
    return doc


# ------------------------------------------------------------------------------------------------ API content parts
DOCUMENT_PARTS = ("document", "file", "input_file")


def _data_url(s: str) -> tuple[str, bytes]:
    """'data:<type>;base64,<data>' or bare base64 -> (media type, bytes).  ValueError when it is neither."""
    mt = ""
    if s.startswith("data:"):
        head, _, s = s.partition(",")
        mt = head[5:].split(";")[0]
        if ";base64" not in head:
            from urllib.parse import unquote_to_bytes
            return mt, unquote_to_bytes(s)
    try:
        return mt, base64.b64decode(re.sub(r"\s+", "", s).replace("-", "+").replace("_", "/"), validate=True)
    except (binascii.Error, ValueError):
        raise ValueError("its data is not base64") from None


def _note(name: str, why: str) -> str:
    return f"[Document \"{_attr(name)}\" was not read: {why}]"


def part_text(part: dict) -> str:
    """An API document / file part -> the text that goes into the prompt in its place (never raises)."""
    kind = part.get("type")
    if kind == "document":                                         # Anthropic
        src = part.get("source") or {}
        name = part.get("title") or "document"
        context = part.get("context")
        st = src.get("type")
        if st == "base64":
            try:
                mt, data = _data_url(src.get("data") or "")
            except ValueError as e:
                return _note(name, str(e))
            doc = extract(data, name if "." in name else name + _ext_for(src.get("media_type")),
                          src.get("media_type") or mt)
            # a copy: extract() returns the cached Doc, shared by every request (and thread) with the same file
            doc = Doc(name, doc.kind, doc.text, doc.pages, doc.note, doc.error)
        elif st == "text":
            doc = Doc(name, "text", str(src.get("data") or ""))
        elif st == "content":
            items = src.get("content")
            if isinstance(items, str):
                text = items
            else:
                text = "\n\n".join(b.get("text", "") for b in items or [] if isinstance(b, dict)
                                   and b.get("type") == "text")
            doc = Doc(name, "text", text)
        elif st == "url":
            return _note(name, "this server does not fetch URLs; send the file itself (base64)")
        elif st == "file":
            return _note(name, "this server keeps no uploaded files; send the file itself (base64)")
        else:
            return _note(name, f"unknown source type {st!r}")
        text = doc.render()
        if context and not doc.error:
            text = text.replace(">\n", f">\nContext: {context}\n", 1)
        return text
    if kind == "file":                                             # OpenAI chat completions
        f = part.get("file") or {}
        name = f.get("filename") or "file"
        if f.get("file_data"):
            try:
                mt, data = _data_url(f["file_data"])
            except ValueError as e:
                return _note(name, str(e))
            return extract(data, name, mt).render()
        return _note(name, "this server keeps no uploaded files (file_id); send file_data")
    if kind == "input_file":                                       # OpenAI Responses
        name = part.get("filename") or "file"
        if part.get("file_data"):
            try:
                mt, data = _data_url(part["file_data"])
            except ValueError as e:
                return _note(name, str(e))
            return extract(data, name, mt).render()
        if part.get("file_url"):
            return _note(name, "this server does not fetch URLs; send file_data")
        return _note(name, "this server keeps no uploaded files (file_id); send file_data")
    return ""


def _ext_for(media_type) -> str:
    for ext, kind in EXT_KINDS.items():
        if MEDIA_KINDS.get((media_type or "").lower()) == kind:
            return ext
    return ""


def estimate_seconds(tokens: int, rate: float | None = None) -> float:
    """How long the model takes to read `tokens` new prompt tokens the first time (follow-ups reuse the cache)."""
    return tokens / (rate if rate and rate > 0 else READ_TOK_S)


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in READERS:
        sys.exit("usage: documents.py pdf|docx|xlsx|pptx < file")
    _limit_memory(CHILD_MEMORY)
    sys.exit(_child_main(sys.argv[1]))
