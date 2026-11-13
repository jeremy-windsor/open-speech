"""Helpers for the web UI tests: the Node runner, an HTML tree with a CSS-selector subset, accessible names,
a small CSS parser with token resolution and WCAG contrast, and a JS lexer for the architecture lint.

Nothing here imports the app (src.main); it reads files only.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
STATIC = REPO / "src" / "static"
JS_ROOT = STATIC / "js"
CSS_ROOT = STATIC / "css"
PARTIALS_DIR = STATIC / "partials"
SPRITE_PATH = STATIC / "icons" / "sprite.svg"
TOKENS_PATH = CSS_ROOT / "tokens.css"

# The shell, served at /web (and /static/index.html). It replaced the old UI.
SHELL_PATH = "src/static/index.html"

VIEW_PARTIALS = ("reader", "speak", "transcribe", "talk", "voices", "studio", "history", "models", "settings")
SLOT_PARTIALS = ("reader-player", "speak-live")


# ───────────────────────────── Node ─────────────────────────────


def require_node() -> str:
    """The node executable. Skips the test locally when Node is missing; fails on CI."""
    node = shutil.which("node")
    if node:
        return node
    if os.environ.get("CI"):
        pytest.fail("node is required on CI for the web UI behaviour tests")
    pytest.skip("node is not installed")


# A test file's whole process tree (node --test runs each file in a child process) may use this much
# memory. A test that loops while allocating audio buffers used to grow to ~50 GB and starve the host;
# now the allocation fails inside node and the test fails.
NODE_MEMORY_LIMIT_MB = 2048


def run_node_test(path: str | Path, timeout: int = 120, memory_limit_mb: int = NODE_MEMORY_LIMIT_MB) -> str:
    """Run `node --test <path>` from the repo root; fail with the failing tests and their output.

    The run is capped at `memory_limit_mb` (Windows) and its whole process tree is killed when it
    ends or times out, so no child node process outlives the test.
    """
    node = require_node()
    target = str(path)
    returncode, output, peak_mb = _run_process_tree(
        [node, "--test", "--test-reporter=spec", "--test-timeout=30000", target],
        timeout=timeout,
        memory_limit_mb=memory_limit_mb,
        env={**os.environ, "NODE_NO_WARNINGS": "1", "FORCE_COLOR": "0", "NO_COLOR": "1"},
    )
    if returncode is None:
        pytest.fail(f"node --test {target} timed out after {timeout} s (its processes were stopped)\n\n{output[-12000:]}",
                    pytrace=False)
    if returncode != 0:
        failing = [line.strip() for line in output.splitlines() if line.strip().startswith(("✖", "not ok"))]
        tail = output[-12000:]
        summary = "\n".join(dict.fromkeys(failing))
        if peak_mb is not None and peak_mb >= memory_limit_mb * 0.9:
            summary = (f"the run reached its {memory_limit_mb} MB memory cap (peak {peak_mb:.0f} MB): "
                       f"something allocates without bound\n{summary}")
        pytest.fail(f"node --test {target} failed (exit {returncode})\n{summary}\n\n{tail}", pytrace=False)
    return output


def _run_process_tree(argv: list[str], *, timeout: int, memory_limit_mb: int, env: dict) -> tuple[int | None, str, float | None]:
    """Run argv from the repo root; return (exit code or None on timeout, output, peak memory in MB or None).

    Windows: inside a job object with a memory cap whose closing kills every process in it.
    Elsewhere: in its own process group, killed as a group on timeout (no memory cap).
    """
    popen = dict(cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
    if os.name != "nt":
        import signal

        proc = subprocess.Popen(argv, start_new_session=True, **popen)
        try:
            out, _ = proc.communicate(timeout=timeout)
            code = proc.returncode
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            out, _ = proc.communicate()
            code = None
        return code, out.decode("utf-8", "replace"), None

    job = _WindowsJob(memory_limit_mb)
    proc = subprocess.Popen(argv, creationflags=_CREATE_SUSPENDED, **popen)
    try:
        job.adopt(proc)
        try:
            out, _ = proc.communicate(timeout=timeout)
            code = proc.returncode
        except subprocess.TimeoutExpired:
            job.close()  # kills the whole tree, so the pipes reach EOF
            out, _ = proc.communicate()
            code = None
        return code, out.decode("utf-8", "replace"), job.peak_mb()
    finally:
        job.close()


_CREATE_SUSPENDED = 0x4


class _WindowsJob:
    """A job object with a memory cap for all its processes; closing it kills them (Windows only)."""

    def __init__(self, memory_limit_mb: int):
        import ctypes
        import ctypes.wintypes as wt

        class Basic(ctypes.Structure):
            _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                        ("LimitFlags", wt.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                        ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wt.DWORD),
                        ("Affinity", ctypes.c_size_t), ("PriorityClass", wt.DWORD), ("SchedulingClass", wt.DWORD)]

        class Extended(ctypes.Structure):
            _fields_ = [("Basic", Basic), ("Io", ctypes.c_uint64 * 6), ("ProcessMemoryLimit", ctypes.c_size_t),
                        ("JobMemoryLimit", ctypes.c_size_t), ("PeakProcessMemoryUsed", ctypes.c_size_t),
                        ("PeakJobMemoryUsed", ctypes.c_size_t)]

        self._ctypes = ctypes
        self._k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self._k32.CreateJobObjectW.restype = wt.HANDLE
        self._k32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wt.LPCWSTR]
        self._k32.SetInformationJobObject.argtypes = [wt.HANDLE, ctypes.c_int, ctypes.c_void_p, wt.DWORD]
        self._k32.QueryInformationJobObject.argtypes = [wt.HANDLE, ctypes.c_int, ctypes.c_void_p, wt.DWORD, ctypes.c_void_p]
        self._k32.AssignProcessToJobObject.argtypes = [wt.HANDLE, wt.HANDLE]
        self._k32.CloseHandle.argtypes = [wt.HANDLE]
        self._ntdll = ctypes.WinDLL("ntdll")
        self._ntdll.NtResumeProcess.argtypes = [wt.HANDLE]
        self._info = Extended()
        self._info.Basic.LimitFlags = 0x200 | 0x2000  # JOB_OBJECT_LIMIT_JOB_MEMORY | KILL_ON_JOB_CLOSE
        self._info.JobMemoryLimit = memory_limit_mb * 2**20
        self._handle = self._k32.CreateJobObjectW(None, None)
        if not self._handle or not self._k32.SetInformationJobObject(
                self._handle, 9, ctypes.byref(self._info), ctypes.sizeof(self._info)):
            raise OSError(ctypes.get_last_error(), "could not create the job object for the node run")

    def adopt(self, proc: subprocess.Popen) -> None:
        """Put the (suspended) process in the job, then let it run."""
        if not self._k32.AssignProcessToJobObject(self._handle, int(proc._handle)):
            proc.kill()
            raise OSError(self._ctypes.get_last_error(), "could not put the node run in its job object")
        self._ntdll.NtResumeProcess(int(proc._handle))

    def peak_mb(self) -> float | None:
        if not self._handle:
            return None
        ctypes = self._ctypes
        if not self._k32.QueryInformationJobObject(self._handle, 9, ctypes.byref(self._info), ctypes.sizeof(self._info), None):
            return None
        return self._info.PeakJobMemoryUsed / 2**20

    def close(self) -> None:
        if self._handle:
            self._k32.CloseHandle(self._handle)
            self._handle = None


def js_test_files(subdir: str) -> list[str]:
    """The sorted `tests/js/<subdir>/*.test.js` paths (repo-relative, POSIX), one pytest case each."""
    folder = REPO / "tests" / "js" / subdir
    return sorted(p.relative_to(REPO).as_posix() for p in folder.glob("*.test.js"))


def run_node_script(source: str, timeout: int = 60, memory_limit_mb: int = NODE_MEMORY_LIMIT_MB) -> str:
    """Run an ES module script with `node --input-type=module -e` from the repo root; return its output."""
    return run_node_command([require_node(), "--input-type=module", "-e", source], what="node script",
                            timeout=timeout, memory_limit_mb=memory_limit_mb)


def run_node_command(argv: list[str], *, what: str, timeout: int = 60,
                     memory_limit_mb: int = NODE_MEMORY_LIMIT_MB) -> str:
    """Run a node command line (argv[0] is node) from the repo root with run_node_test's memory cap and
    process-tree kill; return its output, stdout and stderr together (node's warnings off). Fail on a
    timeout, the cap or a non-zero exit. These runs had no cap."""
    returncode, output, peak_mb = _run_process_tree(
        argv, timeout=timeout, memory_limit_mb=memory_limit_mb, env={**os.environ, "NODE_NO_WARNINGS": "1"})
    if returncode is None:
        pytest.fail(f"{what} timed out after {timeout} s (its processes were stopped)\n\n{output[-12000:]}",
                    pytrace=False)
    if returncode != 0:
        capped = ""
        if peak_mb is not None and peak_mb >= memory_limit_mb * 0.9:
            capped = (f": the run reached its {memory_limit_mb} MB memory cap (peak {peak_mb:.0f} MB), "
                      "something allocates without bound")
        pytest.fail(f"{what} failed (exit {returncode}){capped}\n\n{output[-12000:]}", pytrace=False)
    return output


# ───────────────────────────── HTML ─────────────────────────────

VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "track", "wbr"}
IMPLIED_END = {
    "li": {"li"}, "option": {"option"}, "optgroup": {"optgroup", "option"}, "dt": {"dt", "dd"}, "dd": {"dt", "dd"},
    "tr": {"tr", "td", "th"}, "td": {"td", "th"}, "th": {"td", "th"}, "p": {"p"},
}


@dataclass(eq=False)
class Node:
    """An element ('#text' for text nodes, '#document' for the root)."""

    tag: str
    attrs: dict[str, str] = field(default_factory=dict)
    children: list["Node"] = field(default_factory=list)
    text: str = ""
    line: int = 0
    parent: "Node | None" = None
    source: str = ""

    @property
    def is_element(self) -> bool:
        return not self.tag.startswith("#")

    def get(self, name: str, default: str | None = None) -> str | None:
        return self.attrs.get(name, default)

    def has(self, name: str) -> bool:
        return name in self.attrs

    @property
    def id(self) -> str:
        return self.attrs.get("id", "")

    @property
    def classes(self) -> list[str]:
        return (self.attrs.get("class") or "").split()

    def elements(self):
        """Every element below this node, in document order."""
        for child in self.children:
            if child.is_element:
                yield child
                yield from child.elements()

    def ancestors(self):
        node = self.parent
        while node is not None:
            yield node
            node = node.parent

    def text_content(self) -> str:
        if self.tag == "#text":
            return self.text
        return "".join(c.text_content() for c in self.children)

    def __repr__(self) -> str:
        bits = [self.tag]
        if self.id:
            bits.append(f"#{self.id}")
        return f"<{''.join(bits)} line {self.line}{' ' + self.source if self.source else ''}>"


class _TreeBuilder(HTMLParser):
    def __init__(self, source: str):
        super().__init__(convert_charrefs=True)
        self.root = Node("#document", source=source)
        self.stack = [self.root]
        self.source = source

    def _open(self, tag, attrs, self_closing=False):
        if tag in IMPLIED_END:
            while len(self.stack) > 1 and self.stack[-1].tag in IMPLIED_END[tag]:
                self.stack.pop()
        node = Node(tag, {k: ("" if v is None else v) for k, v in attrs}, line=self.getpos()[0], parent=self.stack[-1], source=self.source)
        self.stack[-1].children.append(node)
        if tag not in VOID and not self_closing:
            self.stack.append(node)

    def handle_starttag(self, tag, attrs):
        self._open(tag, attrs)

    def handle_startendtag(self, tag, attrs):
        self._open(tag, attrs, self_closing=True)

    def handle_endtag(self, tag):
        for i in range(len(self.stack) - 1, 0, -1):
            if self.stack[i].tag == tag:
                del self.stack[i:]
                return

    def handle_data(self, data):
        if data:
            self.stack[-1].children.append(Node("#text", text=data, line=self.getpos()[0], parent=self.stack[-1], source=self.source))


def parse_html(path_or_text: str | Path) -> Node:
    """Parse a file (a Path, or a str path that exists) or markup text into a Node tree."""
    source = ""
    if isinstance(path_or_text, Path) or (isinstance(path_or_text, str) and "<" not in path_or_text and len(path_or_text) < 400 and (REPO / path_or_text).exists()):
        path = Path(path_or_text)
        if not path.is_absolute():
            path = REPO / path
        text = path.read_text(encoding="utf-8")
        source = path.relative_to(REPO).as_posix() if path.is_relative_to(REPO) else str(path)
    else:
        text = str(path_or_text)
    builder = _TreeBuilder(source)
    builder.feed(text)
    builder.close()
    return builder.root


# ─────────── selectors (the same subset as the JS fake DOM) ───────────

_SEL_CACHE: dict[str, list] = {}


class _SelParser:
    def __init__(self, text: str):
        self.s = text
        self.i = 0

    def error(self, msg):
        return ValueError(f"bad selector {self.s!r}: {msg}")

    def peek(self):
        return self.s[self.i] if self.i < len(self.s) else ""

    def ws(self):
        start = self.i
        while self.i < len(self.s) and self.s[self.i].isspace():
            self.i += 1
        return self.i > start

    def ident(self):
        m = re.compile(r"(?:\\.|[\w\-\u00a0-\uffff])+").match(self.s, self.i)
        if not m:
            return ""
        self.i = m.end()
        return re.sub(r"\\(.)", r"\1", m.group(0))

    def parse_list(self):
        out = []
        while True:
            self.ws()
            out.append(self.parse_complex())
            self.ws()
            if self.i >= len(self.s) or self.peek() == ")":
                break
            if self.peek() == ",":
                self.i += 1
                continue
            raise self.error(f"unexpected {self.peek()!r}")
        return out

    def parse_complex(self):
        parts = [(None, self.parse_compound())]
        while True:
            had_ws = self.ws()
            c = self.peek()
            if c in ("", ",", ")"):
                break
            comb = " "
            if c in ">+~":
                comb = c
                self.i += 1
                self.ws()
            elif not had_ws:
                raise self.error(f"unexpected {c!r}")
            parts.append((comb, self.parse_compound()))
        return parts

    def parse_compound(self):
        comp = {"tag": None, "id": None, "classes": [], "attrs": [], "not": [], "pseudos": []}
        any_ = False
        if self.peek() == "*":
            self.i += 1
            any_ = True
        elif re.match(r"[\w\\-]", self.peek() or " "):
            comp["tag"] = self.ident().lower()
            any_ = True
        while True:
            c = self.peek()
            if c == "#":
                self.i += 1
                comp["id"] = self.ident()
            elif c == ".":
                self.i += 1
                comp["classes"].append(self.ident())
            elif c == "[":
                self.i += 1
                comp["attrs"].append(self.parse_attr())
            elif c == ":":
                self.i += 1
                name = self.ident().lower()
                if name in ("not", "is", "where"):
                    if self.peek() != "(":
                        raise self.error("missing (")
                    self.i += 1
                    inner = self.parse_list()
                    if self.peek() != ")":
                        raise self.error("missing )")
                    self.i += 1
                    comp["not" if name == "not" else "pseudos"].append(inner if name == "not" else ("is", inner))
                elif name in ("first-child", "last-child", "checked", "disabled", "empty"):
                    comp["pseudos"].append((name, None))
                else:
                    raise self.error(f":{name} is not supported")
            else:
                break
            any_ = True
        if not any_:
            raise self.error("empty compound")
        return comp

    def parse_attr(self):
        self.ws()
        name = self.ident().lower()
        self.ws()
        c = self.peek()
        if c == "]":
            self.i += 1
            return (name, None, None)
        if c == "=":
            op = "="
            self.i += 1
        elif c in "~|^$*" and self.s[self.i + 1:self.i + 2] == "=":
            op = c + "="
            self.i += 2
        else:
            raise self.error("bad attribute selector")
        self.ws()
        q = self.peek()
        if q in ("'", '"'):
            end = self.s.index(q, self.i + 1)
            value = self.s[self.i + 1:end]
            self.i = end + 1
        else:
            value = self.ident()
        self.ws()
        if self.peek() != "]":
            raise self.error("missing ]")
        self.i += 1
        return (name, op, value)


def _parse_selector(css: str):
    if css not in _SEL_CACHE:
        _SEL_CACHE[css] = _SelParser(css).parse_list()
    return _SEL_CACHE[css]


def _attr_ok(node: Node, spec) -> bool:
    name, op, value = spec
    if name not in node.attrs:
        return False
    if op is None:
        return True
    actual = node.attrs[name]
    if op == "=":
        return actual == value
    if op == "~=":
        return value in actual.split()
    if op == "|=":
        return actual == value or actual.startswith(value + "-")
    if op == "^=":
        return bool(value) and actual.startswith(value)
    if op == "$=":
        return bool(value) and actual.endswith(value)
    if op == "*=":
        return bool(value) and value in actual
    return False


def _siblings(node: Node) -> list[Node]:
    return [c for c in node.parent.children if c.is_element] if node.parent else [node]


def _compound_ok(node: Node, comp) -> bool:
    if not node.is_element:
        return False
    if comp["tag"] and node.tag != comp["tag"]:
        return False
    if comp["id"] is not None and node.attrs.get("id") != comp["id"]:
        return False
    for cls in comp["classes"]:
        if cls not in node.classes:
            return False
    for spec in comp["attrs"]:
        if not _attr_ok(node, spec):
            return False
    for inner in comp["not"]:
        if any(_complex_ok(node, c) for c in inner):
            return False
    for name, arg in comp["pseudos"]:
        sibs = _siblings(node)
        if name == "is" and not any(_complex_ok(node, c) for c in arg):
            return False
        if name == "first-child" and sibs[0] is not node:
            return False
        if name == "last-child" and sibs[-1] is not node:
            return False
        if name == "checked" and "checked" not in node.attrs and "selected" not in node.attrs:
            return False
        if name == "disabled" and "disabled" not in node.attrs:
            return False
        if name == "empty" and any(c.is_element or c.text for c in node.children):
            return False
    return True


def _complex_ok(node: Node, parts, index: int | None = None) -> bool:
    index = len(parts) - 1 if index is None else index
    comb, comp = parts[index]
    if not _compound_ok(node, comp):
        return False
    if index == 0:
        return True
    comb = parts[index][0]
    if comb == ">":
        return node.parent is not None and node.parent.is_element and _complex_ok(node.parent, parts, index - 1)
    if comb == " ":
        return any(a.is_element and _complex_ok(a, parts, index - 1) for a in node.ancestors())
    sibs = _siblings(node)
    pos = sibs.index(node)
    if comb == "+":
        return pos > 0 and _complex_ok(sibs[pos - 1], parts, index - 1)
    if comb == "~":
        return any(_complex_ok(s, parts, index - 1) for s in sibs[:pos])
    return False


def matches(node: Node, css: str) -> bool:
    return any(_complex_ok(node, c) for c in _parse_selector(css))


def select(root: Node, css: str) -> list[Node]:
    """Every element under `root` matching `css` (tag, #id, .class, [attr], [attr="v"], ~= ^= $= *= |=,
    :not(), :is(), :first-child, :last-child, descendant ' ', '>', '+', '~' and comma lists)."""
    parsed = _parse_selector(css)
    return [el for el in root.elements() if any(_complex_ok(el, c) for c in parsed)]


def select_one(root: Node, css: str) -> Node | None:
    found = select(root, css)
    return found[0] if found else None


def by_id(root: Node, id_: str) -> Node | None:
    for el in root.elements():
        if el.attrs.get("id") == id_:
            return el
    return None


# ─────────── accessible names ───────────


def is_aria_hidden(node: Node) -> bool:
    """True when the node or an ancestor has aria-hidden="true"."""
    n: Node | None = node
    while n is not None:
        if n.is_element and n.attrs.get("aria-hidden") == "true":
            return True
        n = n.parent
    return False


def _name_from_content(node: Node) -> str:
    if node.tag == "#text":
        return node.text
    if not node.is_element or node.attrs.get("aria-hidden") == "true":
        return ""
    if node.tag == "svg":
        return node.attrs.get("aria-label", "") if node.attrs.get("role") == "img" else ""
    if node.tag == "img":
        return node.attrs.get("alt", "")
    label = node.attrs.get("aria-label")
    if label and label.strip():
        return label
    return "".join(_name_from_content(c) for c in node.children)


NAME_FROM_CONTENT_ROLES = {"button", "link", "option", "menuitem", "menuitemradio", "menuitemcheckbox", "tab", "radio", "checkbox", "switch", "heading", "treeitem"}


def accessible_name(node: Node, root: Node) -> str:
    """aria-labelledby, aria-label, <label for>, a wrapping <label>, the text of a button or link (and
    name-from-content roles), then title. aria-hidden subtrees are ignored; a placeholder never counts."""
    labelledby = node.attrs.get("aria-labelledby")
    if labelledby:
        parts = []
        for ref in labelledby.split():
            target = by_id(root, ref)
            if target is not None:
                parts.append(_name_from_content(target).strip())
        name = " ".join(p for p in parts if p).strip()
        if name:
            return name
    label = node.attrs.get("aria-label")
    if label and label.strip():
        return label.strip()
    if node.tag in ("input", "select", "textarea", "button", "meter", "output", "progress"):
        if node.id:
            texts = [_name_from_content(l).strip() for l in select(root, "label") if l.attrs.get("for") == node.id]
            name = " ".join(t for t in texts if t).strip()
            if name:
                return name
        for anc in node.ancestors():
            if anc.tag == "label":
                text = "".join(_name_from_content(c) for c in anc.children if c is not node).strip()
                if text:
                    return text
                break
    if node.tag in ("button", "a", "summary", "option") or node.attrs.get("role") in NAME_FROM_CONTENT_ROLES:
        text = re.sub(r"\s+", " ", "".join(_name_from_content(c) for c in node.children)).strip()
        if text:
            return text
    if node.tag == "input" and node.attrs.get("type") in ("submit", "reset", "button") and node.attrs.get("value"):
        return node.attrs["value"]
    title = node.attrs.get("title")
    if title and title.strip():
        return title.strip()
    return ""


def visible_text(node: Node) -> str:
    """The text a sighted user sees (aria-hidden content excluded, visually-hidden text included)."""
    return re.sub(r"\s+", " ", "".join(_name_from_content(c) for c in node.children if c.tag != "svg")).strip()


# ─────────── markup files ───────────


def shell_path() -> Path:
    return REPO / SHELL_PATH


def partials() -> dict[str, Node]:
    """{name: tree} for every partial in src/static/partials/."""
    if not PARTIALS_DIR.exists():
        return {}
    return {p.stem: parse_html(p) for p in sorted(PARTIALS_DIR.glob("*.html"))}


def all_markup() -> list[tuple[str, Node]]:
    """The shell (when it exists) plus every partial: [(name, tree)]."""
    out = []
    if shell_path().exists():
        out.append(("shell", parse_html(shell_path())))
    out.extend(partials().items())
    return out


STUB_MARKERS = ("data-stub", "data-stub-body")
STUB_MAX_ELEMENTS = 4   # header, h1 and the empty stub body (a slot stub is one element)
STUB_CONTROLS = {"button", "input", "select", "textarea", "a", "form", "dialog", "details", "canvas", "audio", "video"}


def stub_markers(tree: Node) -> list[Node]:
    """The elements carrying F2's placeholder markers (data-stub, data-stub-body)."""
    return [el for el in tree.elements() if any(m in el.attrs for m in STUB_MARKERS)]


def is_stub(tree: Node) -> bool:
    """True only for F2's bare placeholder partial: a stub marker, at most a header, the h1 and the empty stub
    body, and no controls. A partial that keeps a marker after real markup landed is NOT a stub, so the
    audits (live regions, headings) still run on it; audit_stub_markers() flags the leftover marker."""
    if not stub_markers(tree):
        return False
    elements = list(tree.elements())
    return len(elements) <= STUB_MAX_ELEMENTS and not any(el.tag in STUB_CONTROLS for el in elements)


def sprite_ids() -> set[str] | None:
    """The symbol ids in sprite.svg, or None before the sprite exists."""
    if not SPRITE_PATH.exists():
        return None
    return set(re.findall(r"<symbol\b[^>]*\bid=\"([^\"]+)\"", SPRITE_PATH.read_text(encoding="utf-8")))


# ───────────────────────────── CSS ─────────────────────────────


def _strip_css_comments(text: str) -> str:
    out = []
    i = 0
    quote = ""
    while i < len(text):
        c = text[i]
        if quote:
            out.append(c)
            if c == "\\" and i + 1 < len(text):
                out.append(text[i + 1])
                i += 2
                continue
            if c == quote:
                quote = ""
        elif c in "\"'":
            quote = c
            out.append(c)
        elif text.startswith("/*", i):
            end = text.find("*/", i + 2)
            i = len(text) if end < 0 else end + 2
            out.append(" ")
            continue
        else:
            out.append(c)
        i += 1
    return "".join(out)


def _split_declarations(block: str) -> list[tuple[str, str]]:
    decls = []
    depth = 0
    quote = ""
    current = []
    for c in block:
        if quote:
            current.append(c)
            if c == quote:
                quote = ""
            continue
        if c in "\"'":
            quote = c
        elif c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
        elif c == ";" and depth == 0:
            decls.append("".join(current))
            current = []
            continue
        current.append(c)
    decls.append("".join(current))
    out = []
    for d in decls:
        if ":" not in d:
            continue
        name, _, value = d.partition(":")
        name = name.strip()
        value = value.strip()
        if name:
            out.append((name if name.startswith("--") else name.lower(), value))
    return out


def _find_block_end(text: str, start: int) -> int:
    """Index of the '}' that closes the block whose '{' is at text[start]."""
    depth = 0
    quote = ""
    i = start
    while i < len(text):
        c = text[i]
        if quote:
            if c == "\\":
                i += 2
                continue
            if c == quote:
                quote = ""
        elif c in "\"'":
            quote = c
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return len(text)


def _parse_css(text: str, chain: tuple[str, ...], out: list) -> None:
    i = 0
    n = len(text)
    while i < n:
        brace = text.find("{", i)
        semi = text.find(";", i)
        if brace < 0:
            break
        if 0 <= semi < brace and text[i:semi].strip().startswith("@"):
            # statement at-rule: @import, @layer a, b;, @charset
            out.append((text[i:semi].strip(), [], chain))
            i = semi + 1
            continue
        prelude = text[i:brace].strip()
        end = _find_block_end(text, brace)
        body = text[brace + 1:end]
        if prelude.startswith("@"):
            keyword = prelude.split()[0].lower()
            if keyword in ("@media", "@supports", "@layer", "@container", "@scope", "@document"):
                _parse_css(body, chain + (re.sub(r"\s+", " ", prelude),), out)
            else:
                # @font-face, @keyframes, @property, @page: keep as one rule
                out.append((re.sub(r"\s+", " ", prelude), _split_declarations(body) if keyword != "@keyframes" else [("@keyframes-body", body.strip())], chain))
        else:
            out.append((re.sub(r"\s+", " ", prelude), _split_declarations(body), chain))
        i = end + 1


def css_rules(path: str | Path) -> list[tuple[str, list[tuple[str, str]], tuple[str, ...]]]:
    """[(selector, [(property, value)], at_rule_chain)] for a CSS file (or CSS text). Handles comments,
    strings and @media/@supports/@layer blocks. CSS nesting is not used anywhere, so it is not parsed."""
    p = Path(path) if not isinstance(path, Path) else path
    if isinstance(path, str) and ("{" in path or "\n" in path):
        text = path
    else:
        if not p.is_absolute():
            p = REPO / p
        text = p.read_text(encoding="utf-8")
    out: list = []
    _parse_css(_strip_css_comments(text), (), out)
    return out


def css_files() -> list[Path]:
    return sorted(CSS_ROOT.rglob("*.css")) if CSS_ROOT.exists() else []


def _block_tokens(rules, selector_pred, chain_pred) -> dict[str, str]:
    out: dict[str, str] = {}
    for selector, decls, chain in rules:
        if selector_pred(selector) and chain_pred(chain):
            for name, value in decls:
                if name.startswith("--"):
                    out[name] = value
    return out


def _resolve_vars(value: str, table: dict[str, str], depth: int = 0) -> str:
    if depth > 20:
        raise ValueError(f"var() cycle in {value!r}")

    def repl(m):
        name = m.group(1).strip()
        fallback = m.group(2)
        if name in table:
            return _resolve_vars(table[name], table, depth + 1)
        if fallback is not None:
            return _resolve_vars(fallback.strip(), table, depth + 1)
        raise KeyError(name)

    prev = None
    while prev != value:
        prev = value
        value = re.sub(r"var\(\s*(--[\w-]+)\s*(?:,\s*((?:[^()]|\([^()]*\))*))?\)", repl, value)
    return value


def tokens(theme: str = "light", contrast: bool = False, path: str | Path | None = None) -> dict[str, str]:
    """Resolved custom properties from tokens.css (var() substituted).
    light: the plain `:root` block; dark: `:root[data-theme="dark"]` on top of light; contrast=True adds the
    `@media (prefers-contrast: more)` overrides for that theme."""
    rules = css_rules(path or TOKENS_PATH)
    no_media = lambda chain: not any(c.lower().startswith("@media") for c in chain)  # noqa: E731
    table = _block_tokens(rules, lambda s: s.strip() == ":root", no_media)
    if theme == "dark":
        table.update(_block_tokens(rules, lambda s: re.fullmatch(r":root\[data-theme=\"?dark\"?\]", s.strip()) is not None, no_media))
    if contrast:
        in_contrast = lambda chain: any("prefers-contrast" in c for c in chain)  # noqa: E731
        table.update(_block_tokens(rules, lambda s: s.strip() == ":root", in_contrast))
        if theme == "dark":
            table.update(_block_tokens(rules, lambda s: "dark" in s, in_contrast))
    resolved = {}
    for name in table:
        try:
            resolved[name] = _resolve_vars(table[name], table)
        except KeyError:
            resolved[name] = table[name]
    return resolved


_NAMED = {"white": (255, 255, 255), "black": (0, 0, 0)}


def parse_color(value: str) -> tuple[float, float, float, float]:
    """#rgb, #rrggbb, #rrggbbaa, rgb()/rgba() (comma or space syntax, / alpha), white, black, transparent."""
    v = value.strip().lower()
    if v == "transparent":
        return (0.0, 0.0, 0.0, 0.0)
    if v in _NAMED:
        return (*map(float, _NAMED[v]), 1.0)
    m = re.fullmatch(r"#([0-9a-f]{3,8})", v)
    if m:
        h = m.group(1)
        if len(h) in (3, 4):
            h = "".join(c * 2 for c in h)
        r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
        a = int(h[6:8], 16) / 255 if len(h) == 8 else 1.0
        return (float(r), float(g), float(b), a)
    m = re.fullmatch(r"rgba?\(\s*([^)]*)\)", v)
    if m:
        parts = re.split(r"[\s,/]+", m.group(1).strip())
        nums = []
        for p in parts:
            nums.append(float(p[:-1]) * 2.55 if p.endswith("%") else float(p))
        r, g, b = nums[:3]
        a = 1.0
        if len(parts) > 3:
            a = float(parts[3][:-1]) / 100 if parts[3].endswith("%") else float(parts[3])
        return (r, g, b, a)
    raise ValueError(f"unsupported colour {value!r}")


def composite(fg: tuple[float, float, float, float], bg: tuple[float, float, float]) -> tuple[float, float, float]:
    """Alpha-composite fg over an opaque bg."""
    r, g, b, a = fg
    return (r * a + bg[0] * (1 - a), g * a + bg[1] * (1 - a), b * a + bg[2] * (1 - a))


def resolve_color(value: str, table: dict[str, str] | None = None, over: str | tuple | None = None) -> tuple[float, float, float]:
    """An opaque sRGB colour for a token value: var() resolved, `color-mix(in srgb, X p%, transparent|Y)`
    mixed, and anything translucent composited over `over` (default white)."""
    table = table or {}
    v = _resolve_vars(value.strip(), table)
    backdrop = (255.0, 255.0, 255.0)
    if over is not None:
        backdrop = over if isinstance(over, tuple) else resolve_color(over, table)
    rgba = _color_rgba(v, table)
    # a screen shows 8-bit channels: round the composite as a contrast checker does
    return tuple(float(round(c)) for c in composite(rgba, backdrop))


def _color_rgba(v: str, table: dict[str, str]) -> tuple[float, float, float, float]:
    v = v.strip()
    m = re.fullmatch(r"color-mix\(\s*in\s+srgb\s*,\s*(.+?)\s+(\d+(?:\.\d+)?)%\s*,\s*(.+?)\s*\)", v, flags=re.S)
    if m:
        a = _color_rgba(_resolve_vars(m.group(1), table), table)
        p = float(m.group(2)) / 100
        b = _color_rgba(_resolve_vars(m.group(3), table), table)
        # premultiplied interpolation (CSS Color 5): transparent as B gives A at alpha p
        alpha = a[3] * p + b[3] * (1 - p)
        if alpha == 0:
            return (0.0, 0.0, 0.0, 0.0)
        mixed = [(a[i] * a[3] * p + b[i] * b[3] * (1 - p)) / alpha for i in range(3)]
        return (mixed[0], mixed[1], mixed[2], alpha)
    return parse_color(v)


def luminance(rgb: tuple[float, float, float]) -> float:
    def lin(c):
        c = c / 255
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    r, g, b = rgb
    return 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b)


def contrast(a, b, over=None, table: dict[str, str] | None = None) -> float:
    """WCAG contrast ratio of two colours (strings or RGB tuples); translucent ones go over `over`."""
    ca = a if isinstance(a, tuple) else resolve_color(a, table, over)
    cb = b if isinstance(b, tuple) else resolve_color(b, table, over)
    la, lb = luminance(ca[:3]), luminance(cb[:3])
    hi, lo = max(la, lb), min(la, lb)
    return (hi + 0.05) / (lo + 0.05)


# ───────────────────────────── JavaScript ─────────────────────────────


def js_files() -> list[Path]:
    """Every module under src/static/js."""
    return sorted(JS_ROOT.rglob("*.js")) if JS_ROOT.exists() else []


@dataclass
class JsSource:
    path: Path
    text: str
    code: str                       # comments and literal contents blanked (same length; newlines kept)
    no_comments: str                # comments blanked, literals kept
    strings: list[tuple[int, str]]  # (line, value) of '…' and "…" literals
    templates: list[tuple[int, str]]  # (line, static text) of `…` literals (the ${} parts left out)

    @property
    def rel(self) -> str:
        return self.path.relative_to(REPO).as_posix()

    def line_of(self, index: int) -> int:
        return self.text.count("\n", 0, index) + 1


_REGEX_PREV = set("(,=:[!&|?{};+-*%<>~^")
_REGEX_KEYWORDS = {"return", "typeof", "instanceof", "in", "of", "new", "delete", "void", "throw", "case", "do", "else", "yield", "await"}


def lex_js(path: Path) -> JsSource:
    """A small JS lexer: enough to separate code, comments, strings, templates and regex literals."""
    text = path.read_text(encoding="utf-8")
    n = len(text)
    code = list(text)
    no_comments = list(text)
    strings: list[tuple[int, str]] = []
    templates: list[tuple[int, str]] = []

    def blank(buf, start, end):
        for k in range(start, end):
            if buf[k] != "\n":
                buf[k] = " "

    def line(idx):
        return text.count("\n", 0, idx) + 1

    def prev_significant(idx):
        j = idx - 1
        while j >= 0 and code[j].isspace():
            j -= 1
        if j < 0:
            return "", ""
        if re.match(r"[\w$]", code[j]):
            k = j
            while k >= 0 and re.match(r"[\w$]", code[k]):
                k -= 1
            return "word", "".join(code[k + 1:j + 1])
        return "punct", code[j]

    def read_template(start):
        """start at the backtick; blanks the static text (not the ${} expressions); returns the end index."""
        i = start + 1
        static = []
        seg_start = i
        while i < n:
            c = text[i]
            if c == "\\":
                static.append(text[i:i + 2])
                i += 2
                continue
            if c == "`":
                blank(code, seg_start, i)
                templates.append((line(start), "".join(static)))
                return i + 1
            if c == "$" and i + 1 < n and text[i + 1] == "{":
                blank(code, seg_start, i)
                static.append("\x00")
                depth = 1
                j = i + 2
                while j < n and depth:
                    ch = text[j]
                    if ch in "'\"":
                        j = read_string(j)
                        continue
                    if ch == "`":
                        j = read_template(j)
                        continue
                    if ch == "{":
                        depth += 1
                    elif ch == "}":
                        depth -= 1
                    j += 1
                i = j
                seg_start = i
                continue
            static.append(c)
            i += 1
        blank(code, seg_start, n)
        templates.append((line(start), "".join(static)))
        return n

    def read_string(start):
        q = text[start]
        i = start + 1
        value = []
        while i < n:
            c = text[i]
            if c == "\\":
                value.append(text[i:i + 2])
                i += 2
                continue
            if c == q or c == "\n":
                break
            value.append(c)
            i += 1
        strings.append((line(start), "".join(value)))
        blank(code, start + 1, i)
        return i + 1

    i = 0
    while i < n:
        c = text[i]
        if c == "/" and i + 1 < n and text[i + 1] == "/":
            end = text.find("\n", i)
            end = n if end < 0 else end
            blank(code, i, end)
            blank(no_comments, i, end)
            i = end
            continue
        if c == "/" and i + 1 < n and text[i + 1] == "*":
            end = text.find("*/", i + 2)
            end = n if end < 0 else end + 2
            blank(code, i, end)
            blank(no_comments, i, end)
            i = end
            continue
        if c in "'\"":
            i = read_string(i)
            continue
        if c == "`":
            i = read_template(i)
            continue
        if c == "/":
            kind, tok = prev_significant(i)
            if kind == "" or (kind == "punct" and tok in _REGEX_PREV) or (kind == "word" and tok in _REGEX_KEYWORDS):
                j = i + 1
                in_class = False
                while j < n and text[j] != "\n":
                    ch = text[j]
                    if ch == "\\":
                        j += 2
                        continue
                    if ch == "[":
                        in_class = True
                    elif ch == "]":
                        in_class = False
                    elif ch == "/" and not in_class:
                        break
                    j += 1
                blank(code, i + 1, j)
                i = j + 1
                continue
        i += 1
    return JsSource(path, text, "".join(code), "".join(no_comments), strings, templates)


def static_imports(path: Path) -> list[str]:
    """The static import/export-from specifiers of a module (dynamic import() excluded)."""
    src = lex_js(path).no_comments
    specs = re.findall(r"""(?:^|[;\n}])\s*(?:import|export)\s+(?:[\w*{}\s,$]+?\s+from\s+)?(['"])([^'"]+)\1""", src)
    return [s for _, s in specs]


def static_import_closure(entry: Path) -> list[Path]:
    """Every module reachable from `entry` through static imports (entry excluded), sorted."""
    seen: set[Path] = set()
    stack = [entry.resolve()]
    while stack:
        current = stack.pop()
        for spec in static_imports(current):
            if not spec.startswith("."):
                continue
            target = (current.parent / spec).resolve()
            if target not in seen and target.exists():
                seen.add(target)
                stack.append(target)
    seen.discard(entry.resolve())
    return sorted(seen)
