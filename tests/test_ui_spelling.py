"""US spelling in everything a person reads (for example "favorite", not "favourite").

These tests read files only. They look for British spellings in:

- the shown text of the web app: the text nodes of src/static/index.html and src/static/partials/*.html, and their
  placeholder, title, aria-label and alt attributes;
- the string literals of src/static/js/ (comments are skipped);
- the prose of README.md, docs/*.md, src/static/README.md, client-js/README.md and CHANGELOG.md (fenced code, code
  spans, link targets and HTML comments are skipped), which is every tracked Markdown file (outside the internal
  notes folder review/, if present);
- the server's messages: the string literals of src/**/*.py, except docstrings and log lines.

Not prose, so allowed: names in code (identifiers such as `row.favourite` or
`normaliseKeyName`, which a word match can't see inside, and CSS classes or keys joined to another word by `-`, `.`,
`:` or `/`), anything stored in the browser or on the server, the attribute `aria-labelledby`, and protocol and API
values such as the job status `cancelled`, the event `response.cancelled` and the reason `client_cancelled`
(OpenAI's spellings), which the docs quote as code. "Cancellation" is US spelling too.
"""

from __future__ import annotations

import ast
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tests import ui_support as ui

REPO = ui.REPO

# British spellings and their forms. Each alternative is a whole word (case-insensitive).
BRITISH_WORDS = (
    r"favourit(?:e|es|ed|ing|ism)",
    r"(?:un)?favour(?:s|ed|ing|able|ably)?",
    r"colour(?:s|ed|ing|ful|less|ise|ised|ize|ized|ization)?",
    r"behaviour(?:s|al|ally)?",
    r"(?:dis)?honour(?:s|ed|ing|able)?",
    r"cent(?:re|res|red|ring)",
    r"neighbour(?:s|ing|hood|hoods|ly)?",
    r"catalogu(?:e|es|ed|ing)",
    r"grey(?:s|ed|er|est|ish|ness|scale|scaled)?",
    r"cancell(?:ed|ing|er|ers)",
    r"labell(?:ed|ing|er|ers)",
    r"(?:re|dis)?organis(?:e|es|ed|ing|er|ers|ation|ations|ational)",
    r"recognis(?:e|es|ed|ing|able|ably)",
    r"customis(?:e|es|ed|ing|able|ation|ations)",
    r"normalis(?:e|es|ed|ing|er|ation)",
    r"summaris(?:e|es|ed|ing|er|ation|ations)",
    r"optimis(?:e|es|ed|ing|er|ers|ation|ations)",
    r"(?:re)?initialis(?:e|es|ed|ing|er|ation)",
    r"(?:re)?analys(?:e|ed|ing|er|ers)",
    r"licences?",
    r"modell(?:ed|ing|er|ers)",
    r"travell(?:ed|ing|er|ers)",
    r"programmes?",
    r"judgements?",
    r"whilst",
    r"amongst",
    r"flavour(?:s|ed|ing|ful|less)?",
    # The rest of the usual -ise/-yse, -our, -re, -ce and doubled-l forms.
    r"realis(?:e|es|ed|ing|ation)",
    r"(?:de)?prioritis(?:e|es|ed|ing|ation)",
    r"visualis(?:e|es|ed|ing|er|ers|ation|ations)",
    r"(?:de)?serialis(?:e|es|ed|ing|er|ers|ation)",
    r"(?:un|re)?authoris(?:e|es|ed|ing|ation|ations)",
    r"minimis(?:e|es|ed|ing)",
    r"maximis(?:e|es|ed|ing)",
    r"finalis(?:e|es|ed|ing)",
    r"utilis(?:e|es|ed|ing|ation)",
    r"(?:de|re)?synchronis(?:e|es|ed|ing|ation)",
    r"standardis(?:e|es|ed|ing|ation)",
    r"categoris(?:e|es|ed|ing|ation)",
    r"personalis(?:e|es|ed|ing|ation)",
    r"specialis(?:e|es|ed|ing|ation)",
    r"localis(?:e|es|ed|ing|ation)",
    r"sanitis(?:e|es|ed|ing|er|ation)",
    r"tokenis(?:e|es|ed|ing|er|ers|ation)",
    r"quantis(?:e|es|ed|ing|ation)",
    r"capitalis(?:e|es|ed|ing|ation)",
    r"emphasis(?:e|es|ed|ing)",
    r"apologis(?:e|es|ed|ing)",
    r"memoris(?:e|es|ed|ing)",
    r"stabilis(?:e|es|ed|ing|er|ers|ation)",
    r"(?:para|cata)lys(?:e|ed|ing)",
    r"(?:labour|humour|rumour|harbour|vapour|armour|odour|endeavour|rigour|vigour|parlour|savour)(?:s|ed|ing)?",
    r"(?:kilo|centi|milli)?metres?",
    r"litres?",
    r"fibres?",
    r"theatres?",
    r"(?:calibre|spectre|lustre|sombre|meagre|sabre)s?",
    r"defences?",
    r"offences?",
    r"pretences?",
    r"practis(?:e|es|ed|ing)",
    r"enrolment|enrolments|fulfil|fulfils|fulfilment|skilful|skilfully|wilful|wilfully|instalments?",
    r"artefacts?",
    r"sceptic(?:al|ally|ism)?",
    r"ageing",
    r"learnt",
    r"focuss(?:ed|es|ing)",
    r"benefitt(?:ed|ing)",
    r"(?:level|signal|fuel|total|channel|tunnel|dial|funnel|unravel|rival|spiral|equal)l(?:ed|ing)",
    r"marvellous(?:ly)?",
    r"counsell(?:or|ors|ed|ing)",
    r"jewellery",
    r"cheques?",
    r"tyres?",
    r"aluminium",
    r"manoeuvr(?:e|es|ed|ing)",
)
BRITISH = re.compile(r"\b(?:" + "|".join(BRITISH_WORDS) + r")\b", re.IGNORECASE)

# A word joined to another by one of these is a name in code (a CSS class, a key, an event, a path), not prose:
# `vp-favourite`, `response.cancelled`, `voices/favourites`.
JOINERS = "-.:/#@$"

# Whole JS or Python string literals that are protocol, API or stored values, allowed as they are.
CODE_VALUES = {
    "cancelled",   # the job status (assistant jobs, batch, Live Reader), OpenAI's spelling
}


def _british(text: str) -> list[re.Match]:
    return list(BRITISH.finditer(text))


def _joined(text: str, m: re.Match) -> bool:
    """True when the word is part of a name in code: joined to another word by one of JOINERS, or a protocol
    value quoted inside the literal (SQL's `status IN ('queued', …, 'cancelled')`)."""
    before, after = text[:m.start()], text[m.end():]
    if m.group(0) in CODE_VALUES and before[-1:] in ("'", '"') and after[:1] == before[-1:]:
        return True
    if len(before) >= 2 and before[-1] in JOINERS and (before[-2].isalnum() or before[-2] in "_-"):
        return True
    if len(after) >= 2 and after[0] in JOINERS and (after[1].isalnum() or after[1] == "_"):
        return True
    return False


def _context(text: str, m: re.Match) -> str:
    start, end = max(0, m.start() - 30), min(len(text), m.end() + 30)
    return re.sub(r"\s+", " ", text[start:end]).strip()


# ───────────────────────────── the web app's markup ─────────────────────────────

SHOWN_ATTRS = ("placeholder", "title", "aria-label", "alt")


def _markup_texts():
    """(where, text) for every text node and shown attribute of the shell and the partials."""
    for name, tree in ui.all_markup():
        source = "src/static/index.html" if name == "shell" else f"src/static/partials/{name}.html"
        stack = [tree]
        while stack:
            node = stack.pop()
            if node.tag in ("script", "style"):
                continue
            if node.tag == "#text":
                yield f"{source}:{node.line}", node.text
                continue
            for attr in SHOWN_ATTRS:
                value = node.attrs.get(attr)
                if value:
                    yield f"{source}:{node.line} [{attr}]", value
            stack.extend(reversed(node.children))


def test_shown_markup_uses_us_spelling():
    """The text nodes and placeholder, title, aria-label and alt attributes of index.html and the partials."""
    found = [f"{where}: {m.group(0)!r} in {_context(text, m)!r}"
             for where, text in _markup_texts() for m in _british(text)]
    assert found == [], "British spellings in shown text:\n" + "\n".join(found)


# ───────────────────────────── the web app's JS string literals ─────────────────────────────

_ESCAPES = {"n": " ", "r": " ", "t": " ", "v": " ", "b": " ", "f": " ", "0": " "}


def _decode_js(raw: str) -> str:
    """A literal's text with its escapes resolved (\\n as a space), so a word after one is still a word."""
    def one(m):
        body = m.group(1)
        if body[0] == "u":
            hexa = body[1:].strip("{}")
            try:
                return chr(int(hexa, 16))
            except ValueError:
                return " "
        if body[0] == "x":
            try:
                return chr(int(body[1:], 16))
            except ValueError:
                return " "
        return _ESCAPES.get(body, body)
    return re.sub(r"\\(u\{[0-9A-Fa-f]+\}|u[0-9A-Fa-f]{4}|x[0-9A-Fa-f]{2}|.)", one, raw, flags=re.S)


def _js_literals():
    """(where, text) for every '…', "…" and `…` literal under src/static/js (a template's ${} parts as \\x00)."""
    for path in ui.js_files():
        src = ui.lex_js(path)
        rel = path.relative_to(REPO).as_posix()
        for line, raw in src.strings + src.templates:
            yield f"{rel}:{line}", _decode_js(raw)


def _code_literal(text: str) -> bool:
    return text in CODE_VALUES


def test_js_string_literals_use_us_spelling():
    """The string literals of src/static/js: shown text, messages and labels; names in code are allowed."""
    found = []
    for where, text in _js_literals():
        if _code_literal(text):
            continue
        for m in _british(text):
            if not _joined(text, m):
                found.append(f"{where}: {m.group(0)!r} in {_context(text, m)!r}")
    assert found == [], "British spellings in the web app's strings:\n" + "\n".join(found)


# ───────────────────────────── README.md and docs/*.md ─────────────────────────────


def _blank(m: re.Match) -> str:
    """The match blanked, its newlines kept, so line numbers stay right."""
    return re.sub(r"[^\n]", " ", m.group(0))


# Where a new block starts on the next line: a blank line, a list item, a heading, a block quote or a table row. A code
# span ends with its block (CommonMark), so an unpaired backtick in one list item can't pair the rest of the list
# the wrong way round.
_BLOCK_START = r"\n[ \t]*(?:\n|[-*+][ \t]|\d{1,9}[.)][ \t]|#{1,6}[ \t]|>|\|)"


def _prose(text: str) -> str:
    """A Markdown file with what isn't prose blanked: fenced code, HTML comments, code spans, link targets."""
    text = re.sub(r"^[ \t]*(`{3,}|~{3,}).*?^[ \t]*\1[ \t]*$", _blank, text, flags=re.S | re.M)
    text = re.sub(r"<!--.*?-->", _blank, text, flags=re.S)
    # A code span: a run of backticks, then text inside one block, then the same run.
    text = re.sub(r"(?<!`)(`+)(?!`)(?:(?!" + _BLOCK_START + r").)+?(?<!`)\1(?!`)", _blank, text, flags=re.S)
    text = re.sub(r"\]\([^)\s]*(?:\s+\"[^\"]*\")?\)", _blank, text)       # [text](target "title")
    text = re.sub(r"^[ \t]*\[[^\]]+\]:[ \t]*\S+.*$", _blank, text, flags=re.M)  # [ref]: target
    text = re.sub(r"<https?://[^>]+>|https?://\S+", _blank, text)
    text = re.sub(r"</?[A-Za-z][^>]*>", _blank, text)                     # inline HTML tags and attributes
    return text


def _doc_files() -> list[Path]:
    """README.md, docs/*.md, the web app's own README, the JS SDK's README (client-js/README.md) and
    CHANGELOG.md."""
    return [REPO / "README.md", *sorted((REPO / "docs").glob("*.md")), ui.STATIC / "README.md",
            REPO / "client-js" / "README.md", REPO / "CHANGELOG.md"]


def test_readme_and_docs_prose_uses_us_spelling():
    """The prose of README.md, docs/*.md, src/static/README.md, client-js/README.md and CHANGELOG.md; code spans and
    fenced code (API values, names in code) are skipped. client-js/README.md and
    CHANGELOG.md were left out, and "behaviour" got into the SDK README unnoticed."""
    found = []
    for path in _doc_files():
        prose = _prose(path.read_text(encoding="utf-8"))
        rel = path.relative_to(REPO).as_posix()
        for m in _british(prose):
            line = prose.count("\n", 0, m.start()) + 1
            found.append(f"{rel}:{line}: {m.group(0)!r} in {_context(prose, m)!r}")
    assert found == [], ("British spellings in the prose of README.md, docs/*.md, src/static/README.md, "
                         "client-js/README.md and CHANGELOG.md:\n" + "\n".join(found))


def test_prose_guard_reads_every_tracked_markdown_file():
    """Every Markdown file git tracks (outside the internal notes folder review/, if present) is a doc
    a person reads, so the prose test reads it: a new doc can't slip past the guard the way client-js/README.md
    did.

    It skips only when there is no checkout (no .git) or no git, and only locally, as
    tests/ui_support.require_node does; a git that fails in a checkout fails the test with its exit code and stderr."""
    on_ci = bool(os.environ.get("CI"))
    if not (REPO / ".git").exists():
        if on_ci:
            pytest.fail(f"a git checkout is required on CI to list the tracked Markdown files: no .git in {REPO}")
        pytest.skip(f"no git checkout to list tracked files in: no .git in {REPO}")
    try:
        listed = subprocess.run(["git", "ls-files", "-z", "--", "*.md"], cwd=REPO, capture_output=True,
                                timeout=60, check=True).stdout.decode("utf-8")
    except FileNotFoundError as exc:
        if on_ci:
            pytest.fail(f"git is required on CI to list the tracked Markdown files: {exc}")
        pytest.skip(f"git is not installed: {exc}")
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or b"").decode("utf-8", "replace").strip()
        pytest.fail(f"git ls-files failed in this checkout (exit status {exc.returncode}): {stderr or 'no stderr'}")
    tracked = {name for name in listed.split("\0") if name and not name.startswith("review/")}
    read = {path.relative_to(REPO).as_posix() for path in _doc_files()}
    assert "client-js/README.md" in tracked and "CHANGELOG.md" in tracked
    assert sorted(tracked - read) == [], "tracked Markdown files the prose test doesn't read"


# The guard above turned every git error into "no git checkout" and skipped, in a real
# checkout and on CI alike (a full run once skipped it when git couldn't start, exit 0xC0000142). It may skip
# only when there is no checkout (no .git) or no git, and only locally, as tests/ui_support.require_node does.

def _git_run(returncode, stderr=b""):
    """A subprocess.run for git that exits with returncode, as the real one does with or without check=True."""
    def run(argv, **kwargs):
        if returncode and kwargs.get("check"):
            raise subprocess.CalledProcessError(returncode, argv, output=b"", stderr=stderr)
        return subprocess.CompletedProcess(argv, returncode, stdout=b"", stderr=stderr)

    return run


def _guard_outcome() -> str:
    """Run the guard and say how it ended: 'passed', 'skipped: <reason>' or 'failed: <message>'. Its skip must not
    become this test's skip, or a guard that wrongly skips would pass here unseen."""
    try:
        test_prose_guard_reads_every_tracked_markdown_file()
    except pytest.skip.Exception as exc:
        return f"skipped: {exc}"
    except pytest.fail.Exception as exc:  # Failed is a BaseException, not an Exception
        return f"failed: {exc}"
    return "passed"


@pytest.fixture
def _checkout(monkeypatch, tmp_path):
    """A checkout (a folder with .git) for the guard to run git in, run locally (CI unset)."""
    (tmp_path / ".git").mkdir()
    monkeypatch.setattr(sys.modules[__name__], "REPO", tmp_path)
    monkeypatch.delenv("CI", raising=False)
    return tmp_path


@pytest.mark.parametrize(("returncode", "stderr"), [
    (3221225794, b""),  # 0xC0000142 STATUS_DLL_INIT_FAILED: git could not start (seen in a full run)
    (128, b"fatal: detected dubious ownership in repository at '/src'\n"),  # safe.directory, another user
], ids=["git-could-not-start", "dubious-ownership"])
def test_prose_guard_fails_when_git_fails_in_a_checkout(monkeypatch, _checkout, returncode, stderr):
    """A git that fails in a checkout fails the guard with its exit code and stderr; it
    never skips it as "no git checkout"."""
    monkeypatch.setattr(subprocess, "run", _git_run(returncode, stderr))
    outcome = _guard_outcome()
    assert outcome.startswith("failed: "), outcome
    assert str(returncode) in outcome
    if stderr:
        assert "dubious ownership" in outcome


def test_prose_guard_fails_when_real_git_finds_no_repository(monkeypatch, _checkout):
    """Real git in a broken environment (GIT_DIR names nothing, exit 128) fails."""
    if shutil.which("git") is None:
        pytest.skip("git is not installed")
    monkeypatch.setenv("GIT_DIR", str(_checkout / "not-a-repository"))
    outcome = _guard_outcome()
    assert outcome.startswith("failed: ") and "128" in outcome, outcome


def test_prose_guard_skips_without_git_or_a_checkout_only_locally(monkeypatch, _checkout, tmp_path_factory):
    """No git installed, or no .git folder, is a local skip and a failure on CI."""
    def missing(argv, **kwargs):
        raise FileNotFoundError(2, "The system cannot find the file specified", "git")

    with monkeypatch.context() as m:
        m.setattr(subprocess, "run", missing)
        assert _guard_outcome().startswith("skipped: git is not installed")
        m.setenv("CI", "true")
        assert _guard_outcome().startswith("failed: git is required on CI")
    with monkeypatch.context() as m:
        m.setattr(subprocess, "run", _git_run(128, b"fatal: not a git repository\n"))
        m.setattr(sys.modules[__name__], "REPO", tmp_path_factory.mktemp("copy"))  # a copy of the tree, no .git
        assert _guard_outcome().startswith("skipped: no git checkout")
        m.setenv("CI", "true")
        assert _guard_outcome().startswith("failed: a git checkout is required on CI")


def test_prose_guard_passes_with_a_working_git():
    """A control: in this checkout, with git working, the guard runs and passes."""
    if not (REPO / ".git").exists() or shutil.which("git") is None:
        pytest.skip("no git checkout, or no git")
    assert _guard_outcome() == "passed"


def test_prose_pairs_code_spans_within_one_block():
    """An unpaired backtick run in one list item (`` `dom.html``` `` in
    src/static/README.md) paired every later span of the list the wrong way round, so the prose of the next items was
    skipped and their code read as prose. Each list item, heading, quote and table row pairs its own spans."""
    text = (
        "- Built with `dom.html```, which escapes; `raw()` is for partials.\n"
        "- A `.check` (a labelled row) and `forced-colors` rules.\n"
        "  Second line of the same item: `a` and `b`.\n"
        "# A heading with `code`\n"
        "| `cell` | a colour |\n"
    )
    prose = _prose(text)
    assert len(prose) == len(text)
    lines = prose.split("\n")
    assert "labelled" in lines[1] and ".check" not in lines[1] and "forced-colors" not in lines[1]
    assert "Second line of the same item" in lines[2] and "`" not in lines[2]
    assert "A heading with" in lines[3] and "code" not in lines[3]
    assert "colour" in lines[4] and "cell" not in lines[4]
    assert [m.group(0) for m in _british(prose)] == ["labelled", "colour"]


def test_readme_and_docs_examples_use_us_spelling():
    """What the prose test skips in README.md and docs/*.md (code spans, fenced
    code, link targets, HTML comments) holds example sentences too, such as a prompt or a request body, and a person
    reads those: an ordinary word there is US spelling. Names in code (joined to another word) and protocol values
    such as `cancelled` are allowed."""
    found = []
    # client-js/README.md too. CHANGELOG.md is left out: it may quote old strings.
    for path in [REPO / "README.md", *sorted((REPO / "docs").glob("*.md")), REPO / "client-js" / "README.md"]:
        text = path.read_text(encoding="utf-8")
        prose = _prose(text)
        assert len(prose) == len(text), f"_prose changed the length of {path.name}"
        rel = path.relative_to(REPO).as_posix()
        for m in _british(text):
            if prose[m.start():m.end()].strip():
                continue            # prose: the test above reports it
            if m.group(0) in CODE_VALUES or _joined(text, m):
                continue
            line = text.count("\n", 0, m.start()) + 1
            found.append(f"{rel}:{line}: {m.group(0)!r} in {_context(text, m)!r}")
    assert found == [], "British spellings in README.md and docs/*.md examples:\n" + "\n".join(found)


# ───────────────────────────── the server's messages ─────────────────────────────

_LOG_METHODS = {"debug", "info", "warning", "warn", "error", "exception", "critical", "log"}


def _docstring_nodes(tree: ast.AST) -> set[int]:
    ids = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = getattr(node, "body", [])
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
                    and isinstance(body[0].value.value, str):
                ids.add(id(body[0].value))
    return ids


def _log_call_constants(tree: ast.AST) -> set[int]:
    """The string constants inside logger.info(…), log.warning(…), logging.error(…) and the like."""
    ids = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in _LOG_METHODS:
            target = node.func.value
            name = target.id if isinstance(target, ast.Name) else getattr(target, "attr", "")
            if "log" in name.lower():
                for sub in ast.walk(node):
                    if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
                        ids.add(id(sub))
    return ids


def _server_strings():
    for path in sorted((REPO / "src").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        skip = _docstring_nodes(tree) | _log_call_constants(tree)
        rel = path.relative_to(REPO).as_posix()
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in skip:
                yield f"{rel}:{node.lineno}", node.value


def test_server_messages_use_us_spelling():
    """The server's string literals (what a response or the web app can show); docstrings and log lines aside."""
    found = []
    for where, text in _server_strings():
        if _code_literal(text):
            continue
        for m in _british(text):
            if not _joined(text, m):
                found.append(f"{where}: {m.group(0)!r} in {_context(text, m)!r}")
    assert found == [], "British spellings in the server's messages:\n" + "\n".join(found)
