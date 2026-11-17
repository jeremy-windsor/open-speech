"""The switch to the new web UI: /web serves the new shell, everything the shell names
resolves with its content type, the old UI is gone, and no file still names it as a path.

Route tests use the real app through TestClient (inside pytest only); everything else reads files.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tests import ui_support as ui

# The old UI's global scripts and stylesheets (deleted), and the new shell's preview name.
OLD_UI_FILES = ("app.js", "app.css", "reader.js", "reader.css")
PREVIEW_SHELL = "next.html"
THIS_FILE = Path(__file__).resolve()

TYPES = {".js": "text/javascript", ".css": "text/css", ".svg": "image/svg+xml", ".webmanifest": "application/manifest+json",
         ".png": "image/png", ".html": "text/html"}


@pytest.fixture(scope="module")
def client():
    from src.main import app

    return TestClient(app)


@pytest.fixture(scope="module")
def web(client):
    page = client.get("/web")
    assert page.status_code == 200
    return page


def _shell_urls(markup: str) -> tuple[list[str], set[str]]:
    """(every /static/ href and src of the served shell except sprite <use> references, the sprite files <use> names)."""
    tree = ui.parse_html(markup)
    urls, sprites = set(), set()
    for el in tree.elements():
        for attr in ("href", "src"):
            value = el.get(attr) or ""
            if not value.startswith("/static/"):
                continue
            if el.tag == "use":
                sprites.add(value.split("#", 1)[0])
            else:
                urls.add(value)
    return sorted(urls), sprites


def _view_partials() -> list[str]:
    """Every partial path the view modules name: their `partial` fields and the slot partials they fetch."""
    paths = set()
    for path in ui.js_files():
        text = path.read_text(encoding="utf-8")
        if path.name == "index.js" and path.parent.parent.name == "views":
            fields = re.findall(r"partial:\s*'(/static/partials/[\w-]+\.html)'", text)
            assert fields, f"{path} has no partial field"
            paths.update(fields)
        paths.update(re.findall(r"['\"`](/static/partials/[\w-]+\.html)['\"`]", text))
    return sorted(paths)


# ───────────────────────────── /web ─────────────────────────────


def test_web_serves_the_new_shell_revalidated_under_the_web_ui_csp(web):
    """[GS-01] /web answers 200 with Cache-Control: no-cache, the web-UI CSP and the module entry point: the
    shell that was /static/next.html, read from disk as src/static/index.html."""
    from src.middleware import WEB_UI_CSP

    assert web.headers["content-type"].startswith("text/html")
    assert web.headers["cache-control"] == "no-cache"
    assert web.headers["content-security-policy"] == WEB_UI_CSP
    version = re.search(r'<script type="module" src="/static/v/(\w+)/js/main\.js"></script>', web.text).group(1)
    shell = ui.shell_path().read_text(encoding="utf-8")
    assert web.text == shell.replace('"/static/js/', f'"/static/v/{version}/js/'), "scripts under a URL per release"
    assert ui.SHELL_PATH == "src/static/index.html"


def test_every_asset_partial_and_icon_the_shell_names_resolves_with_its_type(client, web):
    """[GS-01] Every href/src in the served shell, every partial path the view modules name, the sprite, and the
    manifest's icons answer 200 with the right content type and Cache-Control: no-cache."""
    urls, sprites = _shell_urls(web.text)
    partials = _view_partials()
    manifest = json.loads((ui.STATIC / "manifest.webmanifest").read_text(encoding="utf-8"))
    icons = [icon["src"] for icon in manifest["icons"]]
    assert any(re.fullmatch(r"/static/v/\w+/js/main\.js", url) for url in urls)
    assert "/static/manifest.webmanifest" in urls
    assert sprites == {"/static/icons/sprite.svg"}
    assert len(partials) == len(ui.VIEW_PARTIALS) + len(ui.SLOT_PARTIALS)
    assert {f"/static/partials/{name}.html" for name in (*ui.VIEW_PARTIALS, *ui.SLOT_PARTIALS)} == set(partials)

    for path in sorted({*urls, *partials, *sprites, *icons}):
        res = client.get(path)
        assert res.status_code == 200, path
        suffix = "." + path.rsplit(".", 1)[-1]
        assert res.headers["content-type"].split(";")[0] == TYPES[suffix], (path, res.headers["content-type"])
        assert res.headers["cache-control"] == "no-cache", path


# ───────────────────────────── the old UI is gone ─────────────────────────────


def test_the_old_ui_and_the_preview_shell_are_gone(client):
    """[BUG-25] The swap deletes the old UI instead of keeping dead code and markup alive for its tests: its
    scripts and stylesheets, and the preview /static/next.html, are no longer files and answer 404."""
    for name in (*OLD_UI_FILES, PREVIEW_SHELL):
        assert not (ui.STATIC / name).exists(), name
        assert client.get(f"/static/{name}").status_code == 404, name
    assert (ui.STATIC / "index.html").is_file()


# The scan: a mention of an old file "as a path". Prose ("ported from the old reader.js") is not a path, and
# css/views/reader.css (the new Reader stylesheet) is a different file.
_NAMES = r"(?:app|reader)\.(?:js|css)|next\.html"
_CANDIDATE = re.compile(rf"(?<![\w-])(?P<name>{_NAMES})(?![\w-])")


def path_mentions(text: str) -> list[str]:
    """The old UI files `text` names as a path: under the static root (/static/app.js, src/static/reader.css),
    as a relative path (./app.js, ../reader.js), or as a whole string literal ('app.css', STATIC / "reader.js")."""
    found = []
    for m in _CANDIDATE.finditer(text):
        before = text[max(0, m.start() - 40):m.start()]
        after = text[m.end():m.end() + 1]
        if re.search(r"(?:^|[^\w-])static/$", before):
            found.append(f"static/{m['name']}")
        elif re.search(r"views/$|views[\"']\s*/\s*[\"']$", before):
            continue                                   # css/views/reader.css, the new stylesheet
        elif re.search(r"(?:^|[^\w.-])\.\.?/(?:\.\./)*$", before):
            found.append(f"relative {m['name']}")
        elif before[-1:] in ("'", '"', "`") and after == before[-1:]:
            found.append(f"literal {m['name']}")
    return found


def scanned_files() -> list[Path]:
    """Every text file under src/static and tests, except this file, which names the old files to check they are
    gone."""
    out = []
    for root in (ui.STATIC, ui.REPO / "tests"):
        for path in sorted(root.rglob("*")):
            if not path.is_file() or "__pycache__" in path.parts or path.suffix in {".png", ".pyc", ".woff2"}:
                continue
            if path.resolve() == THIS_FILE:
                continue
            out.append(path)
    return out


def test_no_file_names_the_old_ui_as_a_path():
    """[BUG-25][REV-28] No file under src/static, and no test file, names app.js,
    reader.js, app.css, reader.css or next.html as a path: nothing still loads, slices or serves the old UI."""
    files = scanned_files()
    assert ui.shell_path() in files and (ui.REPO / "tests" / "test_api.py") in files
    problems = {}
    for path in files:
        hits = path_mentions(path.read_text(encoding="utf-8", errors="replace"))
        if hits:
            problems[path.relative_to(ui.REPO).as_posix()] = hits
    assert problems == {}


def test_the_old_file_scan_reads_paths_not_prose():
    """[REV-28] The scan covers every test file and reports path forms only: a "src/static/app.js" comment
    would count, prose about the old reader.js and the new css/views/reader.css do not."""
    flagged = {
        '<script src="/static/app.js" defer></script>': ["static/app.js"],
        '<link rel="stylesheet" href="/static/reader.css">': ["static/reader.css"],
        "# Contract: src/static/app.js ensureModelReady shows ...": ["static/app.js"],
        'APP_JS = Path("src/static/app.js").read_text()': ["static/app.js"],
        'LEGACY = ("index.html", "app.js", "reader.css")': ["literal app.js", "literal reader.css"],
        'script = STATIC / "reader.js"': ["literal reader.js"],
        "import { byId } from '../../app.js';": ["relative app.js"],
        "page = client.get('/static/next.html')": ["static/next.html"],
        "loadHtml('src/static/next.html')": ["static/next.html"],
    }
    for text, expected in flagged.items():
        assert path_mentions(text) == expected, text
    clean = (
        '<link rel="stylesheet" href="/static/css/views/reader.css">',
        'CSS = ui.CSS_ROOT / "views" / "reader.css"',
        "the pure helpers, ported from the old reader.js with its fixes",
        "reader.css hides the top bar's button for it",
        "assert.equal(reading.getAttribute('data-rd-transport'), 'ready', 'reader.css decides by width')",
        'import "./core/api.js"; const mapp = "mapp.js"; const appjs = "app.json";',
    )
    for text in clean:
        assert path_mentions(text) == [], text

    files = scanned_files()
    tests_dir = ui.REPO / "tests"
    expected = {p for p in tests_dir.glob("test_*.py") if p.resolve() != THIS_FILE}
    assert expected and expected <= set(files)
    assert any(p.suffix == ".js" and p.is_relative_to(tests_dir / "js") for p in files)
    assert ui.REPO / "tests" / "ui_must_preserve.json" in files
