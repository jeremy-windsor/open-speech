"""Web UI: the core JS behaviour tests and the global audits every UI file must pass.

The audits run over the shell, every partial, every CSS file and every JS module, so every view's files are
checked. Each audit is a function that returns its problems; a test runs it on the real files (expecting none)
and on a bad fixture (proving it catches the problem). The CSS audits (tokens, contrast, motion, forced
colours) are in the section marked for them below.

Nothing here imports src.main: files are read as text.
"""

from __future__ import annotations

import contextlib
import json
import mimetypes
import re
import subprocess
import sys
from pathlib import Path

import pytest

from tests import ui_support as ui
from tests.ui_coverage import missing

# ───────────────────────────── JS behaviour tests ─────────────────────────────


@pytest.mark.parametrize("path", ui.js_test_files("core"))
def test_js(path):
    ui.run_node_test(path)


def test_f1_inventory_ids_and_defects_have_tests():
    """Every inventory id and defect owned by F1 leads a test title in its assigned file."""
    assert missing("F1") == []


# ───────────────────────────── helpers ─────────────────────────────

URL_ATTRS = ("href", "src", "action", "formaction", "poster", "data", "srcset", "xlink:href", "manifest", "background")
EXTERNAL = re.compile(r"^\s*(?:[a-z][a-z0-9+.-]*:)?//", re.I)
SAFE_EXTERNAL = {"http://www.w3.org/2000/svg", "http://www.w3.org/1999/xlink", "http://www.w3.org/1999/xhtml"}
IDREF_ATTRS = ("aria-controls", "aria-labelledby", "aria-describedby", "aria-owns", "aria-errormessage", "aria-details")
REQUIRED_REGIONS = {
    # The Ask AI sheet's two are in the shell's #app-ask-ai.
    "shell": [("app-sr-status", "status"), ("app-sr-alert", "alert"), ("app-toasts-polite", "status"), ("app-toasts-alert", "alert"), ("app-conn-banner", "status"),
              ("ask-status", "status"), ("ask-error", "alert")],
    "reader": [("rd-opening", "status")],
    "reader-player": [("rdp-status", "status")],
    "speak": [("sp-gen-status", "status")],
    "speak-live": [("spl-status", "status")],
    "transcribe": [("tr-log", "log"), ("tr-upload-status", "status")],
    "talk": [("tk-log", "log"), ("tk-status", "status")],
    "voices": [("vo-status", "status")],
    "studio": [("st-render-status", "status"), ("st-mix-status", "status")],
    "history": [("hi-count", "status")],
    "models": [("mo-summary", "status")],
    "settings": [("se-key-status", "status")],
}


def where(name: str, node: ui.Node) -> str:
    label = node.tag + (f"#{node.id}" if node.id else "")
    return f"{name}:{node.line} <{label}>"


def markup_from(snippets: dict[str, str]) -> list[tuple[str, ui.Node]]:
    return [(name, ui.parse_html(text)) for name, text in snippets.items()]


def real_markup() -> list[tuple[str, ui.Node]]:
    return ui.all_markup()


# ───────────────────────────── HTML audits ─────────────────────────────


def audit_unique_ids(markup):
    seen: dict[str, str] = {}
    problems = []
    for name, tree in markup:
        for el in tree.elements():
            if not el.id:
                continue
            if el.id in seen:
                problems.append(f"{where(name, el)} duplicates id {el.id!r} (first in {seen[el.id]})")
            else:
                seen[el.id] = name
    return problems


def audit_idrefs(markup):
    ids = {el.id for _, tree in markup for el in tree.elements() if el.id}
    problems = []
    for name, tree in markup:
        for el in tree.elements():
            for attr in IDREF_ATTRS:
                for ref in (el.get(attr) or "").split():
                    if ref not in ids:
                        problems.append(f"{where(name, el)} {attr}={ref!r} points to no element")
            if el.tag in ("label", "output") and el.get("for"):
                for ref in el.get("for").split():
                    if ref not in ids:
                        problems.append(f"{where(name, el)} for={ref!r} points to no element")
    return problems


def audit_control_names(markup):
    problems = []
    for name, tree in markup:
        for el in tree.elements():
            if el.tag not in ("input", "select", "textarea"):
                continue
            if el.tag == "input" and el.get("type") == "hidden":
                continue
            if ui.is_aria_hidden(el):
                continue
            if not ui.accessible_name(el, tree):
                problems.append(f"{where(name, el)} has no accessible name (label, aria-label or aria-labelledby)")
    return problems


def audit_buttons(markup):
    problems = []
    for name, tree in markup:
        for el in tree.elements():
            if el.tag != "button" or ui.is_aria_hidden(el):
                continue
            if el.get("type") not in ("button", "submit", "reset"):
                problems.append(f"{where(name, el)} needs an explicit type")
            if not ui.accessible_name(el, tree):
                problems.append(f"{where(name, el)} has no accessible name")
            elif not ui.visible_text(el) and not (el.get("aria-label") or el.get("aria-labelledby")):
                problems.append(f"{where(name, el)} is icon-only without aria-label")
    return problems


def audit_csp_markup(markup):
    problems = []
    for name, tree in markup:
        for el in tree.elements():
            for attr, value in el.attrs.items():
                if re.fullmatch(r"on[a-z]+", attr):
                    problems.append(f"{where(name, el)} has an inline handler {attr}=")
                if attr == "style":
                    problems.append(f"{where(name, el)} has a style= attribute")
                if attr in URL_ATTRS and re.match(r"\s*javascript:", value, re.I):
                    problems.append(f"{where(name, el)} has a javascript: URL")
            if el.tag == "script":
                if name != "shell":
                    problems.append(f"{where(name, el)}: partials hold no <script>")
                    continue
                src = el.get("src") or ""
                if not re.fullmatch(r"/static/js/[\w./-]+\.js", src):
                    problems.append(f"{where(name, el)} script src {src!r} is not under /static/js/")
                if el.text_content().strip():
                    problems.append(f"{where(name, el)} is an inline script")
                if el.get("type") not in (None, "module"):
                    problems.append(f"{where(name, el)} script type {el.get('type')!r}")
    return problems


def audit_external_markup(markup):
    problems = []
    for name, tree in markup:
        for el in tree.elements():
            for attr in URL_ATTRS:
                value = el.get(attr)
                if value and EXTERNAL.match(value) and value.strip() not in SAFE_EXTERNAL:
                    problems.append(f"{where(name, el)} {attr}={value!r} points to another origin")
    return problems


def audit_external_css(files):
    problems = []
    for path in files:
        text = ui._strip_css_comments(Path(path).read_text(encoding="utf-8"))
        for m in re.finditer(r"url\(\s*['\"]?\s*((?:[a-z][a-z0-9+.-]*:)?//[^)'\"]*)", text, re.I):
            problems.append(f"{Path(path).name}: url({m.group(1)}) points to another origin")
        for m in re.finditer(r"@import\s+['\"]\s*((?:[a-z][a-z0-9+.-]*:)?//[^'\")\s;]*)", text, re.I):
            problems.append(f"{Path(path).name}: @import {m.group(1)} points to another origin")
    return problems


def audit_use_refs(markup, sprite_ids):
    problems = []
    for name, tree in markup:
        for el in tree.elements():
            if el.tag != "use":
                continue
            href = el.get("href") or el.get("xlink:href") or ""
            m = re.fullmatch(r"(?:/static/icons/sprite\.svg)?#(i-[\w-]+)", href)
            if not m:
                problems.append(f"{where(name, el)} <use href={href!r}> is not /static/icons/sprite.svg#i-…")
            elif sprite_ids is None:
                problems.append(f"{where(name, el)} uses {m.group(1)} but sprite.svg does not exist")
            elif m.group(1) not in sprite_ids:
                problems.append(f"{where(name, el)} uses {m.group(1)}, which is not in sprite.svg")
    return problems


def audit_file_inputs(markup):
    problems = []
    for name, tree in markup:
        for el in tree.elements():
            if el.tag != "input" or el.get("type") != "file":
                continue
            if el.has("hidden"):
                problems.append(f"{where(name, el)} is hidden; use .visually-hidden so the keyboard reaches it")
            if "visually-hidden" not in el.classes:
                problems.append(f"{where(name, el)} file inputs are .visually-hidden")
            labelled = (el.id and any(lab.get("for") == el.id for _, t in markup for lab in ui.select(t, "label"))) or any(a.tag == "label" for a in el.ancestors())
            if not labelled:
                problems.append(f"{where(name, el)} is not reachable through a <label>")
    return problems


def audit_view_headings(markup):
    problems = []
    for name, tree in markup:
        h1s = ui.select(tree, "h1")
        if name in ui.SLOT_PARTIALS:
            for h in h1s:
                problems.append(f"{where(name, h)}: slot partials have no h1 (their headings start at h2)")
            continue
        if name not in ui.VIEW_PARTIALS:
            continue
        allowed = 2 if name == "reader" else 1
        if not h1s or len(h1s) > allowed:
            problems.append(f"{name}: a view partial has exactly {'one h1' if allowed == 1 else 'two h1s (library and reading)'}, found {len(h1s)}")
        for h in h1s:
            if not h.has("data-view-heading"):
                problems.append(f"{where(name, h)} needs data-view-heading")
            if not h.id.endswith("-title"):
                problems.append(f"{where(name, h)} id must end in -title")
            if h.get("tabindex") != "-1":
                problems.append(f'{where(name, h)} needs tabindex="-1" (focus on route change)')
    return problems


def audit_heading_order(markup):
    problems = []
    for name, tree in markup:
        prev = 1   # views start at h1; the shell's sheets and the slot partials start at h2
        for el in tree.elements():
            m = re.fullmatch(r"h([1-6])", el.tag)
            if not m:
                continue
            level = int(m.group(1))
            if el.tag == "h2" and any(a.tag == "dialog" for a in el.ancestors()):
                prev = 2
                continue
            if level > prev + 1:
                problems.append(f"{where(name, el)} skips from h{prev} to h{level}")
            prev = level
    return problems


def audit_live_regions(markup):
    problems = []
    for name, tree in markup:
        required = REQUIRED_REGIONS.get(name)
        if not required or (name != "shell" and ui.is_stub(tree)):
            continue
        for id_, role in required:
            el = ui.by_id(tree, id_)
            if el is None:
                problems.append(f"{name}: #{id_} (role={role}) is missing")
            elif el.get("role") != role:
                problems.append(f"{where(name, el)} needs role={role}, has {el.get('role')!r}")
        if name == "transcribe":
            log = ui.by_id(tree, "tr-log")
            interim = ui.by_id(tree, "tr-interim")
            if log is not None and log.get("aria-live") != "polite":
                problems.append(f"{where(name, log)} needs aria-live=polite")
            if interim is not None:
                if log is not None and log in list(interim.ancestors()):
                    problems.append(f"{where(name, interim)} must sit outside #tr-log (interim text is never announced)")
                if interim.get("aria-live") != "off":
                    problems.append(f"{where(name, interim)} needs aria-live=off")
        if name == "speak-live":
            now = ui.by_id(tree, "spl-now")
            if now is not None and now.get("aria-live") != "off":
                problems.append(f"{where(name, now)} needs aria-live=off")
        if name == "speak":
            err = ui.by_id(tree, "sp-gen-error")
            if err is not None and err.get("role") != "alert":
                problems.append(f"{where(name, err)} needs role=alert")
    return problems


def audit_stub_markers(markup):
    """A stub marker (data-stub, data-stub-body) left in a partial that holds real markup: it is
    removed when the partial gets its real markup (it would otherwise read as a placeholder)."""
    problems = []
    for name, tree in markup:
        if name == "shell":
            continue
        markers = ui.stub_markers(tree)
        if markers and not ui.is_stub(tree):
            problems.append(f"{where(name, markers[0])} keeps a stub marker but holds real markup: remove data-stub / data-stub-body")
    return problems


def audit_canvas(markup):
    problems = []
    for name, tree in markup:
        for el in tree.elements():
            if el.tag != "canvas":
                continue
            if el.get("aria-hidden") == "true":
                continue
            if el.get("role") == "img" and (el.get("aria-label") or el.get("aria-labelledby")):
                continue
            problems.append(f"{where(name, el)} is unlabelled: aria-hidden=true, or role=img with a label")
    return problems


def audit_disclosures(markup):
    problems = []
    for name, tree in markup:
        for el in tree.elements():
            role = el.get("role")
            if role == "tab" and not el.get("aria-controls"):
                problems.append(f"{where(name, el)} role=tab needs aria-controls")
            if role == "tabpanel" and not el.get("aria-labelledby"):
                problems.append(f"{where(name, el)} role=tabpanel needs aria-labelledby")
            if el.has("aria-expanded") and not el.get("aria-controls"):
                problems.append(f"{where(name, el)} has aria-expanded but no aria-controls")
            if role in ("radiogroup", "group", "tablist", "listbox", "menu", "toolbar") and not (el.get("aria-label") or el.get("aria-labelledby")):
                if role in ("radiogroup", "tablist", "listbox"):
                    problems.append(f"{where(name, el)} role={role} needs a label (aria-labelledby or aria-label)")
    return problems


# ───────────────────────────── JS lint ─────────────────────────────


def _rel(src: ui.JsSource) -> str:
    try:
        return src.path.resolve().relative_to(ui.JS_ROOT.resolve()).as_posix()
    except ValueError:
        return src.path.name


def _find(src, pattern, text=None, flags=0):
    hay = src.code if text is None else text
    return [src.line_of(m.start()) for m in re.finditer(pattern, hay, flags)]


def _icons() -> set[str]:
    text = (ui.JS_ROOT / "core" / "icons.js").read_text(encoding="utf-8")
    block = re.search(r"ICONS\s*=\s*Object\.freeze\(\[(.*?)\]\)", text, re.S).group(1)
    block = re.sub(r"//[^\n]*", "", block)
    return set(re.findall(r"'([\w-]+)'", block))


def _storage_keys() -> set[str]:
    text = (ui.JS_ROOT / "core" / "storage.js").read_text(encoding="utf-8")
    keys = set(re.findall(r"'(open-speech-[\w-]+)'", text))
    return keys | {"open-speech-more-tab", "open-speech-tts-history", "open-speech-stt-history"}


# The URL rule for URL properties (el.src = …, a.href = …): a value that is not a literal, safeUrl(…) or an object URL
# must be one of these reviewed sinks, each with the reason its value is safe. A new
# `el.src = serverValue` fails the lint until it goes through dom.safeUrl() or is reviewed into this list.
URL_PROPERTY_SINKS = {
    ("core/audio.js", "el.src = src"): "prime() sets its data: WAV; playClip() and its reload set the clip's src, which "
                                       "every caller makes with URL.createObjectURL, api.mediaUrl() or safeUrl()",
    ("core/download.js", "a.href = href"): "download() passes a blob: URL or a same-origin path",
    ("views/reader/engine.js", "el.src = url"): "a part URL that went through safeUrl()",
    ("views/speak/takes.js", "take.src = take.url"): "a take record (not an element): the object URL made just above",
    ("views/voices/clone.js", "el.draftAudio.src = url"): "the object URL of the recorded or uploaded draft",
    ("views/models/cards.js", "row.action = state.busy ? null : state.action"): "a row model's action object, not a URL",
}
URL_PROPERTY_ASSIGN = re.compile(r"(?<![\w$.])([\w$]+(?:\.[\w$]+)*)\.(src|href|poster|action|formAction)\s*=(?![=>])\s*([^;\n]*)")
URL_PROPERTY_SAFE = re.compile(r"""(?:'(?:[^'\\]|\\.)*'|"(?:[^"\\]|\\.)*"|`[^`$]*`|null\b|(?:[\w$]+\.)?safeUrl\s*\(|URL\.createObjectURL\s*\()""")


def lint_js(src: ui.JsSource, rel: str | None = None) -> list[str]:
    """Every module convention and lint rule for one module (rel: its path under src/static/js)."""
    rel = rel or _rel(src)
    problems: list[str] = []

    def flag(lines, message):
        for line in lines:
            problems.append(f"{rel}:{line}: {message}")

    def only(allowed, pattern, message):
        if rel not in allowed:
            flag(_find(src, pattern), message)

    only({"core/api.js", "shell/view-host.js"}, r"(?<![\w$.])fetch\s*\(|\b(?:window|globalThis|self)\s*\.\s*fetch\b", "fetch() only in core/api.js (and the allowlisted partial fetch in shell/view-host.js)")
    only({"core/api.js"}, r"\bWebSocket\s*\(", "WebSocket only in core/api.js (api.openSocket)")
    only({"core/storage.js", "theme-boot.js"}, r"\b(?:localStorage|sessionStorage)\b", "localStorage/sessionStorage only in core/storage.js")
    flag(_find(src, r"\bindexedDB\b"), "no IndexedDB (localStorage through core/storage.js only)")
    only({"core/audio.js"}, r"\bmediaSession\b", "navigator.mediaSession only in core/audio.js")
    # The page's audio session type is page-wide (Safari 17+): one owner, like the Media Session.
    only({"core/audio.js"}, r"\baudioSession\b", "navigator.audioSession only in core/audio.js")
    flag(_find(src, r"createMediaElementSource"), "never createMediaElementSource (long-form audio stays a plain media element)")
    only({"core/mic.js"}, r"\bcreateScriptProcessor\b", "createScriptProcessor only in core/mic.js (fallback)")
    for m in re.finditer(r"\baddModule\s*\(", src.no_comments):
        rest = src.no_comments[m.end():m.end() + 80]
        if not re.match(r"\s*(['\"])/static/js/core/mic-worklet\.js\1\s*\)", rest):
            problems.append(f"{rel}:{src.line_of(m.start())}: audioWorklet.addModule only with the static '/static/js/core/mic-worklet.js'")
    flag(_find(src, r"\bnew\s+(?:Shared)?Worker\s*\(|\bserviceWorker\s*\.\s*register\b"), "no workers or service workers (the CSP has no worker-src)")
    only({"core/dom.js"}, r"\.(?:inner|outer)HTML\s*\+?=(?!=)", "innerHTML/outerHTML only in core/dom.js (render)")
    flag(_find(src, r"\binsertAdjacentHTML\b|\bdocument\s*\.\s*write(?:ln)?\s*\(|\bcreateContextualFragment\b"), "no insertAdjacentHTML/document.write/createContextualFragment")
    only({"core/dialogs.js"}, r"\.showModal\s*\(", "showModal only in core/dialogs.js")
    flag(_find(src, r"\b(?:window|globalThis|self)\s*\.\s*(?:alert|confirm|prompt)\s*\("), "no window.alert/confirm/prompt: use toast and dialogs")
    flag(_find(src, r"(?<![\w$.])alert\s*\("), "no alert(): use toast")
    for name in ("confirm", "prompt"):
        imported = re.search(rf"import\s*\{{[^}}]*\b{name}\b[^}}]*\}}\s*from", src.no_comments)
        for m in re.finditer(rf"(?<![\w$.]){name}\s*\(", src.code):
            before = src.code[max(0, m.start() - 20):m.start()]
            if imported or re.search(r"function\s*\*?\s*$", before):
                continue
            problems.append(f"{rel}:{src.line_of(m.start())}: bare {name}() is window.{name}: use dialogs.{name}")
    flag(_find(src, r"(?<![\w$.])eval\s*\(|\bnew\s+Function\s*\("), "no eval or new Function")
    flag(_find(src, r"\b(?:setTimeout|setInterval)\s*\(\s*['\"`]", src.no_comments), "no string timers")
    for line, value in src.templates + src.strings:
        if "<" in value and re.search(r"\son[a-z]+\s*=", value):
            problems.append(f"{rel}:{line}: inline on*= handler in markup text (use data-action delegation)")
        if "<" in value and re.search(r"\sstyle\s*=", value):
            problems.append(f"{rel}:{line}: style= in markup text (set custom properties with el.style.setProperty)")
        if re.search(r"javascript\s*:", value, re.I):
            problems.append(f"{rel}:{line}: javascript: URL")
        if (re.search(r"https?://", value, re.I) or re.match(r"//[a-z0-9]", value, re.I)) and value.strip() not in SAFE_EXTERNAL:
            problems.append(f"{rel}:{line}: {value[:60]!r} points to another origin (no CDN or external URL)")
    # Markup and URLs built from data stay safe.
    for line, value in src.templates:
        if "<" in value and re.search(r"<[a-zA-Z][^<>]*?\s[\w:.-]+=\s*\x00", value):
            problems.append(f"{rel}:{line}: unquoted attribute value in a template (quote every attribute, a space would end it)")
    for m in re.finditer(r"(?<![\w$.])raw\s*\(", src.no_comments):
        arg = src.no_comments[m.end():m.end() + 400].lstrip()
        if (arg.startswith("`") and re.match(r"`(?:[^`\\]|\\.)*?\$\{", arg)) or re.match(r"[^()]*?\+", arg.split(")", 1)[0]):
            problems.append(f"{rel}:{src.line_of(m.start())}: raw() fed from a template with ${{}} or a concatenation (build data markup with html``)")
    only({"core/dom.js", "core/icons.js"}, r"\bnew\s+SafeHtml\s*\(", "new SafeHtml() only in core/dom.js and core/icons.js (use html``)")
    # The attribute name stands alone: data-action="${…}" (the dom.actions hook), data-src or data-menu-action
    # are not URL attributes, so no name character may come right before it.
    for m in re.finditer(r"""(?<![\w:-])(?:href|src|action|formaction|poster|srcset|xlink:href)\s*=\s*(["'])\$\{(?!\s*(?:[\w$]+\.)?safeUrl\s*\()(?!\s*[A-Z][A-Z0-9_]*\s*\})""", src.no_comments):
        problems.append(f"{rel}:{src.line_of(m.start())}: a URL attribute that starts with data must go through dom.safeUrl()")
    for m in re.finditer(r"""\.setAttribute\s*\(\s*(['"])(?:href|src|action|formaction|poster|srcset|xlink:href)\1\s*,(?!\s*(?:['"]|`(?!\$\{)|(?:[\w$]+\.)?safeUrl\s*\())""", src.no_comments):
        problems.append(f"{rel}:{src.line_of(m.start())}: setAttribute of a URL from data must go through dom.safeUrl()")
    for m in URL_PROPERTY_ASSIGN.finditer(src.no_comments):
        target, prop, rhs = m.group(1), m.group(2), m.group(3).strip()
        if URL_PROPERTY_SAFE.match(rhs) or (rel, f"{target}.{prop} = {rhs}") in URL_PROPERTY_SINKS:
            continue
        problems.append(f"{rel}:{src.line_of(m.start())}: a URL property ({target}.{prop} = …) from data must go through "
                        "dom.safeUrl(), or be a reviewed sink in URL_PROPERTY_SINKS")
    flag(_find(src, r"\.target\s*\.\s*dataset\b"), "event.target.dataset breaks on clicks on an icon: use closest('[data-…]')")
    # The JS harness's querySelectorAll returns an Array; a browser's returns a NodeList, which has forEach and
    # indexing but none of these, so such a call passes every Node test and throws in the browser (a
    # NodeList .some() in the AI assistant section was caught by a screenshot).
    flag(_find(src, r"querySelectorAll\s*\([^()]*\)\s*\.\s*(?:some|every|map|filter|find|findIndex|findLast|findLastIndex|reduce|reduceRight|indexOf|lastIndexOf|includes|at|flat|flatMap|slice|join|sort|concat)\s*\("),
         "a NodeList has no Array methods: wrap querySelectorAll() in Array.from() (the JS harness returns an Array, a browser doesn't)")
    flag(_find(src, r"\b(?:window|globalThis)\s*\.\s*[\w$]+\s*=(?!=)"), "no globals: nothing is assigned to window")
    for spec in ui.static_imports(src.path):
        if not (spec.startswith(("./", "../")) and spec.endswith(".js")):
            problems.append(f"{rel}: import {spec!r} must be relative and end in .js")
    for m in re.finditer(r"\bimport\s*\(\s*(['\"`])([^'\"`]*)\1", src.no_comments):
        if not (m.group(2).startswith(("./", "../")) and m.group(2).endswith(".js")):
            problems.append(f"{rel}:{src.line_of(m.start())}: import({m.group(2)!r}) must be relative and end in .js")
    icons = _icons()
    sprite = ui.sprite_ids()
    for m in re.finditer(r"(?<![\w$.])icon\(\s*(['\"])([^'\"]+)\1", src.no_comments):
        name = m.group(2)
        if name not in icons:
            problems.append(f"{rel}:{src.line_of(m.start())}: icon({name!r}) is not in icons.ICONS")
        elif sprite is not None and f"i-{name}" not in sprite:
            problems.append(f"{rel}:{src.line_of(m.start())}: icon({name!r}) is not in sprite.svg")
    for line, value in src.strings + src.templates:
        for path in re.findall(r"/static/partials/[\w./-]+\.html", value):
            if not (ui.STATIC / path.removeprefix("/static/")).exists():
                problems.append(f"{rel}:{line}: partial {path} does not exist")
    if rel not in ("core/storage.js", "theme-boot.js"):
        keys = _storage_keys()
        for line, value in src.strings + src.templates:
            if value in keys:
                problems.append(f"{rel}:{line}: storage key {value!r} outside core/storage.js (use KEYS)")
    return problems


def lint_all() -> list[str]:
    problems = []
    for path in ui.js_files():
        problems.extend(lint_js(ui.lex_js(path)))
    return problems


def lint_fixture(tmp_path, rel: str, source: str) -> list[str]:
    path = tmp_path / Path(rel).name
    path.write_text(source, encoding="utf-8")
    return lint_js(ui.lex_js(path), rel)


# ───────────────────────────── fixtures ─────────────────────────────

GOOD_PARTIAL = """
<h1 id="sp-title" data-view-heading tabindex="-1">Speak</h1>
<div class="seg" role="radiogroup" aria-labelledby="sp-mode-label"><span id="sp-mode-label">Mode</span>
  <label class="seg__opt"><input class="seg__input" type="radio" name="sp-mode" value="generate"><span>Generate</span></label>
  <label class="seg__opt"><input class="seg__input" type="radio" name="sp-mode" value="live"><span>Live</span></label>
</div>
<div class="field"><label class="field__label" for="sp-input">Text to speak</label>
  <textarea id="sp-input" class="textarea" aria-describedby="sp-limit-hint"></textarea>
  <p class="field__error" id="sp-limit-hint" hidden>Too long</p></div>
<label class="switch"><input type="checkbox" role="switch" class="switch__input" id="sp-stream"><span class="switch__track" aria-hidden="true"></span><span>Stream</span></label>
<button type="button" class="btn btn--primary" id="sp-generate"><span class="btn__label">Generate</span></button>
<button type="button" class="icon-btn" aria-label="Close" data-tooltip="Close"><svg class="ico" aria-hidden="true" focusable="false"><use href="/static/icons/sprite.svg#i-x"></use></svg></button>
<button type="button" class="icon-btn" id="sp-adv" aria-expanded="false" aria-controls="sp-adv-body">Advanced</button><div id="sp-adv-body" hidden></div>
<label class="btn" for="sp-upload">Open text file…</label><input type="file" id="sp-upload" class="visually-hidden" accept=".txt">
<h2>Takes</h2><ol id="sp-takes"></ol>
<canvas class="sp-wave" aria-hidden="true"></canvas>
<p id="sp-gen-status" role="status"></p><div id="sp-gen-error" class="banner banner--error" role="alert" hidden></div>
<dialog class="sheet" id="sp-sheet" aria-labelledby="sp-sheet-title"><div class="sheet__head"><h2 id="sp-sheet-title">Sheet</h2><button class="icon-btn" type="button" data-close aria-label="Close">x</button></div></dialog>
"""

BAD_PARTIAL = """
<h1 id="sp-heading">Speak</h1>
<h3>Skipped a level</h3>
<input id="dup" type="text" placeholder="Only a placeholder">
<select id="dup"><option>a</option></select>
<button class="icon-btn"><svg aria-hidden="true"><use href="/static/icons/sprite.svg#i-nope"></use></svg></button>
<button type="button" onclick="go()" style="color:red">Go</button>
<a href="javascript:void(0)">x</a>
<a href="https://cdn.example.com/x.js">cdn</a>
<img src="//cdn.example.com/a.png" alt="">
<input type="file" id="f" hidden>
<canvas></canvas>
<div role="tab">Tab</div>
<button type="button" aria-expanded="true">More</button>
<p aria-describedby="nowhere">x</p>
<script src="/static/js/x.js"></script>
"""


# ───────────────────────────── tests: HTML audits ─────────────────────────────


def test_good_markup_passes_every_audit():
    """Markup built from the component catalogue passes every HTML audit (no false positives)."""
    markup = markup_from({"speak": GOOD_PARTIAL})
    sprite = {"i-x"}
    for audit in (audit_unique_ids, audit_idrefs, audit_control_names, audit_buttons, audit_csp_markup, audit_external_markup,
                  audit_file_inputs, audit_view_headings, audit_heading_order, audit_canvas, audit_disclosures):
        assert audit(markup) == [], audit.__name__
    assert audit_use_refs(markup, sprite) == []
    assert audit_live_regions(markup) == []


def test_ids_are_unique_across_the_shell_and_all_partials():
    """[AX-06] Ids are unique across the shell and every partial (every view stays mounted), and every
    aria-controls/aria-labelledby/aria-describedby target exists."""
    assert audit_unique_ids(real_markup()) == []
    assert audit_idrefs(real_markup()) == []
    bad = markup_from({"speak": BAD_PARTIAL, "voices": '<p id="dup"></p>'})
    assert any("duplicates id 'dup'" in p for p in audit_unique_ids(bad))
    assert any("aria-describedby='nowhere'" in p for p in audit_idrefs(bad))


def test_tab_like_controls_reference_their_panels():
    """[AX-06] Tab-like and disclosure controls reference their panels: role=tab and aria-expanded carry
    aria-controls, tabpanels are labelled, and segmented radiogroups have a group label."""
    assert audit_disclosures(real_markup()) == []
    problems = audit_disclosures(markup_from({"speak": BAD_PARTIAL + '<div role="radiogroup"></div>'}))
    assert any("role=tab needs aria-controls" in p for p in problems)
    assert any("aria-expanded but no aria-controls" in p for p in problems)
    assert any("role=radiogroup needs a label" in p for p in problems)


def test_every_form_control_has_an_accessible_name():
    """[AX-01] Every input, select and textarea in the shell and the partials has an accessible name
    (a label, aria-label or aria-labelledby; a placeholder is never the label). JS-built markup is checked
    with the harness's assertAccessibleNames in each view's tests."""
    assert audit_control_names(real_markup()) == []
    problems = audit_control_names(markup_from({"speak": BAD_PARTIAL}))
    assert any("<input#dup>" in p for p in problems)
    assert any("<select#dup>" in p for p in problems)


def test_every_button_has_a_name_and_a_type_and_icon_buttons_have_aria_labels():
    """[AX-02] Every button has an accessible name and an explicit type, and icon-only buttons carry an
    aria-label."""
    assert audit_buttons(real_markup()) == []
    problems = audit_buttons(markup_from({"speak": BAD_PARTIAL}))
    assert any("needs an explicit type" in p for p in problems)
    assert any("has no accessible name" in p for p in problems)
    named_by_title_only = markup_from({"x": '<button type="button" title="Close"><svg aria-hidden="true"></svg></button>'})
    assert any("icon-only without aria-label" in p for p in audit_buttons(named_by_title_only))


def test_reader_icon_buttons_have_accessible_names():
    """[RE-70] Every Reader icon-only button (back, chapters, skips, prev/next, play, close…) has an
    aria-label, in the library/reading partial and in the player slot."""
    reader = [(n, t) for n, t in real_markup() if n in ("reader", "reader-player")]
    assert audit_buttons(reader) == []
    transport = markup_from({"reader-player": """
      <button type="button" class="icon-btn icon-btn--lg" id="rdp-back" aria-label="Back 15 seconds"><svg aria-hidden="true"></svg><span class="tbtn__num" aria-hidden="true">15</span></button>
      <button type="button" class="play-btn" id="rdp-play"><svg aria-hidden="true"></svg></button>"""})
    problems = audit_buttons(transport)
    assert len(problems) == 1 and "rdp-play" in problems[0]


def test_markup_runs_under_the_strict_csp():
    """[GS-03] The shell and partials run under script-src 'self': no inline scripts or on*= handlers, no
    javascript: URLs, no style= attributes, no <script> in partials, and shell scripts only from
    /static/js/*.js. JS-built markup is linted for the same (see the JS lint test)."""
    assert audit_csp_markup(real_markup()) == []
    problems = audit_csp_markup(markup_from({"speak": BAD_PARTIAL}))
    assert any("inline handler onclick=" in p for p in problems)
    assert any("style= attribute" in p for p in problems)
    assert any("javascript: URL" in p for p in problems)
    assert any("partials hold no <script>" in p for p in problems)
    shell = markup_from({"shell": '<script src="/static/js/theme-boot.js"></script><script type="module" src="/static/js/main.js"></script><script>alert(1)</script><script src="https://cdn.example.com/a.js"></script>'})
    shell_problems = audit_csp_markup(shell)
    assert sum("inline script" in p for p in shell_problems) == 1
    assert any("is not under /static/js/" in p for p in shell_problems)
    assert len(shell_problems) == 3, shell_problems


def test_the_web_ui_csp_has_no_worker_src_or_unsafe_eval():
    """[GS-05] The CSP the UI runs under: script-src 'self' (no 'unsafe-eval'), no worker-src (so the
    AudioWorklet is the static /static/js/core/mic-worklet.js, never a blob:), media and images from
    'self', blob: and data:. The JS lint enforces the static worklet URL and no workers."""
    text = (ui.REPO / "src" / "middleware.py").read_text(encoding="utf-8")
    csp = "".join(re.findall(r'"([^"]*)"', re.search(r"WEB_UI_CSP\s*=\s*\((.*?)\)", text, re.S).group(1)))
    assert "script-src 'self';" in csp
    assert "unsafe-eval" not in csp
    assert "worker-src" not in csp
    assert "media-src 'self' blob: data:" in csp
    assert (ui.JS_ROOT / "core" / "mic-worklet.js").exists()


def test_nothing_loads_from_another_origin():
    """[GS-06] No CDN, external font, script or link: no URL to another origin in the shell, the partials,
    the CSS or the JS (LAN, offline and self-signed TLS keep working)."""
    assert audit_external_markup(real_markup()) == []
    assert audit_external_css(ui.css_files()) == []
    assert [p for p in lint_all() if "another origin" in p] == []
    problems = audit_external_markup(markup_from({"speak": BAD_PARTIAL}))
    assert any("cdn.example.com/x.js" in p for p in problems)
    assert any("//cdn.example.com/a.png" in p for p in problems)


def test_external_css_urls_are_flagged(tmp_path):
    """[GS-06] url() and @import pointing to another origin are refused in CSS."""
    css = tmp_path / "bad.css"
    css.write_text("@import url('https://fonts.example.com/x.css');\n.a { background: url(//cdn.example.com/a.png); }\n.b { background: url('/static/icons/icon.svg'); }", encoding="utf-8")
    problems = audit_external_css([css])
    assert len(problems) == 2


def test_icon_references_resolve_to_the_sprite():
    """Every <use href> points to /static/icons/sprite.svg#i-… and the id exists in the sprite."""
    assert audit_use_refs(real_markup(), ui.sprite_ids()) == []
    problems = audit_use_refs(markup_from({"speak": BAD_PARTIAL}), {"i-x"})
    assert any("i-nope, which is not in sprite.svg" in p for p in problems)
    assert any("sprite.svg does not exist" in p for p in audit_use_refs(markup_from({"speak": GOOD_PARTIAL}), None))


def test_file_inputs_are_visually_hidden_and_reached_through_a_label():
    """[AX-05] File inputs are .visually-hidden (never hidden or display:none) and reachable through a
    <label>."""
    assert audit_file_inputs(real_markup()) == []
    problems = audit_file_inputs(markup_from({"speak": BAD_PARTIAL}))
    assert any("is hidden" in p for p in problems)
    assert any("are .visually-hidden" in p for p in problems)
    assert any("not reachable through a <label>" in p for p in problems)


def test_view_partials_have_one_focusable_h1_and_slot_partials_none():
    """[REV-16] Each view partial has exactly one h1 (the Reader two: library and reading) with
    data-view-heading, an id ending in -title and tabindex="-1"; slot partials have no h1."""
    assert audit_view_headings(real_markup()) == []
    problems = audit_view_headings(markup_from({"speak": BAD_PARTIAL, "reader-player": "<h1>Player</h1>", "voices": "<h2>No h1</h2>"}))
    assert any("needs data-view-heading" in p for p in problems)
    assert any("id must end in -title" in p for p in problems)
    assert any('tabindex="-1"' in p for p in problems)
    assert any("slot partials have no h1" in p for p in problems)
    assert any("voices: a view partial has exactly one h1, found 0" in p for p in problems)
    reader = markup_from({"reader": '<h1 id="rd-title" data-view-heading tabindex="-1">Library</h1><h1 id="rd-doc-title" data-view-heading tabindex="-1">Doc</h1>'})
    assert audit_view_headings(reader) == []


def test_headings_never_skip_a_level_and_canvases_are_labelled():
    """[BUG-22] No unnamed controls (the control and button audits), no unlabelled canvases, heading levels
    never skip, and status updates have live regions (the live-region audit)."""
    markup = real_markup()
    assert audit_heading_order(markup) == []
    assert audit_canvas(markup) == []
    assert audit_control_names(markup) == [] and audit_buttons(markup) == []
    bad = markup_from({"speak": BAD_PARTIAL})
    assert any("skips from h1 to h3" in p for p in audit_heading_order(bad))
    assert any("<canvas> is unlabelled" in p or "canvas" in p for p in audit_canvas(bad))
    assert audit_heading_order(markup_from({"reader-player": "<h2>Player</h2><h3>Speed</h3>"})) == []


def test_required_live_regions_exist_with_their_roles():
    """[AX-03] The required live regions exist with the right roles: the shell's status, alert, toast and
    connection regions, and each built view's status line (the log for Transcribe, with the interim text
    outside it). Placeholder partials (data-stub) are skipped until their view is built."""
    assert audit_live_regions(real_markup()) == []
    shell = markup_from({"shell": '<div id="app-sr-status" role="status"></div><div id="app-sr-alert" role="status"></div><div id="app-toasts-polite" role="status"></div><div id="app-toasts-alert" role="alert"></div>'})
    problems = audit_live_regions(shell)
    assert any("#app-sr-alert" in p and "role=alert" in p for p in problems)
    assert any("#app-conn-banner (role=status) is missing" in p for p in problems)
    transcribe = markup_from({"transcribe": '<h1 id="tr-title" data-view-heading tabindex="-1">Transcribe</h1><div id="tr-log" role="log"><p id="tr-interim"></p></div><p id="tr-upload-status" role="status"></p>'})
    tp = audit_live_regions(transcribe)
    assert any("aria-live=polite" in p for p in tp)
    assert any("outside #tr-log" in p for p in tp)
    stub = markup_from({"models": '<header class="view-head"><h1 id="mo-title" data-view-heading tabindex="-1">Models</h1></header><div class="view-stub" data-stub-body></div>'})
    assert audit_live_regions(stub) == []
    # A real partial that kept the marker (on its h1, as an older stub had it) is audited all the same.
    kept = markup_from({"speak": '<h1 id="sp-title" data-view-heading tabindex="-1" data-stub>Speak</h1><textarea id="sp-input" aria-label="Text"></textarea><button type="button" id="sp-generate">Generate</button><p id="sp-gen-status"></p>'})
    assert any("#sp-gen-status" in p and "role=status" in p for p in audit_live_regions(kept))


def test_stub_markers_are_removed_when_a_partial_lands():
    """A partial is a placeholder only while it is the bare stub (h1 plus an empty data-stub-body); a marker left
    in a partial with real markup is flagged, and the real partials pass."""
    assert audit_stub_markers(real_markup()) == []
    kept = markup_from({"speak": '<h1 id="sp-title" data-view-heading tabindex="-1" data-stub>Speak</h1><button type="button" id="sp-generate">Generate</button>'})
    assert any("keeps a stub marker" in p for p in audit_stub_markers(kept))
    bare = markup_from({"speak": '<header class="view-head"><h1 id="sp-title" data-view-heading tabindex="-1">Speak</h1></header><div class="view-stub" data-stub-body></div>', "speak-live": '<div data-stub hidden></div>'})
    assert audit_stub_markers(bare) == []
    for _, tree in real_markup():
        for el in tree.elements():
            if el.tag == "h1":
                assert "data-stub" not in el.attrs, "the marker sits on the stub body, not on the h1 the view keeps"


# ───────────────────────────── tests: JS lint and the view contract ─────────────────────────────


def test_js_architecture_lint_passes_on_every_module():
    """[GS-03][GS-05] Every module under src/static/js follows the architecture rules: fetch/WebSocket only
    in core/api.js, storage only in core/storage.js, innerHTML only in core/dom.js, showModal only in
    core/dialogs.js, mediaSession only in core/audio.js, no inline handlers or style= in markup text, no
    window dialogs or eval, closest() instead of event.target.dataset, relative .js imports, existing icons
    and partials, the static worklet URL and no workers."""
    assert lint_all() == []


def test_js_lint_catches_each_rule(tmp_path):
    """[GS-03] The JS lint flags each forbidden pattern (a fixture per rule)."""
    cases = {
        "fetch() only in core/api.js": "export async function x() { return fetch('/api/models'); }",
        "WebSocket only in core/api.js": "export const s = () => new WebSocket('ws://x');",
        "localStorage/sessionStorage only": "export const v = () => localStorage.getItem('k');",
        "navigator.mediaSession only": "export const m = () => navigator.mediaSession;",
        "navigator.audioSession only": "export const m = () => { navigator.audioSession.type = 'playback'; };",
        "never createMediaElementSource": "export const a = (c, el) => c.createMediaElementSource(el);",
        "createScriptProcessor only": "export const p = (c) => c.createScriptProcessor(4096, 1, 1);",
        "audioWorklet.addModule only": "export const w = (c, u) => c.audioWorklet.addModule(u);",
        "no workers": "export const w = () => new Worker('/x.js');",
        "innerHTML/outerHTML only": "export const r = (el, s) => { el.innerHTML = s; };",
        "no insertAdjacentHTML": "export const r = (el, s) => el.insertAdjacentHTML('beforeend', s);",
        "showModal only": "export const o = (d) => d.showModal();",
        "no window.alert/confirm/prompt": "export const q = () => window.confirm('Sure?');",
        "bare confirm() is window.confirm": "export const q = () => confirm('Sure?');",
        "no eval or new Function": "export const e = (s) => eval(s);",
        "inline on*= handler": "export const t = (h) => h`<button type=\"button\" onclick=\"go()\">x</button>`;",
        "style= in markup text": "export const t = (h) => h`<span style=\"width: 3px\"></span>`;",
        "javascript: URL": "export const u = 'javascript:alert(1)';",
        "points to another origin": "export const u = 'https://cdn.example.com/lib.js';",
        "event.target.dataset": "export const h = (e) => e.target.dataset.id;",
        "a NodeList has no Array methods": "export const n = (el) => el.querySelectorAll('option').some((o) => o.selected);",
        "no globals": "export const g = () => { window.myGlobal = 1; };",
        "must be relative and end in .js": "import { x } from 'lodash';\nexport const y = x;",
        "is not in icons.ICONS": "import { icon } from './icons.js';\nexport const i = () => icon('not-an-icon');",
        "does not exist": "export default { partial: '/static/partials/nope.html' };",
        "storage key": "export const k = 'open-speech-api-key';",
    }
    for message, source in cases.items():
        problems = lint_fixture(tmp_path, "views/fixture/index.js", source)
        assert any(message in p for p in problems), (message, problems)
    allowed = lint_fixture(tmp_path, "core/api.js", "export const f = () => fetch('/x');\nexport const s = () => new WebSocket('ws://h/x');")
    assert allowed == []


def test_js_lint_enforces_the_markup_and_url_rules(tmp_path):
    """[GS-03] The markup and URL rules are enforced, not only described: unquoted template attributes (a space in a
    server string would add attributes), raw() fed from a template or a concatenation, new SafeHtml outside
    dom.js/icons.js, and href/src values (or setAttribute, or a URL property such as el.src = …) that start with
    data without dom.safeUrl()."""
    bad = {
        "unquoted attribute value": "import { html } from '../../core/dom.js';\nexport const a = (t) => html`<li class=\"sp-take\" data-id=${t.id} title=${t.title}>x</li>`;",
        "raw() fed from a template": "import { raw } from '../../core/dom.js';\nexport const b = (name) => raw(`<p>${name}</p>`);",
        "raw() fed from a template with ${} or a concatenation": "import { raw } from '../../core/dom.js';\nexport const c = (name) => raw('<p>' + name + '</p>');",
        "new SafeHtml() only": "import { SafeHtml } from '../../core/dom.js';\nexport const d = (s) => new SafeHtml(`<b>${s}</b>`);",
        "A URL attribute that starts with data".lower(): "import { html } from '../../core/dom.js';\nexport const e = (t) => html`<a href=\"${t.url}\">Open</a>`;",
        "setAttribute of a URL from data": "export const f = (el, t) => el.setAttribute('href', t.url);",
        "a URL property (el.src = …) from data": "export const g = (el, t) => { el.src = t.url; };",
        "a URL property (a.href = …) from data": "export const h = (a, t) => { a.href = t.link; };",
        "a URL property (player.el.poster = …) from data": "export const i = (player, t) => { player.el.poster = t.cover; };",
        "a URL property (form.action = …) from data": "export const j = (form, t) => { form.action = t.target; };",
    }
    for message, source in bad.items():
        problems = lint_fixture(tmp_path, "views/speak/takes.js", source)
        assert any(message.lower() in p.lower() for p in problems), (message, problems)
    good = (
        "import { html, raw, safeUrl } from '../../core/dom.js';\n"
        "const SPRITE = '/static/icons/sprite.svg';\n"
        "export const a = (t) => html`<li class=\"sp-take\" data-id=\"${t.id}\"><a href=\"${safeUrl(t.url)}\">Open</a><a href=\"#reader/${encodeURIComponent(t.id)}\">Read</a></li>`;\n"
        "export const b = (markup) => raw(markup);\n"
        "export const c = (el, t) => { el.setAttribute('href', safeUrl(t.url)); el.setAttribute('src', '/static/x.png'); };\n"
        "export const d = () => html`<svg><use href=\"${SPRITE}#i-x\"></use></svg>`;\n"
        "export const e = (el, a, t, blob) => { el.src = safeUrl(t.url); el.src = URL.createObjectURL(blob); a.href = '/static/x.png'; el.src = ''; el.src = null; };\n"
        "export const f = (x) => x.src === 'a' || x.href == null;\n"
    )
    assert lint_fixture(tmp_path, "views/speak/ok.js", good) == []
    # A reviewed sink passes only in its own module and with its own value.
    assert lint_fixture(tmp_path, "views/speak/takes.js", "export const s = (take) => { take.src = take.url; };") == []
    assert lint_fixture(tmp_path, "views/speak/other.js", "export const s = (take) => { take.src = take.url; };") != []
    assert lint_fixture(tmp_path, "core/audio.js", "export const s = (el, row) => { el.src = row.url; };") != []
    # All three unsafe patterns in one module.
    review = (
        "import { html, render, raw } from '../../core/dom.js';\n"
        "export function paint(el, take, name) {\n"
        "  render(el, html`<li class=\"sp-take\" data-id=${take.id} title=${take.title}><a href=\"${take.url}\">Open</a></li>`);\n"
        "  render(el, raw(`<p>${name}</p>`));\n"
        "  el.setAttribute('href', take.url);\n"
        "}\n"
    )
    problems = lint_fixture(tmp_path, "views/speak/takes.js", review)
    assert len([p for p in problems if "unquoted" in p]) == 1
    assert any("raw()" in p for p in problems) and any("safeUrl" in p for p in problems)
    ok = lint_fixture(tmp_path, "views/fixture/ok.js", "import { confirm } from '../../core/dialogs.js';\nimport { icon } from '../../core/icons.js';\nexport const a = () => confirm({ title: 'x' });\nexport const b = () => icon('play');\nexport const c = (el) => el.closest('[data-action]').dataset.action;\nconst re = /https?:\\/\\/[^\\s]+/g;\nexport const d = (s) => s.replace(re, '');\n")
    assert ok == [], ok


def test_js_lint_url_rule_reads_whole_attribute_names(tmp_path):
    """[GS-03] The safeUrl rule is about URL attributes only: data-action="${…}" (the dom.actions delegation
    hook), data-src, data-href and data-menu-action carry no URL and pass, while a
    real href, src, action, formaction, poster, srcset or xlink:href built from data, in any position of a
    tag, is still reported."""
    data_attrs = (
        "import { html } from '../../core/dom.js';\n"
        "export const a = (name, cmd) => html`<button type=\"button\" class=\"btn\" data-action=\"${name}\" data-cmd=\"${cmd}\">Retry</button>`;\n"
        "export const b = (t) => html`<div data-src=\"${t.path}\" data-href=\"${t.link}\"><button type=\"button\" role=\"menuitem\" data-menu-action=\"${t.action}\">Go</button></div>`;\n"
        "export const c = (t) => html`<li data-xlink:href=\"${t.x}\" data-formaction=\"${t.y}\">x</li>`;\n"
    )
    assert lint_fixture(tmp_path, "views/transcribe/ui.js", data_attrs) == []
    real = {
        "href": "html`<a class=\"x\" href=\"${t.url}\">Open</a>`",
        "src": "html`<img alt=\"\" src='${t.url}'>`",
        "action": "html`<form action=\"${t.url}\"></form>`",
        "formaction": "html`<button type=\"submit\" formaction=\"${t.url}\">Go</button>`",
        "poster": "html`<video poster=\"${t.url}\"></video>`",
        "srcset": "html`<img alt=\"\" srcset=\"${t.url} 2x\">`",
        "xlink:href": "html`<svg><use xlink:href=\"${t.url}\"></use></svg>`",
        "after a data attribute": "html`<a data-action=\"open\" href=\"${t.url}\">Open</a>`",
    }
    for name, template in real.items():
        source = f"import {{ html }} from '../../core/dom.js';\nexport const e = (t) => {template};\n"
        problems = lint_fixture(tmp_path, "views/transcribe/ui.js", source)
        assert len([p for p in problems if "dom.safeUrl()" in p]) == 1, (name, problems)


def view_contract_problems(views: list[Path], static_root: Path = ui.STATIC) -> list[str]:
    """Import each views/<id>/index.js in Node and check its default export against the view contract."""
    if not views:
        return []
    script = """
import { pathToFileURL } from 'node:url';
import { VIEWS } from './src/static/js/core/router.js';
const files = %s;
const out = [];
for (const f of files) {
  const m = await import(pathToFileURL(f).href);
  const v = m.default || {};
  out.push({ file: f, id: v.id, title: v.title, partial: v.partial, init: typeof v.init, prefetch: 'prefetch' in v ? typeof v.prefetch : 'absent', views: VIEWS });
}
console.log(JSON.stringify(out));
""" % json.dumps([str(v) for v in views])
    results = json.loads(ui.run_node_script(script).strip().splitlines()[-1])
    problems = []
    for r in results:
        folder = Path(r["file"]).parent.name
        if r.get("id") not in r.get("views", []):
            problems.append(f"{folder}: id {r['id']!r} is not in router.VIEWS")
        if r.get("id") != folder:
            problems.append(f"{folder}: id {r['id']!r} does not match its folder")
        if not isinstance(r.get("title"), str) or not r.get("title"):
            problems.append(f"{folder}: missing title")
        if r.get("init") != "function":
            problems.append(f"{folder}: init() missing")
        if r.get("prefetch", "absent") not in ("absent", "function"):
            problems.append(f"{folder}: prefetch must be a function (it warms the view's lazy imports on idle)")
        partial = r.get("partial") or ""
        if not (partial.startswith("/static/partials/") and (static_root / partial.removeprefix("/static/")).exists()):
            problems.append(f"{folder}: partial {partial!r} does not exist")
    return problems


def test_view_contract_checker_catches_each_rule(tmp_path):
    """The view-contract check passes a correct view module and flags a wrong id, a missing title or
    init(), a partial that does not exist, and an optional prefetch that is not a function (fixtures, so it is
    proven before the views exist)."""
    static = tmp_path / "static"
    (static / "partials").mkdir(parents=True)
    (static / "partials" / "speak.html").write_text('<h1 id="sp-title" data-view-heading tabindex="-1">Speak</h1>', encoding="utf-8")
    good = tmp_path / "views" / "speak" / "index.js"
    bad = tmp_path / "views" / "voices" / "index.js"
    unknown = tmp_path / "views" / "nope" / "index.js"
    for path in (good, bad, unknown):
        path.parent.mkdir(parents=True)
    good.write_text("export default { id: 'speak', title: 'Speak', partial: '/static/partials/speak.html', init(root) { return { show() {}, hide() {} }; }, prefetch: () => Promise.resolve() };\n", encoding="utf-8")
    bad.write_text("export default { id: 'speak', partial: '/static/partials/voices.html', prefetch: './engine.js' };\n", encoding="utf-8")
    unknown.write_text("export default { id: 'nope', title: 'Nope', partial: '/static/partials/speak.html', init() {} };\n", encoding="utf-8")
    assert view_contract_problems([good], static) == []
    problems = view_contract_problems([bad, unknown], static)
    assert "voices: id 'speak' does not match its folder" in problems
    assert "voices: missing title" in problems
    assert "voices: init() missing" in problems
    assert "voices: partial '/static/partials/voices.html' does not exist" in problems
    assert "voices: prefetch must be a function (it warms the view's lazy imports on idle)" in problems
    assert "nope: id 'nope' is not in router.VIEWS" in problems
    assert not any(p.startswith("nope: prefetch") for p in problems), "prefetch is optional"


def test_view_modules_follow_the_view_contract():
    """Every views/*/index.js default export has {id, title, partial, init}; the id is in router.VIEWS and
    matches its folder, and the partial file exists."""
    views = sorted((ui.JS_ROOT / "views").glob("*/index.js"))
    # The view modules always exist, so none at all fails.
    assert views, f"no view modules under {ui.JS_ROOT / 'views'}"
    assert view_contract_problems(views) == []


# ───────────────────────────── tests: serving ─────────────────────────────


def test_js_mime_type_does_not_depend_on_the_host():
    """[REV-27] .js is served as text/javascript (and .webmanifest, .woff2 with their types) whatever the
    host's MIME table says (Windows registry, Linux /etc/mime.types)."""
    # A host table that gets every one of them wrong, in a fresh interpreter so this process is untouched.
    probe = (
        "import mimetypes\n"
        "for ext in ('.js', '.webmanifest', '.woff2'):\n"
        "    mimetypes.add_type('text/plain', ext)\n"
        "import src.routes.web\n"
        "print([mimetypes.guess_type(n)[0] for n in ('main.js', 'manifest.webmanifest', 'font.woff2')])\n"
    )
    result = subprocess.run([sys.executable, "-c", probe], cwd=ui.REPO, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "['text/javascript', 'application/manifest+json', 'font/woff2']"

    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from src.routes import web

    assert mimetypes.guess_type("main.js")[0] == "text/javascript"
    app = FastAPI()
    app.mount("/static", web.RevalidatingStaticFiles(directory=str(ui.STATIC)), name="static")
    res = TestClient(app).get("/static/js/core/api.js")
    assert res.status_code == 200
    assert res.headers["content-type"].split(";")[0] == "text/javascript"
    assert res.headers["cache-control"] == "no-cache"


def test_the_server_auth_answers_are_the_ones_core_api_parses():
    """The real SecurityMiddleware answers the way core/api.js and core/status.js expect: a 401 envelope
    {error: {message}}, a lockout 429 whose message mentions the API key with Retry-After, an auth-exempt
    /health that still answers during a lockout (a 2xx there never clears "locked" or "unauthorized"), and
    one-time tickets {ticket, expires_in: 60}."""
    from unittest.mock import patch

    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    import src.middleware as mw
    from src.config import Settings

    def client(**overrides):
        app = FastAPI()
        app.add_middleware(mw.SecurityMiddleware)
        app.include_router(mw.create_auth_router())

        @app.get("/api/models")
        async def models():
            return {"models": []}

        @app.get("/health")
        async def health():
            return {"status": "ok"}

        return app, Settings(_env_file=None, **overrides)

    app, settings = client(os_api_key="secret123", os_auth_failure_limit=2)
    mw.reset_security_state_for_tests()
    with patch("src.middleware.settings", settings), _reset_security_state_after(mw):
        c = TestClient(app)
        assert c.get("/api/models").status_code == 401
        wrong = c.get("/api/models", headers={"Authorization": "Bearer guess"})
        assert wrong.status_code == 401
        assert isinstance(wrong.json()["error"]["message"], str)
        good = {"Authorization": "Bearer secret123"}
        ticket = c.post("/api/auth/ws-ticket", headers=good).json()
        assert set(ticket) >= {"ticket", "expires_in"} and ticket["expires_in"] == 60
        assert c.post("/api/auth/ws-ticket", headers=good).json()["ticket"] != ticket["ticket"], "one ticket per call"
        c.get("/api/models", headers={"Authorization": "Bearer guess"})
        locked = c.get("/api/models", headers=good)
        assert locked.status_code == 429
        assert re.search(r"API key", locked.json()["error"]["message"], re.I)
        assert int(locked.headers["Retry-After"]) > 0
        assert c.get("/health").status_code == 200, "/health stays reachable (auth-exempt), so status.js keeps 'locked'"


@contextlib.contextmanager
def _reset_security_state_after(mw):
    """The lockout counters are module state: clear them even when an assertion fails."""
    try:
        yield
    finally:
        mw.reset_security_state_for_tests()


# ───────────────────────────── helpers' own tests ─────────────────────────────


def test_selector_subset_matches_the_js_harness():
    tree = ui.parse_html('<div id="a" class="x y"><p class="x">1</p><span data-k="v w"><b>2</b></span></div><p id="z">3</p>')
    assert [n.id or n.tag for n in ui.select(tree, "div > p.x, #z")] == ["p", "z"]
    assert [n.tag for n in ui.select(tree, "div b")] == ["b"]
    assert [n.tag for n in ui.select(tree, '[data-k~="w"]')] == ["span"]
    assert [n.tag for n in ui.select(tree, "p:not(.x)")] == ["p"]
    assert [n.tag for n in ui.select(tree, "p + span")] == ["span"]
    assert ui.matches(ui.by_id(tree, "a"), "div.x.y")


def test_accessible_name_order():
    tree = ui.parse_html("""
      <span id="lab">From labelledby</span>
      <input id="a" aria-labelledby="lab" aria-label="no">
      <input id="b" aria-label="From aria-label">
      <label for="c">From label for</label><input id="c">
      <label>Wrapped <input id="d"></label>
      <button id="e" type="button"><svg aria-hidden="true"><title>x</title></svg>Text</button>
      <input id="f" title="From title">
      <input id="g" placeholder="Never a name">""")
    names = {i: ui.accessible_name(ui.by_id(tree, i), tree) for i in "abcdefg"}
    assert names == {"a": "From labelledby", "b": "From aria-label", "c": "From label for", "d": "Wrapped", "e": "Text", "f": "From title", "g": ""}


def test_css_rules_tokens_and_contrast():
    css = """
@layer tokens, base;
@layer tokens {
  /* light */
  :root { color-scheme: light; --bg: #F7F7F4; --surface-1: #FFFFFF; --accent: #4F46E5; --text: #15171C;
          --accent-soft: color-mix(in srgb, var(--accent) 12%, transparent); --x: var(--text); }
  @media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { color-scheme: dark; --bg: #0E0F13; } }
  :root[data-theme="dark"] { color-scheme: dark; --bg: #0E0F13; }
  @media (prefers-contrast: more) { :root { --text: #000000; } }
}
.a, .b { color: var(--text); content: "a;b{}"; }
"""
    rules = ui.css_rules(css)
    selectors = [r[0] for r in rules]
    assert ":root" in selectors and '.a, .b' in selectors
    assert rules[selectors.index('.a, .b')][1] == [("color", "var(--text)"), ("content", '"a;b{}"')]
    media = [r for r in rules if r[0] == ':root:not([data-theme="light"])'][0]
    assert media[2] == ("@layer tokens", "@media (prefers-color-scheme: dark)")
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "tokens.css"
        path.write_text(css, encoding="utf-8")
        light = ui.tokens("light", path=path)
        dark = ui.tokens("dark", path=path)
        high = ui.tokens("light", contrast=True, path=path)
    assert light["--bg"] == "#F7F7F4" and dark["--bg"] == "#0E0F13"
    assert light["--x"] == "#15171C" and high["--text"] == "#000000"
    assert light["--accent-soft"] == "color-mix(in srgb, #4F46E5 12%, transparent)"
    soft = ui.resolve_color(light["--accent-soft"], light, over="#FFFFFF")
    assert tuple(round(c) for c in soft) == (234, 233, 252)   # ≈ #EAE9FC
    assert round(ui.contrast("#4F46E5", "#FFFFFF"), 2) == 6.29
    assert round(ui.contrast("#15171C", "#FFFFFF"), 2) == 17.93
    assert round(ui.contrast("#4F46E5", light["--accent-soft"], over="#FFFFFF", table=light), 2) == 5.26
    assert ui.parse_color("rgb(16 18 24 / .40)") == (16.0, 18.0, 24.0, 0.4)


def test_js_lexer_separates_code_strings_templates_and_regexes(tmp_path):
    path = tmp_path / "x.js"
    path.write_text("// fetch('/a')\nconst a = 'fetch(1)'; /* fetch( */\nconst t = html`<p title=\"${x}\">${y.innerHTML}</p>`;\nconst r = /fetch\\(/;\nconst ok = b / c / d;\n", encoding="utf-8")
    src = ui.lex_js(path)
    assert "fetch" not in src.code.replace("y.innerHTML", "")
    assert (1, "") not in src.strings and any(v == "fetch(1)" for _, v in src.strings)
    assert src.templates and src.templates[0][1].startswith('<p title="')
    assert "y.innerHTML" in src.code, "template expressions stay code"
    assert "b / c / d" in src.code


def test_static_import_closure(tmp_path):
    (tmp_path / "a.js").write_text("import { b } from './b.js';\nexport { c } from './c.js';\nconst lazy = () => import('./d.js');\n", encoding="utf-8")
    (tmp_path / "b.js").write_text("import './c.js';\nexport const b = 1;\n", encoding="utf-8")
    (tmp_path / "c.js").write_text("export const c = 1;\n", encoding="utf-8")
    (tmp_path / "d.js").write_text("export const d = 1;\n", encoding="utf-8")
    closure = [p.name for p in ui.static_import_closure(tmp_path / "a.js")]
    assert closure == ["b.js", "c.js"]


def test_no_module_imports_the_removed_ui_scripts():
    """No module imports the old UI's global scripts, which were deleted (tests/test_ui_swap.py checks the
    files are gone and nothing names them as a path)."""
    for path in ui.js_files():
        text = path.read_text(encoding="utf-8")
        assert not re.search(r"""from\s+['"][./]*(?:app|reader)\.js['"]""", text), path


# ───────────────────────────── CSS audits ─────────────────────────────
# The CSS audits (tokens complete in light and dark, contrast pairs including the
# composited placement rules, color-scheme, [hidden], reduced motion, data-motion, data-spacing, forced
# colours, no colour literals outside tokens.css, no px font-size, the backdrop-filter/transform/infinite
# allowlists, no CSS nesting, view CSS inside @layer views). Helpers: ui.css_rules, ui.tokens,
# ui.resolve_color, ui.contrast.
#
# Every audit below returns its problems; each test runs it on the real CSS (expecting none) and on
# a bad fixture (proving it catches the problem). Files are read as text; nothing renders.


CSS_TOKENS = ui.CSS_ROOT / "tokens.css"
CSS_BASE = ui.CSS_ROOT / "base.css"
CSS_COMPONENTS = ui.CSS_ROOT / "components.css"
CSS_SHELL = ui.CSS_ROOT / "shell.css"

LIGHT_ONLY_TOKENS = (
    "--font-ui", "--font-mono", "--font-serif", "--font-legible", "--fs-xs", "--fs-sm", "--fs-base", "--fs-md", "--fs-lg",
    "--fs-xl", "--fs-2xl", "--lh-ui", "--lh-tight", "--sp-1", "--sp-2", "--sp-3", "--sp-4", "--sp-5", "--sp-6", "--sp-7",
    "--sp-8", "--r-xs", "--r-sm", "--r-md", "--r-lg", "--r-pill", "--z-sticky", "--z-nav", "--z-player", "--z-popover",
    "--z-toast", "--z-skip", "--dur-1", "--dur-2", "--dur-3", "--ease-out", "--ease-in-out", "--gutter", "--topbar-h",
    "--bottombar-h", "--miniplayer-h", "--rail-w", "--content-max", "--inspector-w", "--measure", "--focus-width",
    "--rd-bg", "--rd-text", "--rd-meta", "--rd-accent", "--rd-hl",
)
THEMED_TOKENS = (
    "--bg", "--surface-1", "--surface-2", "--surface-3", "--border", "--border-strong", "--border-control", "--text", "--text-2",
    "--text-3", "--accent", "--accent-fill", "--on-accent", "--accent-soft", "--focus", "--success", "--warning", "--danger",
    "--danger-fill", "--rec", "--rec-fill", "--hl-sentence", "--scrim", "--shadow-1", "--shadow-2", "--shadow-3",
    "--blend-0", "--blend-1", "--blend-2", "--blend-3", "--blend-4", "--blend-5", "--blend-label",
)
SPEC_HEX = {  # the colour tokens, exact
    "light": {"--bg": "#F7F7F4", "--surface-1": "#FFFFFF", "--surface-2": "#F1F2F5", "--surface-3": "#E7E9EE", "--border": "#DADDE3",
              "--border-strong": "#7F8592", "--text": "#15171C", "--text-2": "#4B5160", "--text-3": "#5F6573", "--accent": "#4F46E5",
              "--accent-fill": "#4F46E5", "--on-accent": "#FFFFFF", "--focus": "#4F46E5", "--success": "#046C4E", "--warning": "#9A4A07",
              "--danger": "#B91C1C", "--danger-fill": "#B91C1C", "--rec": "#C2333A", "--rec-fill": "#C2333A"},
    "dark": {"--bg": "#0E0F13", "--surface-1": "#16181D", "--surface-2": "#1D2027", "--surface-3": "#252932", "--border": "#2E333D",
             "--border-strong": "#6B7385", "--text": "#ECEEF2", "--text-2": "#A9AFBC", "--text-3": "#8F96A3", "--accent": "#9B9CFF",
             "--accent-fill": "#5B5CE6", "--on-accent": "#FFFFFF", "--focus": "#B4B5FF", "--success": "#34D399", "--warning": "#FBBF24",
             "--danger": "#F87171", "--danger-fill": "#DC2626", "--rec": "#E5484D", "--rec-fill": "#C2333A"},
}
PLAIN = ("--bg", "--surface-1", "--surface-2", "--surface-3")
SOFT_BASES = ("--bg", "--surface-1", "--surface-2")   # accent-soft is never placed on surface-3
PAPERS = ("paper", "sepia", "night", "dusk")


def _rules(path):
    return ui.css_rules(path)


def _dark_blocks(rules, contrast=False):
    """The two dark declaration blocks: (OS dark when not forced light, forced dark)."""
    def pick(selector_ok, chain_ok):
        decls = {}
        for selector, body, chain in rules:
            if selector_ok(selector.strip()) and chain_ok(chain):
                decls.update(dict(body))
        return decls
    has_contrast = lambda chain: any("prefers-contrast" in c for c in chain)  # noqa: E731
    os_dark = pick(lambda s: s == ':root:not([data-theme="light"])',
                   lambda ch: any("prefers-color-scheme: dark" in c for c in ch) and has_contrast(ch) == contrast)
    forced = pick(lambda s: s == ':root[data-theme="dark"]', lambda ch: has_contrast(ch) == contrast and not any("prefers-color-scheme" in c for c in ch))
    return os_dark, forced


def audit_token_set(path=CSS_TOKENS):
    """Every token exists in light; every themed token is redefined for dark; the two dark blocks (and the two
    dark more-contrast blocks) are identical; the colour values are exact; color-scheme is declared in each."""
    problems = []
    rules = _rules(path)
    light = ui.tokens("light", path=path)
    for name in LIGHT_ONLY_TOKENS + THEMED_TOKENS:
        if name not in light:
            problems.append(f"light: {name} is missing")
    os_dark, forced = _dark_blocks(rules)
    for name in THEMED_TOKENS:
        if name not in forced:
            problems.append(f"dark: {name} is not redefined")
    if os_dark != forced:
        diff = sorted(set(os_dark.items()) ^ set(forced.items()))
        problems.append(f"the two dark blocks differ: {diff[:6]}")
    hc_os, hc_forced = _dark_blocks(rules, contrast=True)
    if hc_os != hc_forced:
        problems.append("the two dark more-contrast blocks differ")
    if not hc_forced:
        problems.append("no dark more-contrast block")
    for theme, expected in SPEC_HEX.items():
        table = ui.tokens(theme, path=path)
        for name, value in expected.items():
            if table.get(name, "").upper() != value:
                problems.append(f"{theme}: {name} is {table.get(name)!r}, expected {value}")
    root_scheme = [dict(b).get("color-scheme") for s, b, ch in rules if s.strip() == ":root" and not ch[1:]]
    if "light" not in root_scheme:
        problems.append(":root does not declare color-scheme: light")
    if os_dark.get("color-scheme") != "dark" or forced.get("color-scheme") != "dark":
        problems.append("the dark blocks do not declare color-scheme: dark")
    return problems


def contrast_pairs(theme, contrast_mode=False, path=CSS_TOKENS):
    """(label, ratio, minimum) for every colour pair WCAG AA checks, including the composited placement pairs."""
    t = ui.tokens(theme, contrast=contrast_mode, path=path)

    def c(fg, bg, over=None):
        return ui.contrast(t[fg], t[bg], over=t[over] if over else None, table=t)

    pairs = []
    for fg in ("--text", "--text-2", "--text-3", "--accent", "--success", "--warning", "--danger"):
        for bg in PLAIN:
            pairs.append((f"{fg} on {bg}", c(fg, bg), 4.5))
    for fg in ("--border-strong", "--focus", "--rec"):
        for bg in PLAIN:
            pairs.append((f"{fg} on {bg} (UI)", c(fg, bg), 3.0))
    for fill in ("--accent-fill", "--danger-fill", "--rec-fill", "--accent-fill-hover", "--danger-fill-hover", "--rec-fill-hover"):
        pairs.append((f"--on-accent on {fill}", c("--on-accent", fill), 4.5))
    for over in SOFT_BASES:
        for fg in ("--text", "--text-2", "--accent"):
            pairs.append((f"{fg} on --accent-soft over {over}", c(fg, "--accent-soft", over), 4.5))
    pairs.append(("--text-2 on --accent-soft over --surface-3 (meta on a tinted row)", c("--text-2", "--accent-soft", "--surface-3"), 4.5))
    for state in ("success", "warning", "danger"):
        pairs.append((f"--{state} on --{state}-soft", c(f"--{state}", f"--{state}-soft"), 4.5))
        pairs.append((f"--text-2 on --{state}-soft", c("--text-2", f"--{state}-soft"), 4.5))
    pairs.append(("--accent on --info-soft", c("--accent", "--info-soft"), 4.5))
    pairs.append(("--text-2 on --info-soft", c("--text-2", "--info-soft"), 4.5))
    for over in ("--bg", "--surface-1"):
        pairs.append((f"--text on --hl-sentence over {over}", c("--text", "--hl-sentence", over), 4.5))
    for over in ("--surface-1", "--surface-2", "--bg"):
        pairs.append((f"--text-2 on --hover over {over}", c("--text-2", "--hover", over), 4.5))
    pairs.append(("--tooltip-text on --tooltip-bg", c("--tooltip-text", "--tooltip-bg"), 4.5))
    for i in range(6):
        pairs.append((f"--blend-label on --blend-{i}", c("--blend-label", f"--blend-{i}"), 4.5))
        pairs.append((f"--blend-{i} on --surface-2 (UI)", c(f"--blend-{i}", "--surface-2"), 3.0))
    return pairs


def paper_pairs(path=CSS_TOKENS):
    """The reading paper themes: text, meta and accent on the paper, text and the accent underline on the
    highlight."""
    rules = _rules(path)
    pairs = []
    for paper in PAPERS:
        decls = {}
        for selector, body, _ in rules:
            if selector.strip() == f'[data-rd-paper="{paper}"]':
                decls.update(dict(body))
        if not decls:
            pairs.append((f"{paper}: missing", 0.0, 4.5))
            continue
        bg = decls["--rd-bg"]
        for name, minimum in (("--rd-text", 4.5), ("--rd-meta", 4.5), ("--rd-accent", 4.5)):
            pairs.append((f"{paper}: {name} on --rd-bg", ui.contrast(decls[name], bg), minimum))
        pairs.append((f"{paper}: --rd-text on --rd-hl", ui.contrast(decls["--rd-text"], decls["--rd-hl"], over=bg), 4.5))
        pairs.append((f"{paper}: --rd-accent underline on --rd-hl (UI)", ui.contrast(decls["--rd-accent"], decls["--rd-hl"], over=bg), 3.0))
    return pairs


def failing(pairs):
    return [f"{label}: {ratio:.2f} < {minimum}" for label, ratio, minimum in pairs if round(ratio, 2) < minimum]


def _strip_values(value: str) -> str:
    value = re.sub(r"var\(\s*--[\w-]+", "var(", value)       # custom property names are not colours
    value = re.sub(r"(['\"]).*?\1", "", value)              # strings (content: "!")
    value = re.sub(r"url\([^)]*\)", "", value)
    return value


COLOR_FUNCS = re.compile(r"\b(?:rgba?|hsla?|hwb|lab|lch|oklab|oklch|color)\(", re.I)
HEX = re.compile(r"#[0-9a-fA-F]{3,8}\b")
NAMED = re.compile(r"\b(?:white|black|red|green|blue|gray|grey|silver|orange|yellow|purple|pink|navy|teal|maroon|olive|lime|aqua|fuchsia|indigo|violet|brown|gold|crimson|tomato|coral|salmon|khaki|beige|ivory|lavender|plum|orchid|turquoise|cyan|magenta)\b", re.I)
SYSTEM = re.compile(r"\b(?:Canvas|CanvasText|LinkText|VisitedText|ActiveText|ButtonFace|ButtonText|ButtonBorder|Field|FieldText|Highlight|HighlightText|SelectedItem|SelectedItemText|Mark|MarkText|GrayText|AccentColor|AccentColorText)\b")
COLOR_PROPS = re.compile(r"^(?:color|background(?:-color|-image)?|border(?:-[a-z]+)*|outline(?:-color)?|box-shadow|text-shadow|fill|stroke|caret-color|accent-color|text-decoration(?:-color)?|column-rule(?:-color)?|--.*)$")


def audit_color_literals(files):
    """No colour literal outside tokens.css (hex, rgb()/hsl()…, named colours); system colours only inside
    forced-colours blocks; transparent and currentColor are fine."""
    problems = []
    for path in files:
        path = Path(path)
        if path.name == "tokens.css" and path.parent == ui.CSS_ROOT:
            continue
        for selector, body, chain in ui.css_rules(path):
            forced = any("forced-colors" in c for c in chain)
            for name, value in body:
                if name == "@keyframes-body" or not COLOR_PROPS.match(name):
                    continue
                v = _strip_values(value)
                if HEX.search(v) or COLOR_FUNCS.search(v) or NAMED.search(v):
                    problems.append(f"{path.name}: {selector} {{{name}: {value}}} has a colour literal (use a token)")
                if SYSTEM.search(v) and not forced:
                    problems.append(f"{path.name}: {selector} {{{name}: {value}}} uses a system colour outside forced-colors")
    return problems


def audit_type(files):
    """No px font sizes; every field (input, select, textarea and their classes) stays at 1rem or more, so iOS
    never zooms on focus."""
    problems = []
    field = re.compile(r"(?:^|[\s,>+~(])(?:input|select|textarea|\.input|\.select|\.textarea|\.stepper__value)(?=$|[\s,:.\[#)>+~])")
    for path in files:
        for selector, body, _ in ui.css_rules(path):
            for name, value in body:
                if name == "font-size" and re.search(r"\d(?:\.\d+)?px\b", value):
                    problems.append(f"{Path(path).name}: {selector} sets font-size in px ({value})")
                if name in ("font-size", "font") and field.search(selector):
                    ok = value.strip() in ("var(--fs-base)", "1rem", "inherit", "100%") or re.fullmatch(r"(?:1(?:\.\d+)?|[2-9](?:\.\d+)?)rem", value.strip())
                    if name == "font":
                        ok = value.strip() == "inherit"
                    if not ok:
                        problems.append(f"{Path(path).name}: {selector} sets a field below 1rem ({value})")
    return problems


BACKDROP_ALLOWED = re.compile(r"^(?:\.topbar|\.bottombar|\.miniplayer|#app-miniplayer|\.rd-topbar|\.rdp-transport)(?:\s*,\s*(?:\.topbar|\.bottombar|\.miniplayer|#app-miniplayer|\.rd-topbar|\.rdp-transport))*$")
CONTAINERS = re.compile(r"(?:^|[\s,>+~])(?:main|\.view|\.app|#app|#main|section\.view|\[data-view(?:=\"[\w-]+\")?\]|#view-[\w-]+)(?=$|[\s,:\[>+~])")


def _keyframes(files):
    out = {}
    for path in files:
        text = ui._strip_css_comments(Path(path).read_text(encoding="utf-8"))
        for m in re.finditer(r"@keyframes\s+([\w-]+)\s*\{", text):
            end = ui._find_block_end(text, m.end() - 1)
            out[m.group(1)] = text[m.end():end]
    return out


def audit_motion_allowlists(files):
    """The motion rules and GS-23: backdrop-filter only on the five bars (each with an opaque fallback); no transform, filter,
    backdrop-filter or will-change on main, .view, .app or a view section (fixed bars inside them would stop
    being viewport-fixed), and their animations only fade; infinite animations only for a spinner inside
    aria-busy and the recording dot (the busy pill does not pulse, REV-26)."""
    problems = []
    frames = _keyframes(files)
    for path in files:
        name = Path(path).name
        for selector, body, _chain in ui.css_rules(path):
            for prop, value in body:
                if prop in ("backdrop-filter", "-webkit-backdrop-filter") and value.strip() != "none" and not BACKDROP_ALLOWED.match(selector.strip()):
                    problems.append(f"{name}: backdrop-filter on {selector} (allowed: the top bar, bottom bar, mini-player, Reader top bar and transport)")
                for part in selector.split(","):
                    target = part.strip()
                    last = re.split(r"[\s>+~]+", target)[-1]
                    if CONTAINERS.search(" " + last) and prop in ("transform", "filter", "backdrop-filter", "-webkit-backdrop-filter", "perspective", "will-change", "contain") and value.strip() not in ("none", "auto"):
                        problems.append(f"{name}: {prop} on {target} would trap fixed bars inside it (GS-23)")
                    if CONTAINERS.search(" " + last) and prop in ("animation", "animation-name"):
                        for frame, text in frames.items():
                            if re.search(rf"\b{frame}\b", value) and re.search(r"\b(?:transform|translate|scale|rotate|filter)\b", text):
                                problems.append(f"{name}: {target} animates {frame}, which moves it (views only fade)")
                if prop in ("animation", "animation-iteration-count") and "infinite" in value:
                    allowed = ('[aria-busy="true"]' in selector) or (".rec-dot" in selector and "pill--busy" not in selector)
                    if not allowed:
                        problems.append(f"{name}: an infinite animation on {selector} (only a busy spinner and the recording dot loop)")
    text = "\n".join(ui._strip_css_comments(Path(p).read_text(encoding="utf-8")) for p in files)
    if "backdrop-filter" in text:
        if not re.search(r"@supports\s+not\s*\(\(?\s*backdrop-filter", text):
            problems.append("no @supports not (backdrop-filter) opaque fallback")
        if "prefers-reduced-transparency" not in text:
            problems.append("no prefers-reduced-transparency opaque fallback")
    return problems


def audit_nesting(files):
    """No CSS nesting anywhere (the audits parse flat rules)."""
    problems = []
    for path in files:
        for selector, body, _chain in ui.css_rules(path):
            if selector.startswith("@keyframes"):
                continue
            for prop, value in body:
                if "{" in prop or "{" in value or "}" in value or prop.startswith("&"):
                    problems.append(f"{Path(path).name}: nested rule inside {selector}")
                    break
    return problems


VIEW_SCOPES = {
    "reader": ("reader", ("rd-",)), "reader-player": ("reader", ("rdp-",)), "speak": ("speak", ("sp-",)),
    "speak-live": ("speak", ("spl-",)), "transcribe": ("transcribe", ("tr-",)), "talk": ("talk", ("tk-",)), "voices": ("voices", ("vo-",)),
    "studio": ("studio", ("st-",)), "history": ("history", ("hi-",)), "models": ("models", ("mo-",)), "settings": ("settings", ("se-",)),
}


def audit_view_css(files):
    """View CSS lives inside @layer views, and every selector starts with [data-view="<id>"] or the view prefix.
    The one exception is a view's forced-colours block: `@layer utilities { @media (forced-colors: active) {…} }`,
    the last layer, where audit_forced_colours requires every forced-colours rule; its selectors are
    view-scoped like the rest."""
    problems = []
    for path in files:
        path = Path(path)
        scope = VIEW_SCOPES.get(path.stem)
        if scope is None:
            problems.append(f"{path.name}: not a known view stylesheet")
            continue
        view, prefixes = scope
        text = ui._strip_css_comments(path.read_text(encoding="utf-8"))
        if "@layer views" not in text:
            problems.append(f"{path.name}: no @layer views block")
        for selector, _body, chain in ui.css_rules(path):
            if selector.startswith("@") and not selector.startswith(("@keyframes", "@font-face")):
                continue
            forced = bool(chain) and chain[0] == "@layer utilities" and any("forced-colors" in c for c in chain[1:])
            if forced and selector.startswith("@keyframes"):
                problems.append(f"{path.name}: {selector} belongs in @layer views, not in the forced-colours block")
                continue
            if chain and chain[0] == "@layer utilities" and not forced:
                problems.append(f"{path.name}: {selector} is in @layer utilities outside @media (forced-colors: active); only a forced-colours block may sit there")
            elif not forced and (not chain or chain[0] != "@layer views"):
                problems.append(f"{path.name}: {selector} is outside @layer views")
            if selector.startswith("@keyframes"):
                name = selector.split()[1]
                if not name.startswith(prefixes):
                    problems.append(f"{path.name}: @keyframes {name} needs the view prefix")
                continue
            for part in selector.split(","):
                p = part.strip()
                if not (p.startswith(f'[data-view="{view}"]') or re.match(rf"[.#](?:{'|'.join(map(re.escape, prefixes))})", p)):
                    problems.append(f"{path.name}: {p} is not scoped to [data-view=\"{view}\"] or the {prefixes[0]} prefix")
    return problems


def _decls(path, selector, chain_has=""):
    """The declarations for one exact selector: outside any @media/@supports by default, or inside the at-rule
    whose prelude contains chain_has."""
    out = {}
    for sel, body, chain in ui.css_rules(path):
        conditional = [c for c in chain if c.startswith(("@media", "@supports"))]
        wanted = any(chain_has in c for c in conditional) if chain_has else not conditional
        if sel.strip() == selector and wanted:
            out.update(dict(body))
    return out


def audit_focus(files):
    """Focus is visible on every interactive element (AX-04): a :focus-visible ring from --focus in base.css,
    a visually hidden file input lends its ring to its label, and outline: none appears only where a
    replacement ring is drawn (or on non-control focus targets: main and the view heading)."""
    problems = []
    ring = _decls(CSS_BASE, ":focus-visible")
    if "var(--focus)" not in ring.get("outline", "") or "--focus-width" not in ring.get("outline", ""):
        problems.append("base.css has no :focus-visible { outline: var(--focus-width) solid var(--focus) }")
    base_selectors = {s.strip() for s, _, _ in ui.css_rules(CSS_BASE)}
    if not any(".visually-hidden:focus-visible + label" in s for s in base_selectors):
        problems.append("no focus ring for the label of a visually hidden file input")
    replacements = {
        ".slider:focus-visible": ".slider:focus-visible::-webkit-slider-thumb",
        ".seek:focus-visible": ".seek:focus-visible::-webkit-slider-thumb",
        ".row__main:focus-visible": ".row__main:focus-visible::after",
        ".card__stretch:focus-visible": ".card--interactive:focus-within",
        ".menu__item:hover, .menu__item:focus": ".menu__item:focus-visible",
    }
    non_controls = {"main:focus, [data-view-heading]:focus, .sheet h2:focus"}
    all_selectors = {s.strip() for p in files for s, _, _ in ui.css_rules(p)}
    for path in files:
        for selector, body, _chain in ui.css_rules(path):
            for prop, value in body:
                if prop == "outline" and value.strip() in ("none", "0", "0px") and ":focus" in selector:
                    s = selector.strip()
                    if s in non_controls:
                        continue
                    if s not in replacements or replacements[s] not in all_selectors:
                        problems.append(f"{Path(path).name}: {s} removes the focus ring without a replacement")
    return problems


OPACITY_OK = re.compile(r"disabled|aria-disabled|^\.seg__input$|^\.switch__input$|::placeholder|aria-busy")


def audit_no_dimmed_text(files):
    """Text keeps its contrast: opacity below 1 only on disabled states and on the invisible native inputs of
    the segmented control and the switch (AX-07); the visually-hidden utility exists for announced-only text."""
    problems = []
    for path in files:
        for selector, body, _chain in ui.css_rules(path):
            if selector.startswith("@keyframes"):
                continue
            for prop, value in body:
                if prop == "opacity":
                    try:
                        number = float(value.strip())
                    except ValueError:
                        continue
                    if number < 1 and not all(OPACITY_OK.search(part.strip()) for part in selector.split(",")):
                        problems.append(f"{Path(path).name}: {selector} dims with opacity {value}")
    vh = _decls(CSS_BASE, ".visually-hidden")
    if "clip-path" not in vh or "position" not in vh:
        problems.append("no .visually-hidden utility in base.css")
    return problems


def audit_placement(files):
    """The colour placement rules in use (REV-15): --rec never colours text, --text-3 never sits in a rule that
    paints a tinted background, and a selected active option keeps accent-soft with an inset outline instead of
    surface-3."""
    problems = []
    tinted = re.compile(r"var\(--(?:accent-soft|success-soft|warning-soft|danger-soft|info-soft|hl-sentence|rd-hl)\)")
    for path in files:
        for selector, body, _chain in ui.css_rules(path):
            d = dict(body)
            if "var(--rec)" in d.get("color", ""):
                problems.append(f"{Path(path).name}: {selector} uses --rec as a text colour")
            background = d.get("background", "") + d.get("background-color", "")
            if tinted.search(background) and "var(--text-3)" in d.get("color", ""):
                problems.append(f"{Path(path).name}: {selector} puts --text-3 on a tinted background")
            if "accent-soft" in background and "surface-3" in background:
                problems.append(f"{Path(path).name}: {selector} mixes accent-soft and surface-3")
    active = _decls(CSS_COMPONENTS, '.option[aria-selected="true"]:hover, .option[aria-selected="true"].is-active')
    if "var(--accent-soft)" not in active.get("background", "") or "var(--border-strong)" not in active.get("box-shadow", ""):
        problems.append("a selected active option must keep accent-soft and add an inset --border-strong outline")
    return problems


def audit_safe_areas(files):
    """GS-25: safe-area insets on the top bar, bottom bar, mini-player, sheets' footers, the main gutters and the
    toasts."""
    problems = []
    needs = {".topbar": False, ".bottombar": False, ".miniplayer": False, ".sheet__foot": False, "main": False, ".toasts": False}
    for path in files:
        for selector, body, _chain in ui.css_rules(path):
            if any("safe-area-inset" in value or "--bottombar-h" in value for _, value in body):
                for key in needs:
                    if selector.strip() == key:
                        needs[key] = True
    tokens = ui.tokens("light")
    if "safe-area-inset-bottom" not in tokens.get("--bottombar-h", ""):
        problems.append("--bottombar-h does not include the bottom safe area")
    problems += [f"{key} ignores the safe-area insets" for key, ok in needs.items() if not ok]
    return problems


def audit_sheets_and_targets():
    """AX-09: phone dialogs are full-width bottom sheets (centred from 600 px), and touch targets are 44 px."""
    problems = []
    sheet = _decls(CSS_COMPONENTS, ".sheet")
    if sheet.get("width") != "100%" or not sheet.get("margin", "").startswith("auto 0 0") or "0 0" not in sheet.get("border-radius", ""):
        problems.append(".sheet is not a full-width bottom sheet on phones")
    wide = _decls(CSS_COMPONENTS, ".sheet", "min-width: 600px")
    if "560px" not in wide.get("width", "") or wide.get("margin") != "auto":
        problems.append(".sheet is not centred at 600 px and up")
    coarse = {}
    for selector, body, chain in ui.css_rules(CSS_COMPONENTS):
        if any("pointer: coarse" in c for c in chain):
            for part in selector.split(","):
                coarse.setdefault(part.strip(), {}).update(dict(body))
    for target in (".btn", ".chip", ".menu__item", ".input", ".select", ".banner__details > summary"):
        if coarse.get(target, {}).get("min-height") != "44px":
            problems.append(f"{target} is not 44 px tall on touch screens")
    if coarse.get(".icon-btn", {}).get("width") != "44px":
        problems.append(".icon-btn is not 44 × 44 on touch screens")
    return problems


def audit_motion_and_modes():
    """The reduced-motion kill switch (OS setting unless Full, and data-motion=reduce), the WCAG text-spacing
    overrides, [hidden] winning, and the forced-colours block with Highlight for the spoken sentence."""
    problems = []
    rules = ui.css_rules(CSS_BASE)
    hidden = _decls(CSS_BASE, "[hidden]")
    if hidden.get("display", "").replace(" ", "") != "none!important":
        problems.append("[hidden] { display: none !important } is missing")
    def kills(selector_fragment, chain_fragment):
        for selector, body, chain in rules:
            if selector_fragment in selector and (not chain_fragment or any(chain_fragment in c for c in chain)):
                d = dict(body)
                # Transitions are opacity-only under reduced motion: a layout property never starts a 1 ms transition from its
                # first style, which a headless capture could keep.
                opacity_only = d.get("transition-property", "").replace(" ", "") == "opacity!important"
                if "!important" in d.get("animation-duration", "") and "!important" in d.get("transition-duration", "") and d.get("animation-iteration-count", "").startswith("1") and opacity_only:
                    return True
        return False
    if not kills('html:not([data-motion="full"])', "prefers-reduced-motion: reduce"):
        problems.append("no reduced-motion kill switch under prefers-reduced-motion (unless data-motion=full)")
    if not kills('html[data-motion="reduce"]', ""):
        problems.append("no kill switch for html[data-motion=reduce]")
    spacing = _decls(CSS_BASE, 'html[data-spacing="wcag"] *')
    if not ("1.5" in spacing.get("line-height", "") and ".12em" in spacing.get("letter-spacing", "") and ".16em" in spacing.get("word-spacing", "")):
        problems.append("the WCAG 1.4.12 text-spacing overrides are missing")
    if "2em" not in _decls(CSS_BASE, 'html[data-spacing="wcag"] p').get("margin-block-end", ""):
        problems.append("the text-spacing paragraph spacing is missing")
    spoken = _decls(CSS_BASE, ".rd-s.is-now", "forced-colors: active")
    if spoken.get("background") != "Highlight" or spoken.get("color") != "HighlightText":
        problems.append("forced colours: the spoken sentence must use Highlight / HighlightText")
    focus = _decls(CSS_BASE, ":focus-visible", "forced-colors: active")
    if "Highlight" not in focus.get("outline", ""):
        problems.append("forced colours: focus must use Highlight")
    return problems


def audit_no_mark_colors(files):
    """REV-14: Mark/MarkText are not reliable in forced-colours engines; nothing uses them."""
    problems = []
    for path in files:
        text = ui._strip_css_comments(Path(path).read_text(encoding="utf-8"))
        if re.search(r"\bMark(?:Text)?\b", text):
            problems.append(f"{Path(path).name}: uses Mark/MarkText")
    return problems


def audit_reader_tokens(path=CSS_TOKENS):
    """TH-03: the Reader uses the app's tokens: the reading tokens default to the app ones, the paper themes are
    the only other source, and they live in tokens.css."""
    problems = []
    light = dict(next(b for s, b, ch in ui.css_rules(path) if s.strip() == ":root" and len(ch) == 1))
    expected = {"--rd-bg": "var(--bg)", "--rd-text": "var(--text)", "--rd-meta": "var(--text-2)", "--rd-accent": "var(--accent)", "--rd-hl": "var(--hl-sentence)"}
    for name, value in expected.items():
        if light.get(name) != value:
            problems.append(f"{name} should default to {value}")
    for paper in PAPERS:
        if not _decls(path, f'[data-rd-paper="{paper}"]'):
            problems.append(f"no {paper} paper theme in tokens.css")
    return problems


def _css_fixture(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def test_token_set_is_complete_and_dark_blocks_are_identical():
    """The colour token set exists in light and dark with its exact values; the OS-dark and forced-dark
    blocks (and their more-contrast twins) are identical; color-scheme is declared for both themes."""
    assert audit_token_set() == []


def test_token_audit_catches_drift(tmp_path):
    text = CSS_TOKENS.read_text(encoding="utf-8")
    forced = text.index(':root[data-theme="dark"] {')
    broken = text[:forced] + text[forced:].replace("--text-2: #A9AFBC;", "--text-2: #9A9AB2;", 1)
    problems = audit_token_set(_css_fixture(tmp_path, "tokens.css", broken))
    assert any("the two dark blocks differ" in p for p in problems)
    assert any("dark: --text-2" in p for p in problems)


def test_text_and_ui_contrast_pass_in_both_themes_and_more_contrast():
    """[BUG-24][TH-05] Every colour pair passes, recomputed from tokens.css: text 4.5:1 (state colours, muted
    --text-2/--text-3 on every plain surface, white on fills and their hovers), UI 3:1 (control borders, focus,
    the record dot, blend segments), in light, dark and both more-contrast variants."""
    problems = []
    for theme in ("light", "dark"):
        for mode in (False, True):
            problems += [f"{theme}{' more-contrast' if mode else ''}: {p}" for p in failing(contrast_pairs(theme, mode))]
    assert problems == []


def test_placement_rule_pairs_are_computed_composited_and_enforced():
    """[REV-15] The translucent pairs are composited before measuring: --text-2 on accent-soft over surface-3
    passes (5.51), while the pairs the placement rules forbid really fail (so the rules matter): --text-3 on
    accent-soft over s2/s3, --accent on accent-soft over s3, and dark --rec as text on s2/s3. The CSS never uses
    them that way."""
    light, dark = ui.tokens("light"), ui.tokens("dark")
    assert round(ui.contrast(light["--text-2"], light["--accent-soft"], over=light["--surface-3"], table=light), 2) == 5.51
    forbidden = [
        ui.contrast(light["--text-3"], light["--accent-soft"], over=light["--surface-2"], table=light),
        ui.contrast(light["--text-3"], light["--accent-soft"], over=light["--surface-3"], table=light),
        ui.contrast(dark["--text-3"], dark["--accent-soft"], over=dark["--surface-3"], table=dark),
        ui.contrast(light["--accent"], light["--accent-soft"], over=light["--surface-3"], table=light),
        ui.contrast(dark["--accent"], dark["--accent-soft"], over=dark["--surface-3"], table=dark),
        ui.contrast(dark["--rec"], dark["--surface-3"]),
    ]
    assert all(ratio < 4.5 for ratio in forbidden), [round(r, 2) for r in forbidden]
    assert audit_placement(ui.css_files()) == []


def test_placement_audit_catches_misuse(tmp_path):
    bad = _css_fixture(tmp_path, "bad.css", ".a { color: var(--rec); }\n.b { background: var(--accent-soft); color: var(--text-3); }\n.c { background: color-mix(in srgb, var(--accent-soft), var(--surface-3)); }")
    problems = audit_placement([bad])
    assert len([p for p in problems if "bad.css" in p]) == 3


def test_reading_paper_themes_keep_contrast():
    """[TH-03] The Reader uses the app's tokens (the reading tokens default to them), and each paper theme keeps
    text, meta and accent at 4.5:1, text on the highlight at 4.5:1 and the underline at 3:1."""
    assert audit_reader_tokens() == []
    assert failing(paper_pairs()) == []


def test_no_colour_literals_outside_tokens(tmp_path):
    """[BUG-24] No hard-coded colours outside tokens.css: hex, rgb()/hsl() and named colours are tokens; system
    colours appear only inside forced-colours blocks."""
    assert audit_color_literals(ui.css_files()) == []
    bad = _css_fixture(tmp_path, "bad.css", ".a { color: #9a9ab2; }\n.b { background: rgb(0 0 0 / .4); }\n.c { border: 1px solid white; }\n.d { color: Highlight; }\n@media (forced-colors: active) { .e { color: Highlight; } }\n.f { color: var(--text-2); content: \"#fff\"; }")
    problems = audit_color_literals([bad])
    assert len(problems) == 4, problems


def test_type_uses_rem_and_fields_stay_at_1rem(tmp_path):
    """No px font sizes; inputs, selects and textareas never go below 1rem (no iOS zoom on focus)."""
    assert audit_type(ui.css_files()) == []
    bad = _css_fixture(tmp_path, "bad.css", ".a { font-size: 13px; }\n.input--tiny, .x { font-size: .875rem; }\n.input { font-size: var(--fs-sm); }\ntextarea { font-size: var(--fs-base); }")
    problems = audit_type([bad])
    assert sum("px" in p for p in problems) == 1
    assert sum("below 1rem" in p for p in problems) == 1


def test_fixed_bars_have_no_transformed_ancestors_and_motion_stays_in_the_allowlists(tmp_path):
    """[GS-23][REV-26] No transform, filter or backdrop-filter on main, .view, .app or a view section (the
    Reader's fixed transport and top bar stay viewport-fixed; views only fade in); backdrop-filter only on the five
    bars, with opaque fallbacks; infinite animations only for a busy spinner and the recording dot (the busy pill
    is static)."""
    assert audit_motion_allowlists(ui.css_files()) == []
    bad = _css_fixture(tmp_path, "bad.css", "@keyframes slide { from { transform: translateY(8px); } }\nmain { transform: translateZ(0); }\n.view { animation: slide 1s; }\n[data-view=\"reader\"] { filter: blur(2px); }\n.card { backdrop-filter: blur(8px); }\n.pill--busy .pill__dot { animation: pulse 1s infinite; }")
    problems = audit_motion_allowlists([bad])
    assert any("transform on main" in p for p in problems)
    assert any("animates slide" in p for p in problems)
    assert any('filter on [data-view="reader"]' in p for p in problems)
    assert any("backdrop-filter on .card" in p for p in problems)
    assert any("infinite animation on .pill--busy" in p for p in problems)


def test_reduced_motion_text_spacing_hidden_and_forced_colours():
    """[TH-06][REV-14] The reduced-motion kill switch (the OS setting unless Full, and data-motion=reduce), the
    WCAG 1.4.12 text-spacing overrides, [hidden] winning over every display rule, and forced colours with
    Highlight/HighlightText for the spoken sentence and focus (never Mark/MarkText)."""
    assert audit_motion_and_modes() == []
    assert audit_no_mark_colors(ui.css_files()) == []


def test_mark_colours_are_flagged(tmp_path):
    bad = _css_fixture(tmp_path, "bad.css", "@media (forced-colors: active) { .rd-s.is-now { background: Mark; color: MarkText; } }")
    assert audit_no_mark_colors([bad]) != []


def test_focus_is_always_visible(tmp_path):
    """[AX-04] A :focus-visible ring from --focus on every interactive element, the label of a visually hidden
    file input shows the ring, and outline: none appears only with a replacement ring."""
    assert audit_focus(ui.css_files()) == []
    bad = _css_fixture(tmp_path, "bad.css", ".btn:focus-visible { outline: none; }")
    assert any("removes the focus ring" in p for p in audit_focus([bad]))


def test_text_is_never_dimmed_with_opacity(tmp_path):
    """[AX-07] Text keeps 4.5:1: no opacity dimming outside disabled states (finished chapters stay readable),
    and the visually-hidden utility exists for announced-only text such as phone percentages."""
    assert audit_no_dimmed_text(ui.css_files()) == []
    bad = _css_fixture(tmp_path, "bad.css", ".rd-chapter.is-done { opacity: .6; }\n.btn:disabled { opacity: .5; }")
    problems = [p for p in audit_no_dimmed_text([bad]) if "bad.css" in p]
    assert len(problems) == 1 and "is-done" in problems[0]


def test_phone_sheets_are_bottom_sheets_with_44px_targets():
    """[AX-09] Phone dialogs are full-width bottom sheets (centred dialogs from 600 px), and every control is
    44 × 44 on a touch screen."""
    assert audit_sheets_and_targets() == []


def test_safe_area_insets_on_the_chrome():
    """[GS-25] Safe-area insets on the top bar, bottom bar, mini-player, sheet footers, main gutters and toasts."""
    assert audit_safe_areas(ui.css_files()) == []


def test_no_css_nesting(tmp_path):
    """CSS nesting is not used anywhere (the audits parse flat rules)."""
    assert audit_nesting(ui.css_files()) == []
    bad = _css_fixture(tmp_path, "bad.css", ".a { color: var(--text); .b { color: var(--text-2); } }")
    assert audit_nesting([bad]) != []


def test_view_css_is_layered_and_scoped(tmp_path):
    """View CSS sits inside @layer views and every selector starts with [data-view="<id>"] or the view prefix."""
    views = sorted((ui.CSS_ROOT / "views").glob("*.css"))
    assert {p.stem for p in views} == set(VIEW_SCOPES)
    assert audit_view_css(views) == []
    (tmp_path / "views").mkdir()
    bad = _css_fixture(tmp_path / "views", "speak.css", ".btn { color: var(--text); }\n@layer views { [data-view=\"speak\"] .sp-x { color: var(--text); } .take { color: var(--text-2); } .sp-ok { color: var(--text); } }")
    problems = audit_view_css([bad])
    assert any("outside @layer views" in p for p in problems)
    assert any(".take is not scoped" in p for p in problems)
    assert len(problems) == 3, problems


def test_component_states_the_views_asked_for():
    """Component states the views rely on: an aria-disabled chip looks disabled
    and takes no hover fill; a disabled segmented control keeps its chosen option's fill and edge (only its label
    dims); the checkbox is a 44 px row with an accent box and a disabled state; on phones (and with
    .banner--stacked at every width) a banner's actions take a line of their own under the message, indented to
    its text, with a dismiss button kept on the first line."""
    chip = _decls(CSS_COMPONENTS, '.chip:disabled, .chip[aria-disabled="true"]')
    assert chip.get("opacity") == ".5"
    for hover in ('.chip:hover:not(:disabled):not([aria-disabled="true"])', '.chip--action:hover:not(:disabled):not([aria-disabled="true"])'):
        assert _decls(CSS_COMPONENTS, hover).get("background"), hover
    seg = _decls(CSS_COMPONENTS, ".seg__input:disabled:checked + span")
    assert seg.get("opacity") == "1" and seg.get("color") == "var(--text-2)"
    assert "box-shadow" not in seg and "background" not in seg, "the chosen pill keeps its fill and edge"
    check = _decls(CSS_COMPONENTS, ".check")
    assert check.get("min-height") == "var(--control-h)" and check.get("display") == "flex" and check.get("align-items") == "flex-start"
    box = _decls(CSS_COMPONENTS, ".check__input")
    assert box.get("accent-color") == "var(--accent-fill)" and box.get("width") == "20px" and box.get("height") == "20px"
    assert _decls(CSS_COMPONENTS, ".check:has(> .check__input:disabled)").get("opacity") == ".5"
    assert "calc(20px + var(--sp-3))" in _decls(CSS_COMPONENTS, ".check__hint").get("margin-inline-start", "")
    for scope, chain in ((".banner", "max-width: 599px"), (".banner--stacked", "")):
        assert _decls(CSS_COMPONENTS, scope, chain).get("flex-wrap") == "wrap", scope
        actions = _decls(CSS_COMPONENTS, f"{scope} .banner__actions", chain)
        assert actions.get("flex-basis") == "100%" and actions.get("order") == "3", scope
        assert actions.get("padding-inline-start") == "calc(20px + var(--sp-3))", scope
        assert _decls(CSS_COMPONENTS, f"{scope} > .icon-btn", chain).get("order") == "2", scope
    assert _decls(CSS_COMPONENTS, ".banner__actions:empty").get("display") == "none"


def _top_level_items(value: str) -> list[str]:
    items, depth, start = [], 0, 0
    for i, ch in enumerate(value):
        depth += ch == "("
        depth -= ch == ")"
        if ch == "," and depth == 0:
            items.append(value[start:i].strip())
            start = i + 1
    items.append(value[start:].strip())
    return items


def audit_shadow_lists(files, token_path=None):
    """A box-shadow list (`var(--shadow-1), 0 0 0 1px var(--edge)`) is void as a whole when one of its tokens
    resolves to `none` in a theme: the card edge, the chosen segment's edge and the slider thumb ring would all
    vanish in dark mode. Every token used in a list must resolve to a shadow in both themes."""
    problems = []
    themes = {theme: ui.tokens(theme, path=token_path) for theme in ("light", "dark")}
    for path in files:
        for selector, body, _chain in ui.css_rules(path):
            for prop, value in body:
                if prop != "box-shadow" or len(_top_level_items(value)) < 2:
                    continue
                for name in re.findall(r"var\(\s*(--[\w-]+)", value):
                    for theme, table in themes.items():
                        if table.get(name, "").strip() == "none":
                            problems.append(f"{Path(path).name}: {selector} {{box-shadow: {value}}} is void in {theme}: {name} is none")
    return problems


def test_shadow_tokens_never_void_a_shadow_list(tmp_path):
    """Dark mode's 1px card edge (dark uses an edge instead of a shadow), the chosen segment's edge
    and the slider thumb ring are box-shadow lists that start with --shadow-1; a token that is `none` there voids
    the whole list. Found in Edge: the chosen segment computed to box-shadow none in
    dark."""
    assert audit_shadow_lists(ui.css_files()) == []
    tokens_bad = _css_fixture(tmp_path, "tokens.css", CSS_TOKENS.read_text(encoding="utf-8").replace("--shadow-1: 0 0 0 0 transparent;", "--shadow-1: none;"))
    sheet = _css_fixture(tmp_path, "c.css", ".card { box-shadow: var(--shadow-1), 0 0 0 1px var(--edge); }\n.play { box-shadow: var(--shadow-1); }")
    problems = audit_shadow_lists([sheet], token_path=tokens_bad)
    assert len(problems) == 1 and ".card" in problems[0] and "dark" in problems[0], problems


def test_a_view_sheet_may_carry_a_scoped_forced_colours_block(tmp_path):
    """[REV-14] A view's own tinted surfaces can get a system-colour edge in forced colours: a view
    sheet may hold `@layer utilities { @media (forced-colors: active) {…} }` with view-scoped selectors, which
    both the view-CSS and the forced-colours audits accept. Anything else in @layer utilities, an unscoped
    forced-colours selector, and a forced-colours rule left in @layer views (a base utilities rule would undo
    it) are still reported."""
    (tmp_path / "views").mkdir()
    good = _css_fixture(tmp_path / "views", "speak-live.css", (
        "@layer views { [data-view=\"speak\"] .spl-now { background: var(--accent-soft); } }\n"
        "@layer utilities { @media (forced-colors: active) {\n"
        "  [data-view=\"speak\"] .spl-now, .spl-now__label { border: 1px solid CanvasText; }\n"
        "} }\n"))
    assert audit_view_css([good]) == []
    assert audit_forced_colours([CSS_BASE, good]) == []
    assert audit_color_literals([good]) == []
    (tmp_path / "bad").mkdir()
    (tmp_path / "bad" / "views").mkdir()
    bad = _css_fixture(tmp_path / "bad" / "views", "speak-live.css", (
        "@layer views { [data-view=\"speak\"] .spl-now { color: var(--text); }\n"
        "  @media (forced-colors: active) { [data-view=\"speak\"] .spl-a { border: 1px solid CanvasText; } } }\n"
        "@layer utilities { [data-view=\"speak\"] .spl-b { color: var(--text); }\n"
        "  @media (forced-colors: active) { .btn { border: 2px solid ButtonText; } } }\n"))
    problems = audit_view_css([bad])
    assert any(".spl-b is in @layer utilities outside @media (forced-colors: active)" in p for p in problems), problems
    assert any(".btn is not scoped" in p for p in problems), problems
    assert len(problems) == 2, problems
    layered = audit_forced_colours([CSS_BASE, bad])
    assert any(".spl-a is a forced-colours rule in @layer views" in p for p in layered), layered


def test_coverage_counts_only_tests_that_run(tmp_path, monkeypatch):
    """An id counts only from a test that runs. Commented-out tests, titles inside string literals,
    it.skip / it.todo, {skip: true} options, tests inside describe.skip, and Python tests marked skip or
    xfail are not coverage."""
    from tests import ui_coverage as cov

    js = tmp_path / "fixture.test.js"
    js.write_text(
        "import { describe, it } from 'node:test';\n"
        "// it('[RE-45] commented out', () => {});\n"
        "/* it('[RE-46] in a block comment', () => {}); */\n"
        "it.skip('[RE-47] skipped', () => {});\n"
        "it.todo('[RE-48] todo');\n"
        "it('[RE-49] skip option', { skip: true }, () => {});\n"
        "const s = \"it('[RE-50] inside a string')\";\n"
        "describe.skip('group', () => { it('[RE-51] inside a skipped describe', () => {}); });\n"
        "describe('real', () => { it('[RE-52] runs', () => {}); it.only('[RE-53] only runs', () => {}); });\n"
        "it(`[RE-54] template title`, () => {});\n",
        encoding="utf-8",
    )
    titles = cov.js_titles(js)
    ids = set().union(*(cov._leading(t) for t in titles))
    assert ids == {"RE-52", "RE-53", "RE-54"}, ids
    py = (
        "import pytest\n"
        "@pytest.mark.skip(reason='later')\n"
        "def test_a():\n    '''[RL-01] skipped'''\n"
        "@pytest.mark.xfail\n"
        "def test_b():\n    '''[RL-02] xfail'''\n"
        "@pytest.mark.skipif(True, reason='x')\n"
        "def test_c():\n    '''[RL-03] skipif True'''\n"
        "def test_d():\n    '''[RL-04] runs'''\n"
    )
    ids = set().union(*(cov._leading(t) for t in cov.py_titles(py)))
    assert ids == {"RL-04"}, ids
    assert cov.py_titles("import pytest\npytestmark = pytest.mark.skip\ndef test_x():\n    '''[RL-05] x'''\n") == []


def audit_forced_colours(files, base=None):
    """Forced colours hold against every component rule: every forced-colors rule sits in the last layer
    (utilities), and the states that otherwise vanish (switch on/off, the chosen segment, the selected and the
    active listbox option, focus, sheet and card edges) have their own system-colour rules there."""
    base = base or CSS_BASE
    problems = []
    layers = re.search(r"@layer\s+([\w\s,]+);", CSS_TOKENS.read_text(encoding="utf-8")).group(1).split(",")
    last = layers[-1].strip()
    seen = {}
    for path in files:
        for selector, body, chain in ui.css_rules(path):
            if not any("forced-colors" in c for c in chain):
                continue
            layer = next((c.split()[1] for c in chain if c.startswith("@layer")), "")
            if layer != last:
                problems.append(f"{Path(path).name}: {selector.strip()} is a forced-colours rule in @layer {layer or '(none)'}; it must be in @layer {last}, or a later layer's component rule undoes it")
            for part in selector.split(","):
                seen.setdefault(part.strip(), {}).update(dict(body))
    needs = {
        ".switch__input:checked + .switch__track": ("background", "Highlight"),
        ".switch__track::after": ("background", "ButtonText"),
        ".seg__input:checked + span": ("background", "Highlight"),
        '.option[aria-selected="true"]': ("background", "Highlight"),
        ".option.is-active": ("outline", "dashed"),
        ":focus-visible": ("outline", "Highlight"),
        ".card--interactive:focus-within": ("outline", "Highlight"),
        ".sheet": ("border", "CanvasText"),
        ".chip": ("border", "ButtonText"),
        ".icon-btn": ("border", "ButtonText"),
    }
    for selector, (prop, word) in needs.items():
        if word not in seen.get(selector, {}).get(prop, ""):
            problems.append(f"forced colours: {selector} needs {prop} with {word}")
    return problems


def test_forced_colours_live_in_the_last_layer_and_cover_every_state(tmp_path):
    """[REV-14] In Windows High Contrast every state stays visible: the forced-colours block is in the last layer
    (a components-layer `border: 0` or `outline: none` can no longer win), switches, the chosen segment, the
    selected and active options, focus rings, sheets and cards all have system-colour rules."""
    assert audit_forced_colours(ui.css_files()) == []
    bad = _css_fixture(tmp_path, "bad.css", "@layer base { @media (forced-colors: active) { .chip { border: 1px solid ButtonText; } } }")
    problems = audit_forced_colours([bad])
    assert any("@layer base" in p for p in problems)
    assert any(".switch__input:checked + .switch__track" in p for p in problems)


def test_fields_keep_their_label_and_short_screens_keep_room():
    """A short field next to a tall one keeps its label on its control (.field aligns to the start), and on
    short screens the voice picker scrolls as one body while the chrome below 960 px gets compact."""
    field = _decls(CSS_COMPONENTS, ".field")
    assert field.get("align-content") == "start"
    assert _decls(CSS_COMPONENTS, ".form-grid").get("align-items") == "start"
    shell = ui.CSS_ROOT / "shell.css"
    vp = _decls(shell, ".vp .sheet__body > .vp__list", "max-height: 640px")
    assert vp.get("overflow") == "visible" and vp.get("flex") == "none"
    assert _decls(shell, ".vp__search", "max-height: 640px").get("position") == "sticky"
    assert _decls(shell, ".topbar", "max-height: 520px").get("position") == "static"
    assert "48px" in _decls(shell, ":root", "max-height: 520px").get("--bottombar-h", "")
    assert _decls(shell, ".rail .nav-link", "pointer: coarse").get("height") == "44px"
