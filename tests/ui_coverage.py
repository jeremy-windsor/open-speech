"""Must-preserve coverage: every inventory id in tests/ui_must_preserve.json and every defect with a test file
must appear as "[ID]" at the start of a test title in its assigned file.

Usage: python -m tests.ui_coverage [--owner F1] [--all]
--owner picks one area group (the "owner" field of each entry, such as F1 for the core modules or V1 for the Reader).
Prints the missing ids (one per line, nothing when complete) and exits 1 when something is missing.
Test titles: JS `it('[RE-45] …')` / `test(…)` / `describe(…)`; Python test docstrings that start with
"[ID]". Several ids may lead one title: "[AU-03][GS-26] …".

Only tests that run count: a title in a comment or inside a string literal, `it.skip` / `it.todo` (and
`test.`/`describe.` alike), an options object with `skip` or `todo`, any test inside a skipped `describe`,
and Python tests marked `skip` / `xfail` (or `skipif(True)`, or a module-level `pytestmark` skip) are not
coverage.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from functools import lru_cache
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MAP_PATH = REPO / "tests" / "ui_must_preserve.json"
ID_RE = re.compile(r"\[([A-Z]{2,3}-\d{2,3})\]")
LEADING_IDS = re.compile(r"^\s*((?:\[[A-Z]{2,3}-\d{2,3}\]\s*)+)")
# A test call in the lexed code (comments blanked, literal contents blanked but their quotes kept).
JS_CALL = re.compile(r"""(?<![\w$.])(it|test|describe)(?:\.(only|skip|todo))?\s*\(\s*(['"`])""")
SKIP_OPTIONS = re.compile(r"""^\s*,\s*\{[^{}]*\b(?:skip|todo)\s*:""")


def load_map(path: Path = MAP_PATH) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _leading(title: str) -> set[str]:
    m = LEADING_IDS.match(title)
    return set(ID_RE.findall(m.group(1))) if m else set()


def _matching_paren(code: str, open_index: int) -> int:
    """The index of the ')' closing the '(' at open_index (literal contents are blanked, so they never count)."""
    depth = 0
    for i in range(open_index, len(code)):
        if code[i] == "(":
            depth += 1
        elif code[i] == ")":
            depth -= 1
            if depth == 0:
                return i
    return len(code)


def js_titles(path: Path) -> list[str]:
    """The titles of the JS tests that run: real calls in code, not skipped or todo, not in a skipped describe."""
    from tests.ui_support import lex_js  # imported here so the CLI stays cheap to import

    src = lex_js(path)
    code, text = src.code, src.text
    calls = []
    for m in JS_CALL.finditer(code):
        quote_at = m.end() - 1
        # The literal's contents are blanked in `code`; its closing quote is the next same quote there.
        close = code.find(m.group(3), quote_at + 1)
        if close < 0:
            continue
        title = text[quote_at + 1:close]
        paren = code.find("(", m.start())
        skipped = m.group(2) in ("skip", "todo") or bool(SKIP_OPTIONS.match(code[close + 1:close + 400]))
        calls.append((m.group(1), title, paren, _matching_paren(code, paren), skipped))
    skipped_ranges = [(start, end) for kind, _, start, end, skipped in calls if kind == "describe" and skipped]
    out = []
    for _kind, title, start, _end, skipped in calls:
        if skipped or any(a < start < b for a, b in skipped_ranges):
            continue
        out.append(title)
    return out


def _mark_name(node: ast.expr) -> tuple[str, list[ast.expr]]:
    """'skip', 'xfail', 'skipif'… for a pytest.mark.<name>[(…)] decorator, with its call arguments."""
    args: list[ast.expr] = []
    if isinstance(node, ast.Call):
        args = list(node.args)
        node = node.func
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    parts.reverse()
    if len(parts) >= 3 and parts[-3:-1] == ["pytest", "mark"] or (len(parts) == 2 and parts[0] == "mark"):
        return parts[-1], args
    return "", args


def _never_runs(decorators: list[ast.expr]) -> bool:
    for deco in decorators:
        name, args = _mark_name(deco)
        if name in ("skip", "xfail"):
            return True
        if name == "skipif" and args and isinstance(args[0], ast.Constant) and args[0].value is True:
            return True
    return False


def py_titles(text: str) -> list[str]:
    """The docstrings of the Python tests that run (skip / xfail marks, and a module pytestmark skip, drop them)."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []
    for stmt in tree.body:
        if isinstance(stmt, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "pytestmark" for t in stmt.targets):
            marks = stmt.value.elts if isinstance(stmt.value, (ast.List, ast.Tuple)) else [stmt.value]
            if _never_runs(list(marks)):
                return []
    out = []
    def visit(body, inherited=False):
        for node in body:
            if isinstance(node, ast.ClassDef):
                visit(node.body, inherited or _never_runs(node.decorator_list))
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test"):
                if inherited or _never_runs(node.decorator_list):
                    continue
                out.append(ast.get_docstring(node) or "")
    visit(tree.body)
    return out


@lru_cache(maxsize=None)
def titles_in(path: str) -> frozenset[str]:
    """The ids that lead the title of a test that runs, in one test file (repo-relative path)."""
    file = REPO / path
    if not file.exists():
        return frozenset()
    ids: set[str] = set()
    if file.suffix == ".js":
        for title in js_titles(file):
            ids |= _leading(title)
    elif file.suffix == ".py":
        for doc in py_titles(file.read_text(encoding="utf-8")):
            ids |= _leading(doc)
    return frozenset(ids)


def any_view_test_files() -> list[str]:
    files = [p.relative_to(REPO).as_posix() for p in (REPO / "tests" / "js").rglob("*.test.js")] if (REPO / "tests" / "js").exists() else []
    files += [p.relative_to(REPO).as_posix() for p in (REPO / "tests").glob("test_ui_*.py")]
    return sorted(files)


def covered(item_id: str, test: str | None) -> bool:
    if not test:
        return True
    if test == "any":
        return any(item_id in titles_in(f) for f in any_view_test_files())
    return item_id in titles_in(test)


def missing(owner: str | None = None, data: dict | None = None) -> list[tuple[str, str, str]]:
    """[(id, test file, 'item'|'defect')] that are not covered yet, for one area group or for all."""
    data = data or load_map()
    out = []
    for item in data["items"]:
        if owner and item.get("owner") != owner:
            continue
        if not covered(item["id"], item.get("test")):
            out.append((item["id"], item.get("test") or "", "item"))
    seen = set()
    for defect in data["defects"]:
        if owner and defect.get("owner") != owner:
            continue
        key = (defect["id"], defect.get("test"))
        if key in seen or not defect.get("test"):
            continue
        seen.add(key)
        if not covered(defect["id"], defect.get("test")):
            out.append((defect["id"], defect["test"], "defect"))
    return out


def counts(owner: str | None = None, data: dict | None = None) -> tuple[int, int]:
    data = data or load_map()
    items = [i for i in data["items"] if not owner or i.get("owner") == owner]
    defects = {d["id"] for d in data["defects"] if d.get("test") and (not owner or d.get("owner") == owner)}
    return len(items), len(defects)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--owner", help="only the ids whose \"owner\" field in the map (its area group) is this (default: all area groups)")
    parser.add_argument("--all", action="store_true", help="every area group (the default when --owner is not given)")
    args = parser.parse_args(argv)
    owner = None if args.all else args.owner
    gaps = missing(owner)
    for item_id, test, kind in gaps:
        print(f"{item_id}\t{kind}\t{test}")
    n_items, n_defects = counts(owner)
    who = owner or "all area groups"
    print(f"{who}: {n_items} ids and {n_defects} defects, {len(gaps)} missing", file=sys.stderr)
    return 1 if gaps else 0


if __name__ == "__main__":
    raise SystemExit(main())
