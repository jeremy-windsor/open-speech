"""EPUB 2 and 3 books (DRM-free). Read with the standard library only."""

from __future__ import annotations

import io
import posixpath
import zipfile
import zlib
from urllib.parse import unquote
from xml.etree import ElementTree

from src.reader.document import HEADING, Block, Chapter, Document, normalize_space
from src.reader.sources.html import ParsedPage, parse_page

_CONTAINER_NS = "{urn:oasis:names:tc:opendocument:xmlns:container}"
_OPF_NS = "{http://www.idpf.org/2007/opf}"
_DC_NS = "{http://purl.org/dc/elements/1.1/}"
_NCX_NS = "{http://www.daisy.org/z3986/2005/ncx/}"
_XHTML_NS = "{http://www.w3.org/1999/xhtml}"
_EPUB_NS = "{http://www.idpf.org/2007/ops}"
_MAX_MEMBERS = 5000
# Contents entries this deep or shallower start a chapter (0 = top level, 1 = e.g. chapters inside parts).
TOC_DEPTH = 1


def _read(zf: zipfile.ZipFile, name: str) -> bytes:
    try:
        return zf.read(name)
    except KeyError as exc:
        raise ValueError(f"EPUB is missing {name}") from exc
    except (zipfile.BadZipFile, NotImplementedError, RuntimeError, zlib.error, EOFError) as exc:
        raise ValueError(f"EPUB file {name} is corrupt or uses unsupported compression") from exc


def _xml(data: bytes, name: str) -> ElementTree.Element:
    try:
        return ElementTree.fromstring(data)
    except ElementTree.ParseError as exc:
        raise ValueError(f"EPUB file {name} is not valid XML") from exc


def _resolve(base: str, href: str) -> str:
    path = unquote(href.split("#", 1)[0])
    return posixpath.normpath(posixpath.join(base, path)) if path else ""


def _toc_entries(
    zf: zipfile.ZipFile, opf_dir: str, manifest: dict[str, dict], toc_id: str | None
) -> list[tuple[str, str, str, int]]:
    """Table-of-contents entries in order, as (content file, fragment id, label, nesting depth)."""
    nav = next((item for item in manifest.values() if "nav" in item["properties"].split()), None)
    if nav:
        nav_path = _resolve(opf_dir, nav["href"])
        nav_dir = posixpath.dirname(nav_path)
        root = _xml(_read(zf, nav_path), nav_path)
        entries: list[tuple[str, str, str, int]] = []

        def walk(ol: ElementTree.Element, depth: int) -> None:
            for li in ol.findall(f"{_XHTML_NS}li"):
                link = li.find(f"{_XHTML_NS}a")
                if link is not None:
                    href = link.get("href", "")
                    target = _resolve(nav_dir, href)
                    label = normalize_space("".join(link.itertext()))
                    if target and label:
                        entries.append((target, _fragment(href), label, depth))
                for child in li.findall(f"{_XHTML_NS}ol"):
                    walk(child, depth + 1)

        for element in root.iter(f"{_XHTML_NS}nav"):
            if element.get(f"{_EPUB_NS}type") not in (None, "toc"):
                continue
            top = element.findall(f"{_XHTML_NS}ol") or [next(element.iter(f"{_XHTML_NS}ol"), None)]
            for ol in top:
                if ol is not None:
                    walk(ol, 0)
        if entries:
            return entries
    ncx = manifest.get(toc_id or "")
    if not ncx:
        return []
    ncx_path = _resolve(opf_dir, ncx["href"])
    ncx_dir = posixpath.dirname(ncx_path)
    root = _xml(_read(zf, ncx_path), ncx_path)
    entries = []

    def walk_ncx(parent: ElementTree.Element, depth: int) -> None:
        for point in parent.findall(f"{_NCX_NS}navPoint"):
            content = point.find(f"{_NCX_NS}content")
            label = point.find(f"{_NCX_NS}navLabel/{_NCX_NS}text")
            if content is not None and label is not None:
                src = content.get("src", "")
                target = _resolve(ncx_dir, src)
                text = normalize_space(label.text or "")
                if target and text:
                    entries.append((target, _fragment(src), text, depth))
            walk_ncx(point, depth + 1)

    nav_map = root.find(f"{_NCX_NS}navMap")
    if nav_map is not None:
        walk_ncx(nav_map, 0)
    return entries


def _fragment(href: str) -> str:
    return unquote(href.split("#", 1)[1]) if "#" in href else ""


def _chapters_from_toc(files: list[tuple[str, ParsedPage]], entries: list[tuple[str, str, str, int]]) -> list[Chapter]:
    """Cut the whole book at its contents entries, so a file holding several chapters is split.
    A file the contents never point into continues the chapter before it, unless it opens with a
    heading. Empty when no entry can be placed."""
    blocks: list[Block] = []
    captions: dict[int, str] = {}
    positions: dict[tuple[str, str], int] = {}
    file_starts: list[tuple[str, int]] = []
    for path, page in files:
        offset = len(blocks)
        positions.setdefault((path, ""), offset)
        for anchor, index in page.anchors.items():
            positions.setdefault((path, anchor), offset + index)
        captions.update({offset + index: caption for index, caption in page.heading_captions.items()})
        file_starts.append((path, offset))
        blocks.extend(block for block, _level in page.blocks)
    starts: dict[int, str] = {}
    listed: set[str] = set()
    for path, fragment, label, depth in entries:
        index = positions.get((path, fragment))
        if depth <= TOC_DEPTH and index is not None and index < len(blocks):
            starts.setdefault(index, label)
            listed.add(path)
    if not starts:
        return []
    for path, start in file_starts:
        if path not in listed and start < len(blocks) and blocks[start].kind == HEADING:
            starts.setdefault(start, blocks[start].text)
    cuts = sorted(starts)
    chapters: list[Chapter] = []
    if cuts[0] > 0:
        leading = blocks[:cuts[0]]
        chapters.append(Chapter(next((b.text for b in leading if b.kind == HEADING), ""), leading))
    for number, start in enumerate(cuts):
        end = cuts[number + 1] if number + 1 < len(cuts) else len(blocks)
        label, first = starts[start], blocks[start]
        if first.kind == HEADING and _squash(label) in (_squash(first.text), _squash(captions.get(start, "") + first.text)):
            label = first.text   # the label is that heading (plus its picture caption); the heading reads better
        chapters.append(_titled(label, blocks[start:end]))
    return chapters


def _squash(text: str) -> str:
    return "".join(text.split()).casefold()


def _titled(label: str, blocks: list[Block]) -> Chapter:
    """A chapter named by its contents label, opening with a spoken heading."""
    if blocks and blocks[0].kind != HEADING:
        blocks = [Block(label, HEADING), *blocks]
    return Chapter(label, blocks)


def parse_epub(data: bytes, *, max_uncompressed: int) -> Document:
    try:
        zf = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as exc:
        raise ValueError("Not a valid EPUB (zip) file") from exc
    with zf:
        members = zf.infolist()
        if len(members) > _MAX_MEMBERS or sum(item.file_size for item in members) > max_uncompressed:
            raise ValueError("EPUB is too large when uncompressed")
        encrypted: set[str] = set()
        if "META-INF/encryption.xml" in zf.namelist():
            # Fonts are often obfuscated harmlessly; only encrypted text means DRM.
            encryption = _xml(_read(zf, "META-INF/encryption.xml"), "META-INF/encryption.xml")
            encrypted = {
                _resolve("", element.get("URI", ""))
                for element in encryption.iter()
                if element.tag.endswith("CipherReference")
            }

        container = _xml(_read(zf, "META-INF/container.xml"), "META-INF/container.xml")
        rootfile = container.find(f".//{_CONTAINER_NS}rootfile")
        if rootfile is None or not rootfile.get("full-path"):
            raise ValueError("EPUB container does not name a package file")
        opf_path = rootfile.get("full-path")
        opf_dir = posixpath.dirname(opf_path)
        package = _xml(_read(zf, opf_path), opf_path)

        metadata = package.find(f"{_OPF_NS}metadata")
        title = author = ""
        if metadata is not None:
            title = normalize_space(metadata.findtext(f"{_DC_NS}title") or "")
            author = normalize_space(metadata.findtext(f"{_DC_NS}creator") or "")

        manifest = {
            item.get("id", ""): {
                "href": item.get("href", ""),
                "media_type": item.get("media-type", ""),
                "properties": item.get("properties", ""),
            }
            for item in package.iter(f"{_OPF_NS}item")
        }
        spine = package.find(f"{_OPF_NS}spine")
        if spine is None:
            raise ValueError("EPUB has no reading order (spine)")
        try:
            toc = _toc_entries(zf, opf_dir, manifest, spine.get("toc"))
        except ValueError:
            toc = []  # a broken table of contents falls back to one chapter per file

        files: list[tuple[str, ParsedPage]] = []
        for itemref in spine.iter(f"{_OPF_NS}itemref"):
            if itemref.get("linear") == "no":
                continue
            item = manifest.get(itemref.get("idref", ""))
            if not item or "html" not in item["media_type"]:
                continue
            path = _resolve(opf_dir, item["href"])
            if path in encrypted:
                raise ValueError("This EPUB is DRM-protected and cannot be read")
            markup = _read(zf, path).decode("utf-8", errors="replace")
            files.append((path, parse_page(markup)))

        chapters = _chapters_from_toc(files, toc)
        if not chapters:
            first_labels: dict[str, str] = {}
            for path, _fragment_id, label, _depth in toc:
                first_labels.setdefault(path, label)
            for path, page in files:
                blocks = [block for block, _level in page.blocks]
                if not blocks:
                    continue
                first_heading = next((block.text for block in blocks if block.kind == HEADING), "")
                chapter_title = first_labels.get(path) or first_heading
                if chapter_title:
                    chapters.append(_titled(chapter_title, blocks))
                else:
                    chapters.append(Chapter(f"Section {len(chapters) + 1}", blocks))

    return Document(title=title, author=author or None, source_type="epub", chapters=chapters)
