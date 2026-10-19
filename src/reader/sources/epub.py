"""EPUB 2 and 3 books (DRM-free). Read with the standard library only."""

from __future__ import annotations

import io
import posixpath
import zipfile
import zlib
from urllib.parse import unquote
from xml.etree import ElementTree

from src.reader.document import HEADING, Block, Chapter, Document, normalize_space
from src.reader.sources.html import html_to_blocks

_CONTAINER_NS = "{urn:oasis:names:tc:opendocument:xmlns:container}"
_OPF_NS = "{http://www.idpf.org/2007/opf}"
_DC_NS = "{http://purl.org/dc/elements/1.1/}"
_NCX_NS = "{http://www.daisy.org/z3986/2005/ncx/}"
_XHTML_NS = "{http://www.w3.org/1999/xhtml}"
_EPUB_NS = "{http://www.idpf.org/2007/ops}"
_MAX_MEMBERS = 5000


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


def _toc_titles(zf: zipfile.ZipFile, opf_dir: str, manifest: dict[str, dict], toc_id: str | None) -> dict[str, str]:
    """Map each content file to the first table-of-contents title that points at it."""
    titles: dict[str, str] = {}
    nav = next((item for item in manifest.values() if "nav" in item["properties"].split()), None)
    if nav:
        nav_path = _resolve(opf_dir, nav["href"])
        root = _xml(_read(zf, nav_path), nav_path)
        for element in root.iter(f"{_XHTML_NS}nav"):
            if element.get(f"{_EPUB_NS}type") not in (None, "toc"):
                continue
            for link in element.iter(f"{_XHTML_NS}a"):
                target = _resolve(posixpath.dirname(nav_path), link.get("href", ""))
                label = normalize_space("".join(link.itertext()))
                if target and label:
                    titles.setdefault(target, label)
        if titles:
            return titles
    ncx = manifest.get(toc_id or "")
    if ncx:
        ncx_path = _resolve(opf_dir, ncx["href"])
        root = _xml(_read(zf, ncx_path), ncx_path)
        for point in root.iter(f"{_NCX_NS}navPoint"):
            content = point.find(f"{_NCX_NS}content")
            label = point.find(f"{_NCX_NS}navLabel/{_NCX_NS}text")
            if content is None or label is None:
                continue
            target = _resolve(posixpath.dirname(ncx_path), content.get("src", ""))
            text = normalize_space(label.text or "")
            if target and text:
                titles.setdefault(target, text)
    return titles


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
            toc_titles = _toc_titles(zf, opf_dir, manifest, spine.get("toc"))
        except ValueError:
            toc_titles = {}  # a broken table of contents falls back to each chapter's first heading

        chapters: list[Chapter] = []
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
            _page_title, blocks = html_to_blocks(markup)
            if not blocks:
                continue
            first_heading = next((block.text for block, _level in blocks if block.kind == HEADING), "")
            chapter_title = toc_titles.get(path) or first_heading
            chapter_blocks = [block for block, _level in blocks]
            if chapter_title and chapter_blocks[0].kind != HEADING:
                chapter_blocks.insert(0, Block(chapter_title, HEADING))
            chapters.append(Chapter(chapter_title or f"Section {len(chapters) + 1}", chapter_blocks))

    return Document(title=title, author=author or None, source_type="epub", chapters=chapters)
