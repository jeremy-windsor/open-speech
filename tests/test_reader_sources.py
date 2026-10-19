from __future__ import annotations

import io
import time
import zipfile

import pytest

from src.live_tts.segmenter import split_sentences
from src.reader.document import HEADING, MAX_BLOCK_CHARS
from src.reader.sources import parse_pasted, parse_upload
from src.reader.sources.text import CODE_BLOCK_CUE

MB = 1024 * 1024


def _epub(chapters: list[tuple[str, str]], *, nav: bool = True, encrypted: list[str] | None = None) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        zf.writestr("mimetype", "application/epub+zip")
        zf.writestr(
            "META-INF/container.xml",
            '<?xml version="1.0"?><container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">'
            '<rootfiles><rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/></rootfiles></container>',
        )
        manifest = "".join(
            f'<item id="c{i}" href="text/ch{i}.xhtml" media-type="application/xhtml+xml"/>' for i in range(len(chapters))
        )
        spine = "".join(f'<itemref idref="c{i}"/>' for i in range(len(chapters)))
        if nav:
            manifest += '<item id="nav" href="nav.xhtml" media-type="application/xhtml+xml" properties="nav"/>'
            links = "".join(f'<li><a href="text/ch{i}.xhtml">{title}</a></li>' for i, (title, _body) in enumerate(chapters))
            zf.writestr(
                "OEBPS/nav.xhtml",
                '<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops"><body>'
                f'<nav epub:type="toc"><ol>{links}</ol></nav></body></html>',
            )
        zf.writestr(
            "OEBPS/content.opf",
            '<?xml version="1.0"?><package xmlns="http://www.idpf.org/2007/opf" version="3.0">'
            '<metadata xmlns:dc="http://purl.org/dc/elements/1.1/"><dc:title>Tiny Book</dc:title>'
            "<dc:creator>A. Writer</dc:creator></metadata>"
            f"<manifest>{manifest}</manifest><spine>{spine}</spine></package>",
        )
        for i, (_title, body) in enumerate(chapters):
            zf.writestr(f"OEBPS/text/ch{i}.xhtml", f'<html xmlns="http://www.w3.org/1999/xhtml"><body>{body}</body></html>')
        if encrypted:
            refs = "".join(
                f'<EncryptedData><CipherData><CipherReference URI="{uri}"/></CipherData></EncryptedData>' for uri in encrypted
            )
            zf.writestr("META-INF/encryption.xml", f"<encryption>{refs}</encryption>")
    return buffer.getvalue()


def test_epub_uses_metadata_toc_titles_and_skips_footnote_markers():
    data = _epub([
        ("Chapter One", "<p>It was a bright cold day.<sup>1</sup></p><p>The clocks&nbsp;struck thirteen.</p>"),
        ("Chapter Two", "<h1>Chapter Two</h1><p>Second chapter text.</p>"),
    ])
    document = parse_upload("tiny.epub", data, max_uncompressed=10 * MB)

    assert document.title == "Tiny Book"
    assert document.author == "A. Writer"
    assert [chapter.title for chapter in document.chapters] == ["Chapter One", "Chapter Two"]
    first = document.chapters[0].blocks
    assert first[0].kind == HEADING and first[0].text == "Chapter One"
    assert first[1].text == "It was a bright cold day."
    assert first[2].text == "The clocks struck thirteen."
    # The second chapter already starts with its heading, so it is not duplicated.
    assert [block.text for block in document.chapters[1].blocks] == ["Chapter Two", "Second chapter text."]


def test_epub_without_nav_falls_back_to_first_heading():
    data = _epub([("ignored", "<h2>Opening</h2><p>Words here.</p>")], nav=False)
    document = parse_upload("book.epub", data, max_uncompressed=10 * MB)
    assert document.chapters[0].title == "Opening"


def test_epub_with_encrypted_text_is_rejected_but_obfuscated_fonts_are_fine():
    chapters = [("One", "<p>Readable.</p>")]
    with pytest.raises(ValueError, match="DRM"):
        parse_upload("drm.epub", _epub(chapters, encrypted=["OEBPS/text/ch0.xhtml"]), max_uncompressed=10 * MB)
    document = parse_upload("fonts.epub", _epub(chapters, encrypted=["OEBPS/fonts/a.otf"]), max_uncompressed=10 * MB)
    assert document.chapters[0].blocks[-1].text == "Readable."


def test_epub_size_and_format_limits():
    with pytest.raises(ValueError, match="too large"):
        parse_upload("big.epub", _epub([("One", "<p>" + "x" * 5000 + "</p>")]), max_uncompressed=1000)
    with pytest.raises(ValueError, match="zip"):
        parse_upload("broken.epub", b"not a zip", max_uncompressed=10 * MB)


def test_markdown_headings_become_chapters_and_code_is_skipped():
    text = (
        "# Guide\n\nIntro paragraph\nwrapped onto two lines.\n\n"
        "## Install\n\n- first step\n- second step\n\n```bash\nrm -rf /\n```\n\n"
        "### Details\n\nSee [the docs](https://example.com) for **more**.\n"
    )
    document = parse_upload("README.md", text.encode(), max_uncompressed=MB)

    assert document.title == "Guide"
    assert [chapter.title for chapter in document.chapters] == ["Guide", "Install"]
    install = [block.text for block in document.chapters[1].blocks]
    assert install == ["Install", "first step", "second step", CODE_BLOCK_CUE, "Details", "See the docs for more."]
    assert document.chapters[0].blocks[1].text == "Intro paragraph wrapped onto two lines."


def test_plain_text_detects_chapter_lines_and_gutenberg_wrapper():
    text = (
        "Title: A Small Tale\nAuthor: Someone\n\n"
        "*** START OF THE PROJECT GUTENBERG EBOOK A SMALL TALE ***\n\n"
        "CHAPTER I\n\nOnce upon\na time.\n\nCHAPTER II\n\nThe end.\n\n"
        "*** END OF THE PROJECT GUTENBERG EBOOK A SMALL TALE ***\nLicense text that should not be read.\n"
    )
    document = parse_upload("tale.txt", text.encode(), max_uncompressed=MB)

    assert document.title == "A Small Tale"
    assert document.author == "Someone"
    assert [chapter.title for chapter in document.chapters] == ["CHAPTER I", "CHAPTER II"]
    assert document.chapters[0].blocks[1].text == "Once upon a time."
    assert "License" not in " ".join(block.text for chapter in document.chapters for block in chapter.blocks)


def test_html_upload_and_pasted_text():
    html = "<html><head><title>Page</title><style>p{}</style></head><body><nav>Menu</nav><h1>Top</h1><p>Body text.</p></body></html>"
    document = parse_upload("page.html", html.encode(), max_uncompressed=MB)
    assert document.title == "Page"
    assert [block.text for block in document.chapters[0].blocks] == ["Top", "Body text."]

    pasted = parse_pasted("Just one line of pasted words.", title="My note")
    assert pasted.title == "My note"
    assert pasted.chapters[0].blocks[0].text == "Just one line of pasted words."


def test_rejects_unknown_types_and_empty_text():
    with pytest.raises(ValueError, match="Unsupported"):
        parse_upload("song.mp3", b"ID3", max_uncompressed=MB)
    with pytest.raises(ValueError, match="No readable text"):
        parse_pasted("   \n *** \n")


def test_plain_text_keeps_sentences_that_only_look_like_chapters():
    text = "Part One\n\nPart of me wanted to leave.\n\nIntroduction done, we began.\n\nChapter 3.\n\nThe end."
    document = parse_upload("story.txt", text.encode(), max_uncompressed=MB)
    assert [chapter.title for chapter in document.chapters] == ["Part One", "Chapter 3."]


def test_windows_1252_text_and_utf16_with_bom():
    assert parse_upload("cafe.txt", "Café au lait!\n".encode("cp1252"), max_uncompressed=MB).chapters[0].blocks[0].text == "Café au lait!"
    assert parse_upload("w.txt", "Wide text.".encode("utf-16"), max_uncompressed=MB).chapters[0].blocks[0].text == "Wide text."


def test_self_closing_skip_tags_do_not_hide_the_rest_of_the_page():
    html = "<p>Before</p><svg/><p>After</p><table><tr><td>a</td><td> </td><td>b</td></tr></table>"
    document = parse_upload("page.html", html.encode(), max_uncompressed=MB)
    assert [block.text for block in document.chapters[0].blocks] == ["Before", "After", "a, b"]


@pytest.mark.parametrize("sentence_chars", [25, 599, 799, 990])
def test_long_paragraphs_are_split_at_sentences(sentence_chars):
    sentence = "word " * ((sentence_chars - 1) // 5) + "end."
    paragraph = " ".join([sentence] * max(9, 3000 // sentence_chars))
    blocks = parse_pasted(paragraph).chapters[0].blocks
    assert len(blocks) > 1
    assert all(len(block.text) <= MAX_BLOCK_CHARS for block in blocks)
    assert " ".join(block.text for block in blocks) == paragraph


def test_split_sentences_rules_and_speed():
    text = "Mr. Smith arrived. He sat down! Did he stay? " + "word " * 200
    spans = [text[start:end] for start, end in split_sentences(text, 100)]
    assert spans[:3] == ["Mr. Smith arrived.", "He sat down!", "Did he stay?"]
    assert all(len(span) <= 100 for span in spans)

    huge = "A short sentence here. " * 50_000  # ~1.1 MB in one paragraph
    started = time.perf_counter()
    assert len(split_sentences(huge, 400)) == 50_000
    assert time.perf_counter() - started < 5


def test_html_title_ignores_inline_svg_titles():
    html = "<html><head><title>My Page</title></head><body><svg><title>Open menu</title></svg><p>Text.</p></body></html>"
    assert parse_upload("page.html", html.encode(), max_uncompressed=MB).title == "My Page"
    no_head = "<body><svg><title>Icon</title></svg><h1>Real heading</h1><p>Text.</p></body>"
    assert parse_upload("page.html", no_head.encode(), max_uncompressed=MB).title == "Real heading"


def test_epub_with_a_broken_table_of_contents_still_imports():
    data = bytearray(_epub([("Chapter One", "<h1>Opening</h1><p>Words.</p>")]))
    broken = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(bytes(data))) as source, zipfile.ZipFile(broken, "w") as target:
        for item in source.infolist():
            payload = b"<html><nav" if item.filename == "OEBPS/nav.xhtml" else source.read(item.filename)
            target.writestr(item, payload)
    document = parse_upload("book.epub", broken.getvalue(), max_uncompressed=10 * MB)
    assert document.chapters[0].title == "Opening"
