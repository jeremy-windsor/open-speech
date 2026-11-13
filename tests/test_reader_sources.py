from __future__ import annotations

import io
import os
import sys
import time
import zipfile

import pytest

from src.live_tts.segmenter import split_sentences
from src.reader.document import HEADING, MAX_BLOCK_CHARS, finalize
from src.reader.sources import parse_pasted, parse_upload
from src.reader.sources.text import CODE_BLOCK_CUE

MB = 1024 * 1024


def _epub(
    chapters: list[tuple[str, str]],
    *,
    nav: bool = True,
    encrypted: list[str] | None = None,
    links: list[tuple[str, str]] | None = None,
) -> bytes:
    """A small EPUB 3. `links` replaces the default contents (one entry per file) with (href, label) pairs."""
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
            contents = links or [(f"text/ch{i}.xhtml", title) for i, (title, _body) in enumerate(chapters)]
            items = "".join(f'<li><a href="{href}">{label}</a></li>' for href, label in contents)
            zf.writestr(
                "OEBPS/nav.xhtml",
                '<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops"><body>'
                f'<nav epub:type="toc"><ol>{items}</ol></nav></body></html>',
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


def test_epub_chapters_follow_the_contents_when_files_hold_several_chapters():
    # Shaped like a Project Gutenberg EPUB 3: big files with several chapters each, headings that carry an
    # illustration caption (which the contents label repeats), figure captions, and a continuation file.
    figure = '<div class="figcenter"><img alt="" src="i.jpg"/><div class="caption"><p>Mr. &amp; Mrs. Bennet</p></div></div>'
    data = _epub(
        [
            ("ignored", (
                "<p>The Project Gutenberg eBook of Pride and Prejudice</p>"
                '<h1 id="t"><img alt="" src="f.png"/><br/><i>PRIDE.<br/>and<br/>PREJUDICE</i></h1><p>by Jane Austen</p>'
                '<h2 id="c1"><a id="Chapter_I"/><img alt="" src="a.jpg"/><br/>Chapter I.</h2>'
                '<p class="nind"><span class="letra"><img alt="I" src="i.png"/></span>t is a truth.</p>'
                f"{figure}<p>More of chapter one.</p>"
                '<h2 id="c2"><a id="CHAPTER_II"/><img alt="" src="b.jpg"/>'
                '<span class="caption">I hope Mr. Bingley will like it.</span><br/><br/>CHAPTER II.</h2>'
                '<p><span class="letra"><img alt="M" src="m.png"/></span>R. BENNET was among the earliest.</p>'
                '<p><img alt="\u201cI" src="q.png"/>\u2019m the tallest.\u201d <img alt="**" src="stars.png"/>Done.</p>'
            )),
            ("ignored", (
                '<h2 id="c11"><img alt="" src="c.jpg"/><br/><span class="caption">Piling up the fire.</span>'
                "<br/><br/>CHAPTER XI.</h2><p>When the ladies removed.</p>"
                '<h2 id="c27"><span class="caption">\u201cOn the Stairs.\u201d</span><br/>CHAPTER<br/>XXVII.</h2><p>With no greater events.</p>'
            )),
            ("ignored", "<p>Chapter twenty-seven goes on in the next file.</p>"),
        ],
        links=[
            ("text/ch0.xhtml#t", "PRIDE. and PREJUDICE"),
            ("text/ch0.xhtml#c1", "Chapter I."),
            ("text/ch0.xhtml#c2", "I hope Mr. Bingley will like it. CHAPTER II."),
            ("text/ch0.xhtml#missing", "Not in the book"),
            ("text/ch1.xhtml#c11", "Piling up the fire. CHAPTER XI."),
            ("text/ch1.xhtml#c27", "\u201cOn the Stairs.\u201d CHAPTERXXVII."),
        ],
    )
    document = finalize(parse_upload("pp.epub", data, max_uncompressed=10 * MB), "pp")

    titles = [chapter.title for chapter in document.chapters]
    assert titles == [
        "The Project Gutenberg eBook of Pride and Prejudice",
        "PRIDE. and PREJUDICE",
        "Chapter I.",
        "CHAPTER II.",
        "CHAPTER XI.",
        "CHAPTER XXVII.",
    ]
    texts = [[block.text for block in chapter.blocks] for chapter in document.chapters]
    assert texts[2] == ["Chapter I.", "It is a truth.", "More of chapter one."]   # the picture caption is not read
    # Decorated first letters are pictures whose alt text is the letter; other alt text is not read.
    assert texts[3] == ["CHAPTER II.", "MR. BENNET was among the earliest.", "\u201cI\u2019m the tallest.\u201d Done."]
    # The file the contents never point into carries on the chapter before it.
    assert texts[5] == ["CHAPTER XXVII.", "With no greater events.", "Chapter twenty-seven goes on in the next file."]


def test_epub_contents_entries_without_fragments_start_at_the_file():
    data = _epub(
        [("ignored", "<p>Opening words.</p>"), ("ignored", "<h2>Two</h2><p>More words.</p>"), ("ignored", "<p>Third file.</p>")],
        links=[("text/ch0.xhtml", "First part")],
    )
    document = parse_upload("parts.epub", data, max_uncompressed=10 * MB)
    # An unlisted file that opens with a heading starts its own chapter; one that doesn't carries on.
    assert [chapter.title for chapter in document.chapters] == ["First part", "Two"]
    assert [block.text for block in document.chapters[1].blocks] == ["Two", "More words.", "Third file."]


def _chapter_shapes(document) -> list[tuple[str, list[tuple[str, str]]]]:
    return [(chapter.title, [(block.kind, block.text) for block in chapter.blocks]) for chapter in document.chapters]


def test_epub_contents_entry_for_a_file_without_text_does_not_name_the_next_chapter():
    # An image-only cover has no blocks, so its entry took the next
    # file's first block. "Cover" then named chapter one, was read aloud as its heading, and "Chapter 1"
    # was lost from the chapter list.
    data = _epub([
        ("Cover", '<div><img alt="Cover" src="cover.jpg"/></div>'),
        ("Chapter 1", "<p>It was a bright cold day.</p>"),
        ("Chapter 2", "<p>Second.</p>"),
    ])
    document = parse_upload("book.epub", data, max_uncompressed=10 * MB)

    assert _chapter_shapes(document) == [
        ("Chapter 1", [("heading", "Chapter 1"), ("paragraph", "It was a bright cold day.")]),
        ("Chapter 2", [("heading", "Chapter 2"), ("paragraph", "Second.")]),
    ]


def test_epub_contents_entry_for_an_anchor_after_a_files_last_block_does_not_name_the_next_chapter():
    # Nothing follows the "notes" anchor in its own file, so its
    # entry took the next file's first block and "Notes" named chapter two.
    data = _epub(
        [("ignored", '<p>First chapter text.</p><a id="notes"></a>'), ("ignored", "<p>Second chapter text.</p>")],
        links=[("text/ch0.xhtml", "Chapter 1"), ("text/ch0.xhtml#notes", "Notes"), ("text/ch1.xhtml", "Chapter 2")],
    )
    document = parse_upload("book.epub", data, max_uncompressed=10 * MB)

    assert _chapter_shapes(document) == [
        ("Chapter 1", [("heading", "Chapter 1"), ("paragraph", "First chapter text.")]),
        ("Chapter 2", [("heading", "Chapter 2"), ("paragraph", "Second chapter text.")]),
    ]


def _contents_epub(kind: str, contents: str, bodies: list[str]) -> bytes:
    """An EPUB whose contents are `contents`: the markup inside <nav epub:type="toc"> (kind "nav", EPUB 3)
    or inside <navMap> (kind "ncx", EPUB 2). The files are text/ch0.xhtml, text/ch1.xhtml, ... in that order."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("mimetype", "application/epub+zip")
        zf.writestr(
            "META-INF/container.xml",
            '<?xml version="1.0"?><container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">'
            '<rootfiles><rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/></rootfiles></container>',
        )
        manifest = "".join(
            f'<item id="c{i}" href="text/ch{i}.xhtml" media-type="application/xhtml+xml"/>' for i in range(len(bodies))
        )
        spine = "".join(f'<itemref idref="c{i}"/>' for i in range(len(bodies)))
        if kind == "nav":
            version, spine_toc = "3.0", ""
            manifest += '<item id="nav" href="nav.xhtml" media-type="application/xhtml+xml" properties="nav"/>'
            zf.writestr(
                "OEBPS/nav.xhtml",
                '<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops"><body>'
                f'<nav epub:type="toc">{contents}</nav></body></html>',
            )
        else:
            version, spine_toc = "2.0", ' toc="ncx"'
            manifest += '<item id="ncx" href="toc.ncx" media-type="application/x-dtbncx+xml"/>'
            zf.writestr(
                "OEBPS/toc.ncx",
                '<?xml version="1.0"?><ncx xmlns="http://www.daisy.org/z3986/2005/ncx/" version="2005-1">'
                f"<navMap>{contents}</navMap></ncx>",
            )
        zf.writestr(
            "OEBPS/content.opf",
            f'<?xml version="1.0"?><package xmlns="http://www.idpf.org/2007/opf" version="{version}">'
            '<metadata xmlns:dc="http://purl.org/dc/elements/1.1/"><dc:title>Contents Book</dc:title></metadata>'
            f"<manifest>{manifest}</manifest><spine{spine_toc}>{spine}</spine></package>",
        )
        for i, body in enumerate(bodies):
            zf.writestr(f"OEBPS/text/ch{i}.xhtml", f'<html xmlns="http://www.w3.org/1999/xhtml"><body>{body}</body></html>')
    return buffer.getvalue()


def _contents_markup(kind: str, entries: list[tuple[str, str, list]]) -> str:
    """Nested contents for (href, label, children) entries: nav <ol>/<li>, or NCX navPoints."""
    if kind == "nav":
        items = "".join(
            f'<li><a href="{href}">{label}</a>{_contents_markup(kind, children) if children else ""}</li>'
            for href, label, children in entries
        )
        return f"<ol>{items}</ol>"
    return "".join(
        f'<navPoint><navLabel><text>{label}</text></navLabel><content src="{href}"/>'
        f"{_contents_markup(kind, children)}</navPoint>"
        for href, label, children in entries
    )


def _deep_contents_epub(kind: str) -> bytes:
    """A one-file book whose contents nest deeper than Python's recursion limit, labelled "Level 0" outward
    in, every entry at that file. Bounded, so the contents file stays small."""
    depth = min(sys.getrecursionlimit() + 200, 20_000)
    if kind == "nav":
        nested = "".join(f'<ol><li><a href="text/ch0.xhtml">Level {i}</a>' for i in range(depth)) + "</li></ol>" * depth
    else:
        nested = "".join(
            f'<navPoint><navLabel><text>Level {i}</text></navLabel><content src="text/ch0.xhtml"/>' for i in range(depth)
        ) + "</navPoint>" * depth
    return _contents_epub(kind, nested, ["<p>Hello there.</p>"])


@pytest.mark.parametrize("kind", ["nav", "ncx"])
def test_epub_contents_nested_past_the_recursion_limit_import_with_their_chapter(kind):
    # The contents walks recursed once per level, so a contents list nested
    # past Python's recursion limit (a few KB zipped) raised RecursionError. Neither the broken-contents fallback
    # nor the route's 422 caught it, and the upload answered 500.
    document = parse_upload("deep.epub", _deep_contents_epub(kind), max_uncompressed=10 * MB)

    # The contents were read, outermost entry first: it names the chapter ("Section 1" if they were dropped).
    assert _chapter_shapes(document) == [("Level 0", [("heading", "Level 0"), ("paragraph", "Hello there.")])]


@pytest.mark.parametrize("kind", ["nav", "ncx"])
def test_epub_nested_contents_keep_their_order_and_depth(kind):
    # The walks became iterative; entries keep document order (a parent
    # before its children, children before the next sibling) and their depth (TOC_DEPTH = 1 starts chapters).
    contents = _contents_markup(kind, [
        ("text/ch0.xhtml", "Part One", [
            ("text/ch0.xhtml#c1", "Chapter 1", [("text/ch0.xhtml#s1", "Section 1.1", [])]),
            ("text/ch0.xhtml#c2", "Chapter 2", []),
        ]),
        ("text/ch1.xhtml", "Part Two", [("text/ch1.xhtml", "Opening", [])]),   # one place: the first entry names it
    ])
    data = _contents_epub(kind, contents, [
        '<p>Intro.</p><h3 id="c1">Chapter 1</h3><p>One.</p><h4 id="s1">Section 1.1</h4><p>Deep.</p>'
        '<h3 id="c2">Chapter 2</h3><p>Two.</p>',
        "<p>Later.</p>",
    ])
    document = parse_upload("parts.epub", data, max_uncompressed=10 * MB)

    assert _chapter_shapes(document) == [
        ("Part One", [("heading", "Part One"), ("paragraph", "Intro.")]),
        ("Chapter 1", [
            ("heading", "Chapter 1"), ("paragraph", "One."), ("heading", "Section 1.1"), ("paragraph", "Deep."),
        ]),
        ("Chapter 2", [("heading", "Chapter 2"), ("paragraph", "Two.")]),
        ("Part Two", [("heading", "Part Two"), ("paragraph", "Later.")]),
    ]


def test_epub_size_and_format_limits():
    with pytest.raises(ValueError, match="too large"):
        parse_upload("big.epub", _epub([("One", "<p>" + "x" * 5000 + "</p>")]), max_uncompressed=1000)
    with pytest.raises(ValueError, match="zip"):
        parse_upload("broken.epub", b"not a zip", max_uncompressed=10 * MB)


def test_epub_spine_naming_one_file_many_times_reads_it_once(monkeypatch):
    # A spine of 2,000 itemrefs to one file costs ~20 bytes each and compresses ~1000:1: reading
    # the file again for each made a 1 KB upload decompress and hold far more than the cap allows.
    paragraph = "word " * 190
    data = _rewrite_member(
        _epub([("Chapter One", f"<p>{paragraph}</p>")]),
        "OEBPS/content.opf",
        lambda payload: payload.replace(b'<itemref idref="c0"/>', b'<itemref idref="c0"/>' * 2000),
    )
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        declared = sum(info.file_size for info in zf.infolist())
    read = zipfile.ZipFile.read
    decompressed: list[int] = []

    def counting_read(self, name, pwd=None):
        payload = read(self, name, pwd)
        decompressed.append(len(payload))
        return payload

    monkeypatch.setattr(zipfile.ZipFile, "read", counting_read)
    document = parse_upload("repeats.epub", data, max_uncompressed=declared + 10)

    assert sum(decompressed) <= declared
    assert [block.text for chapter in document.chapters for block in chapter.blocks] == [
        "Chapter One", paragraph.strip(),
    ]


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


def test_import_keeps_a_rule_line_of_underscores_so_stored_documents_keep_their_hash():
    # is_speakable no longer counts "_" as a letter, so the renderer never sends a
    # "__________" line to the model. Import keeps its own rule (a word character, "_" included): the line stays a
    # block, as in documents already in a library, so a re-import has the same content_hash ("Already in your library").
    text = "The first scene ends here.\n\n__________\n\nThe second scene begins."
    blocks = parse_upload("story.txt", text.encode(), max_uncompressed=MB).chapters[0].blocks
    assert [block.text for block in blocks] == ["The first scene ends here.", "__________", "The second scene begins."]


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


# A sound tag reaches the voice model whole. A cut by max_chars
# (the sentences stored at import, a part's pieces for the model's limit) used to fall at the space inside
# "[clear throat]", so the voice got "... [clear" and "throat] ...". A sentence cut must not split [clear throat].
@pytest.mark.parametrize("pad", range(14))
@pytest.mark.parametrize("tag", ["[clear throat]", "[laugh]"])
def test_split_sentences_never_cuts_inside_a_sound_tag(tag, pad):
    text = "x" * pad + " " + " ".join(["word"] * 18) + f" {tag} and then some more words to end it."
    pieces = [text[start:end] for start, end in split_sentences(text, 100)]
    assert sum(piece.count(tag) for piece in pieces) == 1, pieces
    assert all(len(piece) <= 100 for piece in pieces), pieces
    assert " ".join(pieces).split() == text.split()


def test_split_sentences_still_cuts_a_sound_tag_that_starts_the_text_and_is_over_the_cap():
    # The tag can't move before itself, so a cap smaller than it still cuts it.
    text = "[clear throat] and on."
    pieces = [text[start:end] for start, end in split_sentences(text, 8)]
    assert all(len(piece) <= 8 for piece in pieces), pieces
    assert " ".join(pieces).split() == text.split()


@pytest.mark.parametrize("markdown", [False, True], ids=["text", "markdown"])
@pytest.mark.parametrize("tag", ["[laugh]", "[chuckle]", "[sigh]", "[clear throat]", "[whispering]"])
def test_import_keeps_each_sound_tag_whole(tag, markdown):
    # Import (text and Markdown) keeps every tag as written.
    from src.reader.sources.text import parse_text

    document = parse_text(f"Chapter 1\n\nShe looked up. {tag} That was the end of it.\n\n{tag} Then more.\n", markdown=markdown)
    texts = [block.text for chapter in document.chapters for block in chapter.blocks]
    assert sum(text.count(tag) for text in texts) == 2, texts


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


def _rewrite_member(data: bytes, name: str, change) -> bytes:
    """The same EPUB with one member's bytes passed through `change`."""
    rebuilt = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(data)) as source, zipfile.ZipFile(rebuilt, "w") as target:
        for item in source.infolist():
            payload = source.read(item.filename)
            target.writestr(item, change(payload) if item.filename == name else payload)
    return rebuilt.getvalue()


def test_markdown_keeps_the_words_between_comparison_signs():
    # CommonMark raw HTML needs a tag name right after "<"; real tags are still dropped.
    document = parse_pasted("Stop when x < 5 and y > 3, then rest.\n\nUse <b>bold</b> text.")
    assert [block.text for block in document.chapters[0].blocks] == [
        "Stop when x < 5 and y > 3, then rest.",
        "Use bold text.",
    ]


def test_markdown_heading_keeps_a_trailing_hash_that_belongs_to_the_text():
    # A closing "#" run only counts after a space: "# foo#" is the heading "foo#".
    assert parse_pasted("# Learning C#\n\nSome text about the language.").title == "Learning C#"
    assert parse_pasted("# Closed heading ##\n\nSome text.").title == "Closed heading"


def test_markdown_setext_level_two_heading_starts_a_chapter():
    document = parse_pasted("Intro\n=====\n\nHello there.\n\nInstall\n-------\n\nRun it.")

    assert document.title == "Intro"
    assert [chapter.title for chapter in document.chapters] == ["Intro", "Install"]
    assert [block.text for block in document.chapters[1].blocks] == ["Install", "Run it."]


def test_markdown_multi_line_blockquote_drops_every_quote_marker():
    document = parse_pasted("Intro line.\n\n> First line of the quote\n> second line of the quote.\n\nAfter.")

    assert [block.text for block in document.chapters[0].blocks] == [
        "Intro line.",
        "First line of the quote second line of the quote.",
        "After.",
    ]


def test_markdown_list_item_with_a_continuation_line_stays_a_separate_item():
    document = parse_pasted("- one item\n  continued here\n- two")

    assert [block.text for block in document.chapters[0].blocks] == ["one item continued here", "two"]


def test_epub_chapter_encoded_as_utf16_is_read():
    # EPUB 3 content documents "MUST be encoded in UTF-8 or UTF-16".
    data = _rewrite_member(
        _epub([("Chapter One", "<p>Hello from the book.</p>")]),
        "OEBPS/text/ch0.xhtml",
        lambda payload: ('<?xml version="1.0" encoding="UTF-16"?>' + payload.decode("utf-8")).encode("utf-16"),
    )

    document = parse_upload("wide.epub", data, max_uncompressed=10 * MB)

    assert [block.text for block in document.chapters[0].blocks] == ["Chapter One", "Hello from the book."]


def test_epub_chapter_with_an_empty_self_closing_title_keeps_its_text():
    # In XHTML <title/> is an empty element: no end tag follows to stop collecting the title.
    data = _rewrite_member(
        _epub([("Chapter One", "<p>Hello from the book.</p>")]),
        "OEBPS/text/ch0.xhtml",
        lambda payload: payload.replace(b"<body>", b"<head><title/></head><body>"),
    )

    document = parse_upload("untitled-chapters.epub", data, max_uncompressed=10 * MB)

    assert [block.text for block in document.chapters[0].blocks] == ["Chapter One", "Hello from the book."]


def test_html_page_without_the_optional_head_end_tag_keeps_its_body():
    # HTML: "A head element's end tag may be omitted"; head content in <noscript> does not end it.
    page = (
        b"<!doctype html><html><head><meta charset=utf-8><title>Page</title>"
        b"<noscript><img src=pixel.gif></noscript><body><h1>Top</h1><p>Body text.</p></body></html>"
    )

    document = parse_upload("page.html", page, max_uncompressed=MB)

    assert document.title == "Page"
    assert [block.text for block in document.chapters[0].blocks] == ["Top", "Body text."]


def test_html_upload_uses_its_declared_charset():
    shift_jis = (
        '<html><head><meta charset="shift_jis"><title>Greeting</title></head><body><p>こんにちは世界。</p></body></html>'
    ).encode("shift_jis")
    cyrillic = (
        '<html><head><meta http-equiv="Content-Type" content="text/html; charset=windows-1251"></head>'
        "<body><p>Привет, мир.</p></body></html>"
    ).encode("cp1251")
    # A declaration the bytes contradict falls back to the usual guesses.
    mislabelled = '<html><head><meta charset="utf-8"></head><body><p>Café au lait.</p></body></html>'.encode("cp1252")

    assert parse_upload("greeting.html", shift_jis, max_uncompressed=MB).chapters[0].blocks[0].text == "こんにちは世界。"
    assert parse_upload("hello.html", cyrillic, max_uncompressed=MB).chapters[0].blocks[0].text == "Привет, мир."
    assert parse_upload("cafe.html", mislabelled, max_uncompressed=MB).chapters[0].blocks[0].text == "Café au lait."


@pytest.mark.parametrize("label", ["base64", "zlib", "hex", "rot13", "utf-7", "utf-32", "undefined", "punycode"])
def test_html_charset_that_names_no_web_encoding_is_ignored(label):
    # WHATWG Encoding Standard: browsers ignore a label it does not list. Python also knows
    # bytes-to-bytes codecs (base64.decode raises LookupError) and UTF-7, which the HTML Standard
    # forbids and which turns "+2AA-" into a lone surrogate that cannot be stored.
    page = f'<html><head><meta charset="{label}"></head><body><p>Hello +2AA- there.</p></body></html>'.encode("ascii")

    document = parse_upload("page.html", page, max_uncompressed=MB)

    assert [block.text for block in document.chapters[0].blocks] == ["Hello +2AA- there."]


def test_html_charset_after_an_ignored_one_is_used():
    # As in the HTML Standard's prescan, a <meta> whose label names no encoding is passed over.
    page = (
        '<html><head><meta charset="base64"><meta charset="windows-1251"></head><body><p>Привет, мир.</p></body></html>'
    ).encode("cp1251")

    assert parse_upload("hello.html", page, max_uncompressed=MB).chapters[0].blocks[0].text == "Привет, мир."


@pytest.mark.parametrize("declaration", ['<?xml version="1.0" encoding="zlib"?>', '<meta charset="hex"/>'])
def test_epub_chapter_that_names_no_web_encoding_is_read_as_utf8(declaration):
    data = _rewrite_member(
        _epub([("Chapter One", "<p>Hello from the book.</p>")]),
        "OEBPS/text/ch0.xhtml",
        lambda payload: declaration.encode("ascii") + payload,
    )

    document = parse_upload("book.epub", data, max_uncompressed=10 * MB)

    assert [block.text for block in document.chapters[0].blocks] == ["Chapter One", "Hello from the book."]


@pytest.mark.parametrize(
    ("label", "text", "encoded"),
    [
        ("shift_jis", "第①章 テスト文章です。", "第①章 テスト文章です。".encode("cp932")),
        ("gb2312", "朱镕基的文章。", "朱镕基的文章。".encode("gbk")),
        ("euc-kr", "똠방각하 테스트.", "똠방각하 테스트.".encode("cp949")),
        # Windows writes the euro sign as 0x80 in GBK and 0xA3E1 in Big5, and NEC's ① in EUC-JP.
        ("gbk", "价格是€十元。", "价格是".encode("gbk") + b"\x80" + "十元。".encode("gbk")),
        ("big5", "價格是€十元。", "價格是€十元。".encode("cp950")),
        ("euc-jp", "第①章です。", "第".encode("euc_jp") + b"\xad\xa1" + "章です。".encode("euc_jp")),
    ],
    ids=["shift_jis-nec-digit", "gb2312-gbk-hanzi", "euc-kr-uhc-hangul", "gbk-euro", "big5-euro", "euc-jp-nec-digit"],
)
def test_legacy_cjk_charsets_decode_as_browsers_do(label, text, encoded):
    # The WHATWG Encoding Standard reads Shift_JIS as Windows-31J, gb2312 as GBK and EUC-KR as
    # windows-949. Python's strict codecs refuse their extra characters, and the page became mojibake.
    page = f'<html><head><meta charset="{label}"></head><body><p>'.encode("ascii") + encoded + b"</p></body></html>"
    # An EPUB content document is XML: its XML declaration names the encoding.
    xhtml = f'<?xml version="1.0" encoding="{label}"?><html xmlns="http://www.w3.org/1999/xhtml"><body><p>'
    chapter = _rewrite_member(
        _epub([("Chapter One", "<p>placeholder</p>")]),
        "OEBPS/text/ch0.xhtml",
        lambda _payload: xhtml.encode("ascii") + encoded + b"</p></body></html>",
    )

    assert parse_upload("page.html", page, max_uncompressed=MB).chapters[0].blocks[0].text == text
    assert parse_upload("book.epub", chapter, max_uncompressed=10 * MB).chapters[0].blocks[-1].text == text


@pytest.mark.parametrize(
    ("label", "encoded", "text"),
    [
        # A pair no Shift_JIS table maps: both bytes go, and the next character is read whole.
        ("shift_jis", "テスト".encode("cp932") + b"\x85\x9f" + "です。".encode("cp932"), "テスト\ufffdです。"),
        # An ASCII byte after a lead byte is read again on its own.
        ("shift_jis", "テスト".encode("cp932") + b"\x85A" + "です。".encode("cp932"), "テスト\ufffdAです。"),
        ("big5", "價格是".encode("big5") + b"\xff" + "十元。".encode("big5"), "價格是\ufffd十元。"),
    ],
    ids=["unmapped-pair", "lead-byte-before-ascii", "not-a-lead-byte"],
)
def test_bytes_the_declared_charset_cannot_read_cost_one_character(label, encoded, text):
    # A browser keeps the encoding it chose and shows U+FFFD (WHATWG Encoding Standard): the page
    # used to fall back to Windows-1252 as a whole, and a plain "replace" reads the next pair askew.
    page = f'<html><head><meta charset="{label}"></head><body><p>'.encode("ascii") + encoded + b"</p></body></html>"

    assert parse_upload("page.html", page, max_uncompressed=MB).chapters[0].blocks[0].text == text


@pytest.mark.parametrize("prolog", ['<?xml version="1.0"?>', ""], ids=["xml-declaration", "no-declaration"])
def test_epub_chapter_ignores_a_meta_charset_left_from_html(prolog):
    # XML documents name their encoding only with a BOM or XML declaration, and are UTF-8 without
    # either (EPUB 3: "MUST be encoded in UTF-8 or UTF-16"); an XHTML parser ignores <meta>.
    chapter = (
        f'{prolog}<html xmlns="http://www.w3.org/1999/xhtml"><head>'
        '<meta http-equiv="Content-Type" content="text/html; charset=iso-8859-1"/></head>'
        "<body><p>Café crème, naïve résumé.</p></body></html>"
    ).encode("utf-8")
    data = _rewrite_member(
        _epub([("Chapter One", "<p>placeholder</p>")]), "OEBPS/text/ch0.xhtml", lambda _payload: chapter
    )

    document = parse_upload("converted.epub", data, max_uncompressed=10 * MB)

    assert [block.text for block in document.chapters[0].blocks] == ["Chapter One", "Café crème, naïve résumé."]


@pytest.mark.parametrize(
    "fenced",
    [
        "~~~markdown\n```python\nprint(1)\n```\n~~~",
        "```\n~~~\nprint(1)\n~~~\n```",
        "````\n```python\nprint(1)\n```\n````",
    ],
    ids=["tilde-fence-shows-backticks", "backtick-fence-shows-tildes", "four-backtick-fence"],
)
def test_markdown_code_fence_closes_only_on_a_matching_fence(fenced):
    # CommonMark 4.5: the closing fence uses the same character, at least as many times as the opening.
    document = parse_pasted(f"Intro.\n\n{fenced}\n\nAfter the example.")

    assert [block.text for block in document.chapters[0].blocks] == ["Intro.", CODE_BLOCK_CUE, "After the example."]


def test_markdown_backticks_with_more_backticks_on_the_line_open_no_fence():
    # CommonMark 4.5: a backtick fence's info string may not contain backticks; this is inline code.
    document = parse_pasted("Run this:\n```make build```\n\nAfter the example.")

    assert [block.text for block in document.chapters[0].blocks][-1] == "After the example."
    assert CODE_BLOCK_CUE not in [block.text for block in document.chapters[0].blocks]


@pytest.mark.parametrize(
    ("text", "blocks"),
    [
        ("Shopping list:\n- milk\n- eggs", ["Shopping list:", "milk", "eggs"]),
        ("Steps:\n1. Open the lid\n2. Pour the water", ["Steps:", "Open the lid", "Pour the water"]),
        # Only an ordered list that starts at 1 interrupts a paragraph: a wrapped year stays prose.
        ("We moved there in\n2019. It was cold.", ["We moved there in 2019. It was cold."]),
    ],
    ids=["bullets", "ordered", "wrapped-number"],
)
def test_markdown_list_right_after_a_paragraph_line_is_a_list(text, blocks):
    # CommonMark 5.3: a bullet list, or an ordered list starting at 1, can interrupt a paragraph.
    document = parse_pasted(text)

    assert [block.text for block in document.chapters[0].blocks] == blocks


@pytest.mark.parametrize("underline", ["===", "---"])
def test_markdown_setext_heading_text_may_span_several_lines(underline):
    # CommonMark 4.3: "Foo *bar\nbaz*\n====" is one heading.
    document = parse_pasted(f"Line one\nline two\n{underline}\n\nBody text.")

    assert [(block.kind, block.text) for block in document.chapters[0].blocks] == [
        (HEADING, "Line one line two"),
        ("paragraph", "Body text."),
    ]


@pytest.mark.parametrize(
    ("text", "blocks"),
    [
        # CommonMark 4.3: "The setext heading underline cannot be a lazy continuation line in a list item or block quote."
        ("Shopping list:\n- milk\n---", [("paragraph", "Shopping list:"), ("paragraph", "milk")]),
        ("> A quote.\n---\nAfter the rule.", [("paragraph", "A quote."), ("paragraph", "After the rule.")]),
        # CommonMark 4.1: a thematic break can interrupt a paragraph or a list.
        ("- milk\n***\nAfter the list.", [("paragraph", "milk"), ("paragraph", "After the list.")]),
        ("Some text.\n* * *\nMore text.", [("paragraph", "Some text."), ("paragraph", "More text.")]),
        # A setext underline ends its heading; the next line starts a paragraph.
        ("Line one\nline two\n---\nBody text.", [(HEADING, "Line one line two"), ("paragraph", "Body text.")]),
        ("Line one\nline two\n===\nBody text.", [(HEADING, "Line one line two"), ("paragraph", "Body text.")]),
    ],
    ids=[
        "rule-under-a-list", "rule-under-a-quote", "rule-ends-a-list", "rule-in-a-paragraph",
        "text-under-a-heading", "text-under-a-title",
    ],
)
def test_markdown_rule_or_underline_without_a_blank_line_is_not_read_aloud(text, blocks):
    document = parse_pasted(text)

    assert [(block.kind, block.text) for chapter in document.chapters for block in chapter.blocks] == blocks


class _PosixZipNames:
    """os as zipfile sees it on Linux: it leaves a "\\" in a member name alone."""

    sep, altsep = "/", None

    def __getattr__(self, name: str):
        return getattr(os, name)


def _backslash_names(data: bytes) -> bytes:
    """The same EPUB with "\\" between folders in every member name, as some Windows tools write it."""
    rebuilt = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(data)) as source, zipfile.ZipFile(rebuilt, "w") as target:
        for item in source.infolist():
            payload = source.read(item.filename)
            item.filename = item.filename.replace("/", "\\")   # ZipInfo() would turn it back on Windows
            target.writestr(item, payload)
    return rebuilt.getvalue()


@pytest.mark.parametrize("where", ["member-names", "links"])
def test_epub_with_backslash_paths_imports_on_every_platform(where, monkeypatch):
    # ZIP names must use "/", but Python's zipfile turns a "\" into "/" only on Windows: read the
    # archive as the Linux image does. A "\" in a link is a "/" too, as browsers read URLs.
    monkeypatch.setattr(zipfile, "os", _PosixZipNames())
    data = _epub([("Chapter One", "<p>Hello from the book.</p>")])
    if where == "member-names":
        data = _backslash_names(data)
    else:
        data = _rewrite_member(data, "OEBPS/content.opf", lambda payload: payload.replace(b"text/ch0", b"text\\ch0"))

    document = parse_upload("windows.epub", data, max_uncompressed=10 * MB)

    assert [block.text for block in document.chapters[0].blocks] == ["Chapter One", "Hello from the book."]


def test_epub_member_stored_twice_is_read_from_its_last_copy():
    # Updating a zip in place appends a second entry with the same name; zipfile and unzip tools use the last.
    buffer = io.BytesIO(_epub([("Chapter One", "<p>Old text.</p>")]))
    with pytest.warns(UserWarning, match="Duplicate name"), zipfile.ZipFile(buffer, "a") as zf:
        zf.writestr("OEBPS/text/ch0.xhtml", '<html xmlns="http://www.w3.org/1999/xhtml"><body><p>New text.</p></body></html>')

    document = parse_upload("updated.epub", buffer.getvalue(), max_uncompressed=10 * MB)

    assert [block.text for block in document.chapters[0].blocks] == ["Chapter One", "New text."]


# Pasted texts of a few tens of KB. A parser that scans the rest of a line again from every "#",
# "[", "**", "*", "_", "<" or "`" without a closer, or copies a list item's text again for each
# of its lines, spends seconds on each; a regex does it holding the GIL, so not even another
# thread runs meanwhile. A linear parse takes a few milliseconds.
_PATHOLOGICAL_MARKDOWN = {
    "heading-ending-in-spaces": "# a" + " " * 12_000 + "b",
    "unclosed-brackets": "Hello " + "[" * 32_000,
    "links-without-a-closing-parenthesis": "Hello " + "[a](" * 30_000,
    "unclosed-strong-markers": "Hello " + "**x " * 6_000,
    "unclosed-strikethrough": "Hello " + "~~x " * 6_000,
    "unclosed-asterisks": "Hello" + " *x" * 10_000,
    "unclosed-underscores": "Hello" + " _x" * 10_000,
    "unclosed-html-comments": "Hello " + "<!--" * 16_000,
    "unclosed-processing-instructions": "Hello " + "<?" * 32_000,
    "unclosed-declarations": "Hello " + "<!a" * 36_000,
    "many-code-spans": "Hello " + "`a` " * 30_000,
    "list-item-with-many-continuation-lines": "- item\n" + "words\n" * 160_000,
}


@pytest.mark.parametrize("text", list(_PATHOLOGICAL_MARKDOWN.values()), ids=list(_PATHOLOGICAL_MARKDOWN))
def test_markdown_parse_time_grows_linearly_with_pathological_text(text):
    started = time.perf_counter()
    parse_pasted(text)
    assert time.perf_counter() - started < 1.5


def test_markdown_heading_and_inline_markup_parse_as_before():
    # The linear heading and span matching find what the regexes they replaced found.
    document = parse_pasted(
        "# Title #\n\n## Two ## ##\n\n### C# ###\n\n"
        "A **bold** and *soft* and _quiet_ and ~~gone~~ word, a [link](https://x.y/a) and "
        "![image](i.png), <b>tags</b><!-- a comment --> <?pi?><!DOCTYPE html>`**kept**`, "
        "**not\nclosed** and <!-- open."
    )
    assert [(block.kind, block.text) for chapter in document.chapters for block in chapter.blocks] == [
        (HEADING, "Title"),
        (HEADING, "Two ##"),
        (HEADING, "C#"),
        (
            "paragraph",
            "A bold and soft and quiet and gone word, a link and image, tags **kept**, not closed and <!-- open.",
        ),
    ]
