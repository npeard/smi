#!/usr/bin/env python
"""Fail when a Typst slide deck has overflowed onto extra pages.

Typst has no notion of a slide. `report/lib/theme.typ` makes one level-1
heading start a new page (`pagebreak(weak: true)`) on a fixed page size, so a
slide whose content outgrows the page simply continues onto the next one --
no warning, exit status zero, a deck that looks fine until someone pages
through the render. That is precisely the check a human stops performing.

Count the slide boundaries in the source and the pages in the compiled PDF
and require them to agree. Expected pages = one title page + one per level-1
heading. A mismatch means some slide overflowed, and the difference tells you
how many extra pages to hunt for.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# A level-1 heading at the start of a line is what theme.typ turns into a new
# page. Deliberately not a general Typst parser: `==` (level 2) and `=` inside
# a code block or math span are not slide boundaries, and the negative
# lookahead on `=` keeps the former out.
HEADING = re.compile(r'^=(?!=)\s+\S', re.MULTILINE)

# `/Type /Page`, excluding `/Type /Pages` (the node holding the page tree).
# Only `s` may be excluded here: a page object's dictionary continues straight
# into the next key, so `/Type/Page/Parent ...` is the common form and a
# lookahead that also rejected `/` would match nothing at all.
PDF_PAGE = re.compile(rb'/Type\s*/Page(?!s)')

# The theme renders a title page before the first heading whenever a title was
# supplied, so account for it rather than hardcoding an off-by-one.
TITLE_ARG = re.compile(r'^\s*title:\s*"[^"]', re.MULTILINE)


def has_title_page(source: Path) -> bool:
    """Whether the theme will render a title page for this document."""
    return bool(TITLE_ARG.search(source.read_text(encoding='utf-8')))


def count_slides(source: Path) -> int:
    """Expected page count: the title page, if any, plus one per slide."""
    text = source.read_text(encoding='utf-8')
    return len(HEADING.findall(text)) + (1 if TITLE_ARG.search(text) else 0)


def count_pages(pdf: Path) -> int:
    """Number of page objects in a compiled PDF."""
    return len(PDF_PAGE.findall(pdf.read_bytes()))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--pdf', required=True, type=Path)
    args = parser.parse_args()

    expected = count_slides(args.source)
    actual = count_pages(args.pdf)

    if expected <= 1:
        print(
            f'{args.source}: found no level-1 headings, so there is nothing '
            f'to check. Has the slide syntax changed?',
            file=sys.stderr,
        )
        return 1

    if expected != actual:
        # Spell out the arithmetic rather than assuming a title page: a deck
        # built without one makes "1 title + N slides" send the reader looking
        # for an overflow in the wrong place, and this message is the whole
        # diagnostic the build gate emits.
        headings = expected - (1 if has_title_page(args.source) else 0)
        breakdown = (
            f'1 title + {headings} slides'
            if headings != expected
            else f'{headings} slides'
        )
        print(
            f'{args.source.name}: expected {expected} pages ({breakdown}) but '
            f'{args.pdf.name} has {actual}. {actual - expected} slide(s) '
            f'overflowed -- trim content or shrink a figure, then rebuild.',
            file=sys.stderr,
        )
        return 1

    print(f'{args.source.name}: {actual} pages, one per slide')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
