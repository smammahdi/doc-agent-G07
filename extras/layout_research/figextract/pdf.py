"""PDF rendering and the embedded OCR text layer."""

import cv2
import fitz  # PyMuPDF
import numpy as np

from .config import DPI


def open_book(path):
    return fitz.open(path)


def render(page, dpi=DPI):
    """Page -> BGR ndarray at `dpi`."""
    pix = page.get_pixmap(dpi=dpi)
    arr = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR if pix.n == 3 else cv2.COLOR_RGBA2BGR)


def word_boxes(page, img):
    """Embedded text-layer words, scaled to image pixels -> [(x0,y0,x1,y1,text), ...].

    This is the Internet Archive's own OCR. We never trust it for content (it drops or
    garbles 133 of the book's Fig. labels), but it is a fine map of *where the text is*,
    which is all the orphan-ink detector needs.
    """
    h, w = img.shape[:2]
    sx, sy = w / page.rect.width, h / page.rect.height
    return [
        (int(x0 * sx), int(y0 * sy), int(x1 * sx), int(y1 * sy), t)
        for x0, y0, x1, y1, t, *_ in page.get_text("words")
    ]


def page_ids(doc, pages=None, limit=None):
    """-> [(index, 'pXXXX'), ...]. `pages` is a 1-indexed comma string."""
    if pages:
        idx = [int(p) - 1 for p in str(pages).split(",")]
    else:
        idx = range(min(limit, doc.page_count) if limit else doc.page_count)
    return [(i, f"p{i + 1:04d}") for i in idx]
