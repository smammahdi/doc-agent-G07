"""PDF slicing and rendering. The only module that touches the book file."""

import cv2
import fitz
import numpy as np

from .config import DPI


def subset_bytes(pdf_path, page_indices):
    """A new in-memory PDF holding just `page_indices` (0-based), in that order.

    Document AI's sync endpoint caps at 15 pages / 20 MB, and the book is 65 MB, so every
    online call ships a slice rather than the whole volume.
    """
    src = fitz.open(pdf_path)
    dst = fitz.open()
    for i in page_indices:
        dst.insert_pdf(src, from_page=i, to_page=i)
    data = dst.tobytes()
    dst.close()
    src.close()
    return data


def page_sizes_px(pdf_path, indices, dpi=DPI):
    """-> {slice_position: (w, h)} in rendered pixels, without rendering anything.

    `(rect * Matrix(z, z)).irect` is exactly how PyMuPDF sizes a pixmap -- verified equal
    to get_pixmap(dpi=300) on pages 0/20/73/300/700/1033, where naive rounding was off by
    one pixel. Matters because these sizes convert normalized boxes into the same pixel
    space figextract wrote.
    """
    z = dpi / 72
    m = fitz.Matrix(z, z)
    doc = fitz.open(pdf_path)
    out = {}
    for k, i in enumerate(indices):
        r = (doc[i].rect * m).irect
        out[k] = (r.width, r.height)
    doc.close()
    return out


def render(pdf_path, index, dpi=DPI):
    """-> BGR ndarray for one 0-based page index, at the same dpi figextract used."""
    doc = fitz.open(pdf_path)
    pix = doc[index].get_pixmap(dpi=dpi)
    arr = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)
    doc.close()
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR if pix.n == 3 else cv2.COLOR_RGBA2BGR)


def save_crop(img, box, path):
    x0, y0, x1, y1 = box
    crop = img[max(0, y0) : y1, max(0, x0) : x1]
    if crop.size == 0:
        return False
    cv2.imwrite(str(path), crop)
    return True


def ink(img, box):
    """(dark fraction, contrast) with an ABSOLUTE threshold -- same test figextract uses,
    because Otsu adapts on blank paper and binarises the grain into 'ink'."""
    x0, y0, x1, y1 = box
    crop = img[max(0, y0) : y1, max(0, x0) : x1]
    if crop.size == 0:
        return 0.0, 0.0
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return round(float((g < 128).mean()), 4), round(float(np.std(g)), 1)
