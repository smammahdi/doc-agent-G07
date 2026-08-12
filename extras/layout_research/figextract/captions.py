"""Reading the printed `Fig. N.` label and the caption, and binding them to a figure.

Two facts about this book drive everything here, both measured rather than assumed:

1. The `Fig. N.` label sits INSIDE the top of the detected figure box, not above it.
2. The caption sits BELOW the figure, in its own line, often in small italic.

The embedded OCR layer recovers only 156 distinct figure numbers out of a series running
to 289, and garbles others (`Fig. 17.` -> `Rig: I.`), so both regions get re-OCR'd.
"""

import re

import cv2

FIG_RE = re.compile(r"\bF[ir1l]g\s*[.,]?\s*(\d{1,4})", re.I)  # tolerates OCR'd 'Fig'
LABEL_STRIP_FRAC = 0.045  # top slice of the figure box that holds the label
BAND_FRAC = 0.075  # fallback search band above/below when no region detected


def make_ocr():
    """-> (callable, kind). Tesseract if present, else the embedded text layer."""
    try:
        import pytesseract

        pytesseract.get_tesseract_version()

        def _ocr(crop, psm=6):
            if crop is None or crop.size == 0:
                return ""
            return pytesseract.image_to_string(crop, config=f"--psm {psm}")

        return _ocr, "tesseract"
    except Exception:

        def _ocr(crop, psm=6):
            return ""

        return _ocr, "unavailable"


def read_label_strip(img, box, ocr):
    """OCR the top slice inside the figure box, where the `Fig. N.` label actually lives.

    The label is small italic display type, so we try 2x upscale and single-line mode
    first -- that alone took label recovery from 1/14 to 7/14 on the sample pages.
    """
    strip = int(LABEL_STRIP_FRAC * img.shape[0])
    y1 = min(box[3], box[1] + strip)
    if y1 - box[1] < 8:
        return ""
    crop = img[box[1] : y1, box[0] : box[2]]
    if crop.size == 0:
        return ""
    big = cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    text = ""
    for im in (big, crop):
        for psm in (7, 6):
            text = ocr(im, psm)
            if FIG_RE.search(text):
                return text
    return text


def read_band(img, box, ocr, above):
    """Fallback when the detector gives no caption/label region: read a fixed band."""
    h, w = img.shape[:2]
    band = int(BAND_FRAC * h)
    y0, y1 = (max(0, box[1] - band), box[1]) if above else (box[3], min(h, box[3] + band))
    if y1 - y0 < 10:
        return ""
    return ocr(img[y0:y1, max(0, box[0] - 20) : min(w, box[2] + 20)])


def read_region(img, det, ocr, psm=6):
    crop = img[det.y0 : det.y1, det.x0 : det.x1]
    return ocr(crop, psm) if crop.size else ""


def parse(label_text, caption_text, whole_caption=False):
    """-> (fig_label, caption). Label is hunted in both texts; caption comes from the second."""
    m = FIG_RE.search(label_text) or FIG_RE.search(caption_text)
    fig_label = f"Fig. {m.group(1)}" if m else None

    if whole_caption:
        # caption_text came from a detected caption region: all of it is caption, so
        # join the lines instead of truncating at the first one
        caption = re.sub(r"\s+", " ", FIG_RE.sub("", caption_text)).strip(" .-—_|")
    else:
        caption = ""
        for line in caption_text.splitlines():
            s = re.sub(r"\s+", " ", line).strip(" .-—_|")
            if len(s) >= 8 and not FIG_RE.fullmatch(s.strip()):
                caption = s
                break
    if not caption:  # label band sometimes carries it inline: "Fig. 45. The lungs."
        tail = re.sub(r"\s+", " ", FIG_RE.sub("", label_text)).strip(" .-—_|")
        if len(tail) >= 8:
            caption = tail
    return fig_label, caption
