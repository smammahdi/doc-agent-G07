"""Zero-ML baseline: figure ink is ink the OCR text layer does not claim.

No weights, no dependencies beyond OpenCV. It exists as the floor the learned models must
beat -- on the sample pages it found 7 figures with zero false positives, but its captions
were wrong because this book wraps body text around its plates.
"""

import cv2
import numpy as np

from ..geometry import Detection, merge_boxes
from ..pdf import word_boxes
from .base import Detector

MIN_W, MIN_H = 0.07, 0.045  # a figure is at least this fraction of the page
MIN_AREA = 0.006
CLOSE_FRAC = 0.022  # morphological close, merges engraving strokes into a blob
WORD_PAD = 3  # dilate text-layer boxes before subtracting


class OrphanInk(Detector):
    name = "orphan_ink"
    FIGURE = {"figure"}

    def load(self):
        return self

    def detect(self, page, img):
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        ink = binv > 0

        covered = np.zeros((h, w), np.uint8)
        for x0, y0, x1, y1, _ in word_boxes(page, img):
            cv2.rectangle(
                covered, (x0 - WORD_PAD, y0 - WORD_PAD), (x1 + WORD_PAD, y1 + WORD_PAD), 255, -1
            )

        mask = (ink & (covered == 0)).astype(np.uint8) * 255
        k = max(3, int(CLOSE_FRAC * min(h, w)) | 1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))

        n, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        boxes = []
        for i in range(1, n):
            x, y, bw, bh, area = stats[i]
            if bw < MIN_W * w or bh < MIN_H * h or area < MIN_AREA * w * h:
                continue
            if bw > 0.97 * w and bh > 0.97 * h:
                continue  # page border / scan edge
            # int() matters: cv2 hands back numpy int32, which json cannot serialize
            boxes.append((int(x), int(y), int(x + bw), int(y + bh)))

        merged = merge_boxes(boxes, gap=int(0.02 * min(h, w)))
        return [Detection("figure", 0.0, *b) for b in sorted(merged, key=lambda b: (b[1], b[0]))]
