"""Detection record + box maths. No I/O, no models -- so it is trivially testable."""

from typing import NamedTuple


class Detection(NamedTuple):
    cls: str  # detector's own class name, kept verbatim for auditability
    score: float
    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def box(self):
        return (self.x0, self.y0, self.x1, self.y1)

    @property
    def area(self):
        return max(0, self.x1 - self.x0) * max(0, self.y1 - self.y0)


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union else 0.0


def merge_boxes(boxes, gap=0):
    """Union boxes that overlap or sit within `gap` px. Engravings fragment; plates don't."""
    boxes = [list(b) for b in boxes]
    changed = True
    while changed:
        changed = False
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                a, b = boxes[i], boxes[j]
                if (
                    a[0] - gap < b[2]
                    and b[0] - gap < a[2]
                    and a[1] - gap < b[3]
                    and b[1] - gap < a[3]
                ):
                    boxes[i] = [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]
                    boxes.pop(j)
                    changed = True
                    break
            if changed:
                break
    return [tuple(b) for b in boxes]


def match_nearest(fig, candidates, max_gap):
    """Nearest candidate that overlaps `fig` horizontally and sits within `max_gap` px
    above or below it. Used to bind a caption or a `Fig. N` label to its figure."""
    best, best_gap = None, max_gap
    for c in candidates:
        if min(fig.x1, c.x1) - max(fig.x0, c.x0) <= 0:  # no horizontal overlap
            continue
        gap = c.y0 - fig.y1 if c.y0 >= fig.y1 else fig.y0 - c.y1
        if 0 <= gap < best_gap:
            best, best_gap = c, gap
    return best
