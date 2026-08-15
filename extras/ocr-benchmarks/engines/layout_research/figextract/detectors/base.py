"""Detector contract.

A detector maps a rendered page to Detections using its OWN class vocabulary -- we never
rewrite a model's class names, so a JSONL row always says what the model actually said.
Each detector declares which of its classes mean "figure", "caption" and "label", and the
pipeline reads those three sets. That keeps the vocabularies comparable without flattening
them, which matters because the comparison between models is the whole point.
"""


class Detector:
    name = "base"
    FIGURE: set = set()  # classes we crop
    CAPTION: set = set()  # classes holding the caption text
    LABEL: set = set()  # classes holding the "Fig. N." label

    def load(self):
        """Acquire weights/sessions. Raise with an actionable message if unavailable."""
        raise NotImplementedError

    def detect(self, page, img):
        """-> list[Detection] in image pixel coords."""
        raise NotImplementedError

    def split(self, dets):
        figs = [d for d in dets if d.cls in self.FIGURE]
        caps = [d for d in dets if d.cls in self.CAPTION]
        lbls = [d for d in dets if d.cls in self.LABEL]
        return figs, caps, lbls
