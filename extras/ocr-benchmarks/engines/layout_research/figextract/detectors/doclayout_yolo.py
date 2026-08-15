"""DocLayout-YOLO (opendatalab), YOLOv10m-doclayout trained on DocStructBench.

Two backends, same weights:

* `official` -- the upstream `doclayout_yolo` package from vendor/DocLayout-YOLO.
  This is the authors' own inference path (YOLOv10.predict), so it is the faithful one.
* `onnx`     -- the ONNX export, needing only onnxruntime. Survives environments where
  the upstream package's pinned ultralytics/torch cannot be installed (notably an
  offline Kaggle image on torch 2.10).

Weights are AGPL-3.0. Output is (300, 6) = x0,y0,x1,y1,conf,cls -- YOLOv10 is NMS-free.
"""

import ast

import cv2
import numpy as np

from ..config import DOCLAYOUT_ONNX, DOCLAYOUT_PT, VENDOR
from ..geometry import Detection
from .base import Detector

IMGSZ = 1024
CLASSES = {
    0: "title",
    1: "plain text",
    2: "abandon",
    3: "figure",
    4: "figure_caption",
    5: "table",
    6: "table_caption",
    7: "table_footnote",
    8: "isolate_formula",
    9: "formula_caption",
}


class DocLayoutYOLO(Detector):
    name = "doclayout_yolo"
    FIGURE = {"figure", "table"}
    CAPTION = {"figure_caption", "table_caption"}
    LABEL: set = set()  # this model has no separate label class

    def __init__(self, backend="auto", weights=None, conf=0.25):
        self.backend, self.weights, self.conf = backend, weights, conf
        self._sess = self._model = None
        self.classes = CLASSES

    def load(self):
        if self.backend in ("auto", "official"):
            try:
                self._load_official()
                return self
            except Exception as e:
                if self.backend == "official":
                    raise RuntimeError(
                        f"official doclayout_yolo unavailable: {e}\n"
                        f"pip install -e {VENDOR}/DocLayout-YOLO"
                    ) from e
                print(f"  doclayout: official backend unavailable ({e}); using ONNX")
        self._load_onnx()
        return self

    def _load_official(self):
        import torch
        from doclayout_yolo import YOLOv10

        w = self.weights or DOCLAYOUT_PT
        if not str(w).endswith(".pt"):
            raise FileNotFoundError("official backend needs the .pt checkpoint")
        self._model = YOLOv10(str(w))
        self._dev = (
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )
        self.impl = f"official/{self._dev}"
        print(f"DocLayout-YOLO official on {self._dev}")

    def _load_onnx(self):
        import onnxruntime as ort

        w = self.weights or DOCLAYOUT_ONNX
        # CoreML is excluded: it cannot build a plan for this graph on macOS.
        prov = [
            p
            for p in ("CUDAExecutionProvider", "CPUExecutionProvider")
            if p in ort.get_available_providers()
        ] or ["CPUExecutionProvider"]
        self._sess = ort.InferenceSession(str(w), providers=prov)
        names = self._sess.get_modelmeta().custom_metadata_map.get("names")
        if names:
            self.classes = {int(k): v for k, v in ast.literal_eval(names).items()}
        self.impl = f"onnx/{prov[0]}"
        print(f"DocLayout-YOLO ONNX on {prov[0]} ({len(self.classes)} classes)")

    def detect(self, page, img):
        return self._detect_official(img) if self._model else self._detect_onnx(img)

    def _detect_official(self, img):
        res = self._model.predict(
            img, imgsz=IMGSZ, conf=self.conf, device=self._dev, verbose=False
        )[0]
        out = []
        for b in res.boxes:
            x0, y0, x1, y1 = (int(round(v)) for v in b.xyxy[0].tolist())
            out.append(
                Detection(res.names[int(b.cls.item())], float(b.conf.item()), x0, y0, x1, y1)
            )
        return sorted(out, key=lambda d: (d.y0, d.x0))

    def _detect_onnx(self, img):
        h, w = img.shape[:2]
        r = min(IMGSZ / h, IMGSZ / w)
        nh, nw = int(round(h * r)), int(round(w * r))
        canvas = np.full((IMGSZ, IMGSZ, 3), 114, np.uint8)
        canvas[:nh, :nw] = cv2.resize(img, (nw, nh))
        x = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)[None]
        x = np.ascontiguousarray(x, np.float32) / 255.0

        out = self._sess.run(None, {"images": x})[0][0]  # (300,6), already NMS-free
        dets = []
        for x0, y0, x1, y1, sc, cl in out:
            if sc < self.conf:
                continue
            # padding was bottom/right only, so undoing the letterbox is a divide
            dets.append(
                Detection(
                    self.classes.get(int(cl), str(int(cl))),
                    float(sc),
                    int(max(0, x0 / r)),
                    int(max(0, y0 / r)),
                    int(min(w, x1 / r)),
                    int(min(h, y1 / r)),
                )
            )
        return sorted(dets, key=lambda d: (d.y0, d.x0))
