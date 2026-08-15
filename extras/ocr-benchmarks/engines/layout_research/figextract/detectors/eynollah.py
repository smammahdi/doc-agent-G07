"""eynollah (Staatsbibliothek zu Berlin) -- the only candidate built FOR historical print.

Upstream: vendor/eynollah, Apache-2.0, actively maintained. Despite the `Type: Keras`
column in `eynollah models list`, the shipped weights are ONNX, so no TensorFlow is
needed. It does pin `onnxruntime-gpu[cuda,cudnn]`, which has no macOS wheel -- install it
`--no-deps` against plain onnxruntime on a Mac.

Models: models_inference_layout_v0_9_1.zip (1.85GB) from Zenodo doi:10.5281/zenodo.17194823.
`EYNOLLAH_MODELS` must point at the directory *containing* `models_eynollah/`.

STATUS: does NOT run on macOS. Two independent blockers, both traced:

  1. `onnxruntime-gpu[cuda,cudnn]` (a hard requirement) has no macOS wheel.
  2. With plain CPU onnxruntime, the extract_images graph fails to load at all:
     `Node (Reshape__474) Op (Reshape) [ShapeInferenceError] Invalid dimension value:
     -2048`. The graph was exported for the GPU runtime.

Run it on Linux + CUDA (the Kaggle box), where its real dependency installs.

Separately, upstream's own `extract_images.run_single()` is stale: it unpacks 4 values
from `run_enhancement()`, which returns 2 in the current `eynollah.py`. So the shipped
image-extraction entry point does not run as-is on any platform. `detect()` below calls
the core directly to work around that.

It is a pixel-wise segmenter rather than a box detector, so each detected image region's
contour is reduced to its bounding box for comparability with the other detectors. That
discards eynollah's mask precision on purpose: a mask and a box cannot be compared without
flattening one of them, and the question here is which figures get found.
"""

import cv2
import numpy as np

from ..config import EYNOLLAH_MODELS
from ..geometry import Detection
from .base import Detector


class Eynollah(Detector):
    name = "eynollah"
    FIGURE = {"image"}
    CAPTION: set = set()  # eynollah has no caption class; captions fall back to bands
    LABEL: set = set()

    def __init__(self, models=None, conf=0.0, dpi=300):
        self.models, self.conf, self.dpi = models or EYNOLLAH_MODELS, conf, dpi
        self._runner = None

    def load(self):
        from pathlib import Path

        base = Path(self.models)
        if not (base / "models_eynollah").exists():
            raise RuntimeError(
                f"eynollah models not found under {base}/models_eynollah\n"
                "  curl -L -o m.zip 'https://zenodo.org/records/21381102/files/"
                "models_inference_layout_v0_9_1.zip?download=1' && unzip m.zip -d "
                f"{base}"
            )
        try:
            from eynollah.extract_images import EynollahImageExtractor
            from eynollah.model_zoo.model_zoo import EynollahModelZoo
        except ImportError as e:
            raise RuntimeError(
                f"eynollah not importable: {e}\n"
                "  pip install --no-deps -e development/vendor/eynollah\n"
                "  pip install onnxruntime ocrd scikit-learn scikit-image tabulate"
            ) from e
        zoo = EynollahModelZoo(basedir=str(base))
        self._runner = EynollahImageExtractor(model_zoo=zoo, enable_plotting=False)
        self.impl = "official/onnx"
        print(f"eynollah loaded from {base}")
        return self

    def detect(self, page, img):
        r = self._runner
        # Mirror upstream run_single(), minus the PAGE-XML writing we do not need.
        # NOTE: upstream's own extract_images.run_single() is stale -- it unpacks 4 values
        # from run_enhancement(), which returns 2 in the current eynollah.py, so their
        # image-extraction entry point does not run as shipped. We call the core directly.
        # cache_images returns a dict (img / img_grayscale / dpi / name) that the rest of
        # the pipeline mutates in place, adding 'img_res'.
        cached = r.cache_images(image_pil=_to_pil(img), dpi=self.dpi)
        num_col, _ = r.run_enhancement(cached)
        img_res = cached.get("img_res", cached["img"])
        _, _, _, polygons, _, _, _ = r.get_regions_light_v_extract_only_images(img_res, num_col)

        h, w = img.shape[:2]
        dets = []
        for poly in polygons or []:
            arr = np.asarray(poly, dtype=np.int32).reshape(-1, 2)
            if arr.size < 4:
                continue
            x, y, bw, bh = cv2.boundingRect(arr)
            if bw < 0.03 * w or bh < 0.03 * h:
                continue
            dets.append(
                Detection("image", 1.0, int(x), int(y), int(min(w, x + bw)), int(min(h, y + bh)))
            )
        return sorted(dets, key=lambda d: (d.y0, d.x0))


def _to_pil(img):
    from PIL import Image

    return Image.fromarray(img[:, :, ::-1])
