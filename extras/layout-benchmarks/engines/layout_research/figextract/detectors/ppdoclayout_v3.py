"""PP-DocLayoutV3 (Baidu) via the official Transformers integration.

PaddlePaddle publish both the weights (`PaddlePaddle/PP-DocLayoutV3_safetensors`) and the
`PPDocLayoutV3ForObjectDetection` class in Transformers, so this IS the official path --
just distributed through HF rather than the 2GB PaddleOCR clone. Needs transformers>=5.14.

RT-DETR instance segmentation: 25 classes, polygon masks, and reading order in one pass.
Its edge over DocLayout-YOLO on this book is NOT detection (they measured identical) but
that `figure_title` and `vision_footnote` are separate regions, so the label and caption
get read from exact crops instead of guessed strips.
"""

from ..config import PPV3_DIR
from ..geometry import Detection
from .base import Detector


class PPDocLayoutV3(Detector):
    name = "ppdoclayout_v3"
    FIGURE = {"image", "chart", "table"}
    CAPTION = {"vision_footnote", "figure_title"}  # figure_title only if no footnote
    LABEL = {"figure_title"}

    def __init__(self, weights=None, conf=0.4):
        self.weights, self.conf = weights or PPV3_DIR, conf

    def load(self):
        import torch
        from transformers import AutoImageProcessor, AutoModelForObjectDetection

        d = str(self.weights)
        self.proc = AutoImageProcessor.from_pretrained(d)
        self.model = AutoModelForObjectDetection.from_pretrained(d).eval()
        # No MPS: this model's post-processing needs float64, which Metal does not
        # support ("Cannot convert a MPS Tensor to float64"). CUDA on Kaggle, CPU on Mac.
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.dev)
        self.impl = f"transformers/{self.dev}"
        print(f"PP-DocLayoutV3 on {self.dev} ({len(self.model.config.id2label)} classes)")
        return self

    def detect(self, page, img):
        import torch
        from PIL import Image

        im = Image.fromarray(img[:, :, ::-1])
        with torch.no_grad():
            out = self.model(**self.proc(images=[im], return_tensors="pt").to(self.dev))
        # Post-processing wants float64, which MPS cannot do, so it happens on CPU while
        # the forward pass still runs on the accelerator.
        for k in ("logits", "pred_boxes"):
            if getattr(out, k, None) is not None:
                setattr(out, k, getattr(out, k).cpu())
        res = self.proc.post_process_object_detection(
            out, target_sizes=torch.tensor([im.size[::-1]])
        )[0]
        # results arrive in reading order; keep that order, it is information
        return [
            Detection(
                self.model.config.id2label[lb.item()],
                float(sc.item()),
                *(int(round(v)) for v in bx.tolist()),
            )
            for sc, lb, bx in zip(res["scores"], res["labels"], res["boxes"], strict=True)
            if sc.item() >= self.conf
        ]

    def split(self, dets):
        figs = [d for d in dets if d.cls in self.FIGURE]
        lbls = [d for d in dets if d.cls == "figure_title"]
        caps = [d for d in dets if d.cls == "vision_footnote"]
        return figs, caps, lbls
