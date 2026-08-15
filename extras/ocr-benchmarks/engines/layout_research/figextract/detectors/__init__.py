"""Detector registry. `build(name, **kw)` is the only way the pipeline gets a detector."""

import inspect

from .base import Detector
from .doclayout_yolo import DocLayoutYOLO
from .eynollah import Eynollah
from .orphan_ink import OrphanInk
from .ppdoclayout_v3 import PPDocLayoutV3

REGISTRY = {
    OrphanInk.name: OrphanInk,
    DocLayoutYOLO.name: DocLayoutYOLO,
    PPDocLayoutV3.name: PPDocLayoutV3,
    Eynollah.name: Eynollah,
}
NAMES = list(REGISTRY)

__all__ = ["Detector", "REGISTRY", "NAMES", "build"]


def build(name, **kw):
    if name not in REGISTRY:
        raise KeyError(f"unknown detector {name!r}; have {NAMES}")
    cls = REGISTRY[name]
    # the CLI passes a common bag of options; each detector takes only what it understands
    ok = set(inspect.signature(cls.__init__).parameters) - {"self"}
    return cls(**{k: v for k, v in kw.items() if k in ok and v is not None})
