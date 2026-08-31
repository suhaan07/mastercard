"""Structural hashing of identity-document layouts, for template-reuse detection.

The ring signal we are after is that many applicants share a document-generation
artifact -- the same layout engine, the same field positions, the same
proportions. That is a *structural* property, not a photorealism one, so we
detect it structurally and never need to produce anything resembling a real
identity document.

Renders here are deliberately schematic: labelled boxes on a card-shaped canvas.
They exercise exactly the layout-similarity logic a real pipeline needs while
being visibly not identity documents.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from biohash.flyhash import TAG_DIM, FlyHash
from contracts.schemas import SparseTag

#: Canvas the layout is rasterised onto before feature extraction.
DOC_W, DOC_H = 128, 80
#: Grid the layout is summarised on. Coarse on purpose: the same template with
#: different field *contents* must hash the same.
GRID_W, GRID_H = 16, 10
DOC_FEATURE_DIM = GRID_W * GRID_H


@dataclass(frozen=True)
class DocTemplate:
    """A document layout: where the fields sit, not what they say."""

    template_id: str
    #: (x, y, w, h) in normalised [0,1] canvas coordinates.
    fields: tuple[tuple[float, float, float, float], ...]
    portrait_box: tuple[float, float, float, float]
    aspect: float

    @classmethod
    def generate(cls, template_id: str, rng: np.random.Generator) -> "DocTemplate":
        """Invent a plausible card layout: a portrait box and a stack of fields."""
        portrait = (
            float(rng.uniform(0.04, 0.10)),
            float(rng.uniform(0.15, 0.28)),
            float(rng.uniform(0.20, 0.28)),
            float(rng.uniform(0.45, 0.62)),
        )
        n_fields = int(rng.integers(4, 8))
        left = portrait[0] + portrait[2] + float(rng.uniform(0.04, 0.09))
        top = float(rng.uniform(0.12, 0.22))
        gap = float(rng.uniform(0.09, 0.14))
        width = float(rng.uniform(0.34, 0.52))
        fields = tuple(
            (
                left,
                min(0.92, top + i * gap),
                width * float(rng.uniform(0.75, 1.0)),
                float(rng.uniform(0.045, 0.075)),
            )
            for i in range(n_fields)
        )
        return cls(
            template_id=template_id,
            fields=fields,
            portrait_box=portrait,
            aspect=float(rng.uniform(1.5, 1.7)),
        )

    def render(self, rng: np.random.Generator, jitter: float = 0.004) -> np.ndarray:
        """Rasterise the layout.

        ``jitter`` perturbs field positions slightly, standing in for the
        rendering variation between two documents from the same generator. The
        structural hash must survive it -- that is the whole point.
        """
        canvas = np.zeros((DOC_H, DOC_W), dtype=np.float32)

        def _fill(box: tuple[float, float, float, float], value: float) -> None:
            x, y, w, h = box
            x += float(rng.normal(0.0, jitter))
            y += float(rng.normal(0.0, jitter))
            x0, x1 = int(np.clip(x, 0, 1) * DOC_W), int(np.clip(x + w, 0, 1) * DOC_W)
            y0, y1 = int(np.clip(y, 0, 1) * DOC_H), int(np.clip(y + h, 0, 1) * DOC_H)
            if x1 > x0 and y1 > y0:
                canvas[y0:y1, x0:x1] = value

        _fill(self.portrait_box, 0.6)
        for f in self.fields:
            _fill(f, 1.0)
        return canvas


def layout_features(canvas: np.ndarray) -> np.ndarray:
    """Coarse ink-density grid over the layout.

    Downsampling to a 16x10 grid of mean intensity keeps *where the ink is* and
    discards what it says, so two documents from one template hash alike while
    two different templates do not.
    """
    h, w = canvas.shape
    ys = np.linspace(0, h, GRID_H + 1).astype(int)
    xs = np.linspace(0, w, GRID_W + 1).astype(int)
    cells = np.empty((GRID_H, GRID_W), dtype=np.float32)
    for i in range(GRID_H):
        for j in range(GRID_W):
            cell = canvas[ys[i] : ys[i + 1], xs[j] : xs[j + 1]]
            cells[i, j] = float(cell.mean()) if cell.size else 0.0
    return cells.ravel()


class DocTemplateHasher:
    """Hashes document layouts into the same tag space as faces."""

    def __init__(self, seed_id: str = "default", salt: str = "", dim: int = TAG_DIM) -> None:
        self._fly = FlyHash(
            input_dim=DOC_FEATURE_DIM,
            dim=dim,
            sampling_rate=0.15,
            sparsity=0.05,
            seed_id=seed_id,
            salt=salt,
        )
        self.dim = self._fly.dim
        self.hash_length = self._fly.hash_length

    def tag(self, canvas: np.ndarray) -> SparseTag:
        return self._fly.tag(layout_features(canvas))

    def tag_template(
        self, template: DocTemplate, rng: np.random.Generator, jitter: float = 0.004
    ) -> SparseTag:
        return self.tag(template.render(rng, jitter=jitter))
