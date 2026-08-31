"""Image sources and the perceptual feature extractor that feeds FlyHash.

Two source backends behind one interface:

* :class:`CorpusFaceSource` -- a published deepfake/GAN-detection research
  corpus laid out under ``data/corpus/{real,generated}``. This is the primary
  path for headline metrics. We *consume* generated faces from corpora that
  exist for building detectors; we do not manufacture them.
* :class:`ProceduralFaceSource` -- seeded procedural faces with controllable
  identity similarity and injectable generator artifacts. Fully reproducible and
  needs no download, so the pipeline is always runnable and CI-able.

``get_face_source`` picks the corpus when it is present and falls back to
procedural otherwise, announcing which it used. Nothing downstream cares.

Feature extraction is a DCT perceptual descriptor rather than a learned
embedding: it is deterministic, cheap, has no training dependency, and -- most
importantly -- it is *not* an identity embedding, so it never becomes the kind
of invertible biometric artifact this project exists to avoid.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.fftpack import dct, idct
from scipy.ndimage import gaussian_filter

#: Side length images are normalised to before feature extraction.
IMAGE_SIZE = 128
#: DCT block kept from the top-left corner of the high-passed image.
#:
#: Chosen by measurement, not by taste. JPEG's 8-pixel quantisation grid puts
#: compression energy at ~16 cycles per image, so a wider block admits
#: compression noise into what is supposed to be an identity descriptor -- at
#: block 20 that cost ~14 points of same-ring recall. Sweeping 10..20 against
#: link AUC picked 10 (AUC 0.996, 90% recall at a 0.6% false-link rate). The
#: degradation is gradual rather than a clean cutoff at the blocking frequency,
#: because the high-pass and the blocking interact across bands.
DCT_BLOCK = 10
#: Blur radius whose output is subtracted to remove global colour and lighting.
HIGHPASS_SIGMA = 8.0
#: Block size of the simulated block-transform codec.
_BLOCK = 8
#: Resulting descriptor width: the block minus the (now near-zero) DC term.
FEATURE_DIM = DCT_BLOCK * DCT_BLOCK - 1
#: Width of the procedural identity vector: 12 geometry/colour + 20 texture.
IDENTITY_DIM = 32
_N_GEOMETRY = 12


@lru_cache(maxsize=4)
def _upsampling_grid(size: int) -> np.ndarray:
    """Fixed checkerboard pattern of a transposed-convolution stack, cached."""
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    g = (np.cos(math.pi * xx / 2.0) * np.cos(math.pi * yy / 2.0)).astype(np.float32)
    g += 0.5 * (np.cos(math.pi * xx / 4.0) * np.cos(math.pi * yy / 4.0)).astype(np.float32)
    return g


@lru_cache(maxsize=4)
def _texture_basis(size: int) -> np.ndarray:
    """Fixed 2D Fourier basis the identity texture field is a weighted sum of.

    The basis depends only on canvas size, so it is built once. An identity's
    texture is then a tensordot rather than twenty transcendental evaluations
    over a 128x128 grid per face.
    """
    x, y = _grids(size)
    modes = ((1, 0), (0, 1), (1, 1), (2, 0), (0, 2), (2, 1), (1, 2), (2, 2), (3, 0), (0, 3))
    fields = []
    for u, v in modes:
        phase = 2.0 * math.pi * (u * x + v * y)
        fields.append(np.cos(phase))
        fields.append(np.sin(phase))
    return np.stack(fields).astype(np.float32)


@lru_cache(maxsize=4)
def _grids(size: int) -> tuple[np.ndarray, np.ndarray]:
    """Normalised coordinate grids, cached.

    Every face render needs these and they depend only on the canvas size.
    Rebuilding them per image dominated identity construction -- the simulator
    mints thousands of identities per run, and the red-team loop re-runs that
    many times over.
    """
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    return ((xx - size / 2) / size, (yy - size / 2) / size)


def image_features(img: np.ndarray) -> np.ndarray:
    """Perceptual descriptor for an RGB image array in ``[0, 1]``.

    The image is high-passed before the DCT. Measured directly, a low-frequency
    DCT descriptor is dominated by whatever repaints large regions -- skin tone
    and hair -- so two people with similar colouring produced near-identical
    descriptors regardless of their geometry, and unrelated faces overlapped at
    0.26 against a 0.026 chance floor. Subtracting a heavy blur removes that
    global plateau and leaves the structural detail that actually distinguishes
    people, dropping unrelated overlap to ~0.05.

    Contrast is normalised afterwards so exposure does not survive into the
    descriptor. This is a perceptual descriptor, deliberately *not* a learned
    identity embedding -- it has no training dependency and never becomes the
    kind of invertible biometric artifact this project exists to avoid.
    """
    if img.ndim == 3:
        grey = img[..., :3] @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    else:
        grey = img
    grey = grey.astype(np.float32)
    grey = grey - gaussian_filter(grey, HIGHPASS_SIGMA)
    sd = float(grey.std())
    grey = grey / (sd if sd > 1e-6 else 1.0)
    coeffs = dct(dct(grey, axis=0, norm="ortho"), axis=1, norm="ortho")
    return coeffs[:DCT_BLOCK, :DCT_BLOCK].ravel()[1:]


class DescriptorNormalizer:
    """Population whitening, fitted once on a reference sample of faces.

    Every face is an oval with two eyes, so the raw DCT descriptor is dominated
    by structure that all faces share; the identity-carrying variation is a
    small perturbation riding on a large common mean. Hashing that directly
    makes *all* faces look similar -- measured at 0.26 mean overlap between
    unrelated faces, against a 0.026 chance floor.

    Subtracting the population mean and dividing by per-dimension spread puts
    the identity-varying dimensions on equal footing with the structural ones,
    which is what restores separation. This is the same reason the fly's circuit
    centres receptor responses before the Kenyon layer rather than after.
    """

    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "DescriptorNormalizer":
        X = np.asarray(X, dtype=np.float32)
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0) + 1e-6
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("DescriptorNormalizer must be fitted before use")
        return (np.asarray(x, dtype=np.float32) - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def to_dict(self) -> dict[str, list[float]]:
        assert self.mean_ is not None and self.std_ is not None
        return {"mean": self.mean_.tolist(), "std": self.std_.tolist()}

    @classmethod
    def from_dict(cls, d: dict[str, list[float]]) -> "DescriptorNormalizer":
        n = cls()
        n.mean_ = np.asarray(d["mean"], dtype=np.float32)
        n.std_ = np.asarray(d["std"], dtype=np.float32)
        return n


# --------------------------------------------------------------------------
# Procedural faces
# --------------------------------------------------------------------------


@dataclass
class FaceParams:
    """Geometry and colouring of a procedural face, drawn from an identity vector."""

    face_w: float
    face_h: float
    eye_y: float
    eye_dx: float
    eye_r: float
    brow_h: float
    nose_len: float
    mouth_w: float
    mouth_y: float
    skin: np.ndarray
    hair: np.ndarray

    @classmethod
    def from_vector(cls, v: np.ndarray) -> "FaceParams":
        """Map the leading 12 identity dimensions onto plausible face geometry."""
        v = np.clip(np.asarray(v, dtype=np.float32).ravel()[:_N_GEOMETRY], 0.0, 1.0)
        return cls(
            face_w=0.30 + 0.10 * v[0],
            face_h=0.38 + 0.10 * v[1],
            eye_y=-0.10 + 0.06 * v[2],
            eye_dx=0.13 + 0.05 * v[3],
            eye_r=0.030 + 0.020 * v[4],
            brow_h=0.03 + 0.03 * v[5],
            nose_len=0.10 + 0.07 * v[6],
            mouth_w=0.10 + 0.07 * v[7],
            mouth_y=0.17 + 0.06 * v[8],
            skin=np.array(
                [0.55 + 0.35 * v[9], 0.42 + 0.32 * v[9], 0.35 + 0.30 * v[9]], dtype=np.float32
            )
            * (0.85 + 0.25 * v[10]),
            hair=np.array([0.10 + 0.35 * v[11], 0.08 + 0.28 * v[11], 0.07 + 0.22 * v[11]], dtype=np.float32),
        )


def _identity_texture(texture_vec: np.ndarray, size: int = IMAGE_SIZE) -> np.ndarray:
    """A smooth shading field unique to an identity.

    Twelve geometry parameters is far less identity information than a real face
    carries, and it showed: distinct procedural identities collided because
    faces sharing skin tone and rough proportions became indistinguishable.
    Real faces differ in structure, shading and texture across the whole
    surface, so each identity gets its own low-order 2D Fourier field. Small
    perturbations of the identity vector still yield near-duplicates, which is
    what ring reuse must look like.
    """
    basis = _texture_basis(size)
    n = min(len(texture_vec), basis.shape[0])
    if n == 0:
        return np.zeros((size, size), dtype=np.float32)
    coeffs = (np.asarray(texture_vec[:n], dtype=np.float32) - 0.5) * 2.0
    return np.tensordot(coeffs, basis[:n], axes=(0, 0)) / max(1, n // 2)


def render_face(params: FaceParams, size: int = IMAGE_SIZE, texture_vec: np.ndarray | None = None) -> np.ndarray:
    """Render a simple frontal face. Deliberately schematic, not photorealistic."""
    x, y = _grids(size)

    img = np.ones((size, size, 3), dtype=np.float32) * np.array([0.55, 0.58, 0.62], dtype=np.float32)

    # Hair mass sits slightly above and wider than the face oval.
    hair_m = ((x / (params.face_w * 1.12)) ** 2 + ((y + 0.05) / (params.face_h * 1.12)) ** 2) <= 1.0
    img[hair_m] = params.hair

    face_m = ((x / params.face_w) ** 2 + (y / params.face_h) ** 2) <= 1.0
    img[face_m] = params.skin

    # Soft vertical shading so the surface has structure for the DCT to see.
    shade = np.clip(1.0 - 0.28 * (y + 0.35), 0.72, 1.12)[..., None]
    img = np.where(face_m[..., None], np.clip(img * shade, 0, 1), img)

    for sign in (-1.0, 1.0):
        eye = (((x - sign * params.eye_dx) / params.eye_r) ** 2 + ((y - params.eye_y) / (params.eye_r * 0.62)) ** 2) <= 1.0
        img[eye] = np.array([0.97, 0.97, 0.98], dtype=np.float32)
        pupil = (((x - sign * params.eye_dx) / (params.eye_r * 0.42)) ** 2 + ((y - params.eye_y) / (params.eye_r * 0.42)) ** 2) <= 1.0
        img[pupil] = np.array([0.10, 0.09, 0.12], dtype=np.float32)
        brow = (np.abs(x - sign * params.eye_dx) < params.eye_r * 1.5) & (
            np.abs(y - (params.eye_y - params.brow_h)) < params.eye_r * 0.28
        )
        img[brow] = params.hair * 0.85

    nose = (np.abs(x) < 0.018) & (y > params.eye_y) & (y < params.eye_y + params.nose_len)
    img[nose] = np.clip(params.skin * 0.82, 0, 1)

    mouth = (np.abs(x) < params.mouth_w) & (np.abs(y - params.mouth_y) < 0.016)
    img[mouth] = np.array([0.55, 0.26, 0.26], dtype=np.float32)

    if texture_vec is not None and len(texture_vec):
        tex = _identity_texture(texture_vec, size) * 0.18
        img = img + np.where(face_m, tex, 0.0)[..., None]

    return np.clip(img, 0.0, 1.0)


def _blockiness(img: np.ndarray, strength: float) -> np.ndarray:
    """JPEG-style 8x8 block-DCT quantisation.

    This is the honest confounder in the whole artifact story. Block-transform
    compression imposes an 8x8 periodic structure, which appears in the
    frequency domain as regular peaks -- the *same* signature that
    transposed-convolution upsampling leaves. A detector that has only ever seen
    uncompressed camera images will call every compressed photograph generated.
    Without this the classes separated perfectly at every generator quality,
    which is a property of the simulator rather than a result.

    Quantisation is applied to *DCT coefficients within each block*, with a
    coarser step at higher frequencies, as real block-transform codecs do.
    Quantising pixel values with a fixed step would be algebraically identical
    to quantising the whole image at once and would produce posterisation with
    no block structure at all.
    """
    if strength <= 0.0:
        return img
    h, w, c = img.shape
    if h % _BLOCK or w % _BLOCK:
        return img

    blocks = img.reshape(h // _BLOCK, _BLOCK, w // _BLOCK, _BLOCK, c).transpose(0, 2, 4, 1, 3)
    coeffs = dct(dct(blocks, axis=-1, norm="ortho"), axis=-2, norm="ortho")
    quant = _quant_table(strength)
    coeffs = np.round(coeffs / quant) * quant
    out = idct(idct(coeffs, axis=-1, norm="ortho"), axis=-2, norm="ortho")
    return out.transpose(0, 3, 1, 4, 2).reshape(h, w, c)


def _quant_table(strength: float) -> np.ndarray:
    """Coarser quantisation at higher spatial frequencies, as codecs use.

    Calibrated so the effect is actually visible in the output: measured as the
    ratio of pixel discontinuity across block boundaries to discontinuity
    inside blocks, this table gives 1.1x at strength 0.2 rising to 2.4x at 1.0.
    A finer table quantises nothing on a smooth face and leaves the confounder
    absent while appearing to be present.
    """
    u = np.arange(_BLOCK, dtype=np.float32)
    radial = u[:, None] + u[None, :]
    return (0.004 + 0.30 * strength * (1.0 + radial / 2.0)).astype(np.float32)


def apply_capture_realism(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Approximate a camera-captured image, with realistic variation.

    Real photographs are not a single consistent process: exposure, sensor
    noise, vignetting and compression quality all vary widely between devices
    and conditions. Modelling capture as one fixed pipeline gave every real
    image near-identical spectral statistics, which let a detector separate the
    classes perfectly regardless of generator quality -- a flattering result
    that would not survive a real corpus. So each image draws its own capture
    conditions, including a compression pass whose blocking artifacts overlap
    with the generative signature we are looking for.
    """
    size = img.shape[0]
    x, y = _grids(size)
    r = np.sqrt(x**2 + y**2)

    vignette = float(rng.uniform(0.10, 0.45))
    noise_sd = float(rng.uniform(0.004, 0.030))
    gain_sd = float(rng.uniform(0.010, 0.055))
    exposure = float(rng.uniform(0.95, 1.22))
    jpeg = float(rng.beta(1.4, 2.0))

    img = img * (1.0 - vignette * r**2)[..., None]
    img = img * rng.normal(1.0, gain_sd, size=img.shape).astype(np.float32)
    img = img + rng.normal(0.0, noise_sd, size=img.shape).astype(np.float32)
    img = np.clip(img * exposure, 0.0, 1.0)
    return np.clip(_blockiness(img, jpeg), 0.0, 1.0)


def apply_generator_artifacts(
    img: np.ndarray, rng: np.random.Generator, strength: float = 1.0
) -> np.ndarray:
    """Impose the statistical signature of a generative upsampling stack.

    Three effects, all documented in the deepfake-detection literature:

    1. A periodic checkerboard from transposed-convolution upsampling, which
       shows as regular peaks in the frequency domain.
    2. Over-correlated colour channels, because the generator produces all three
       from a shared latent rather than from three sensor filters.
    3. Compressed dynamic range with almost no highlight clipping.

    This shapes *statistics*, not identity: it does not make an image more
    convincing to a human, only more representative of generated-image
    statistics for the detector to learn against.
    """
    size = img.shape[0]
    grid = _upsampling_grid(size)
    img = img + strength * 0.020 * grid[..., None]

    mean_ch = img.mean(axis=2, keepdims=True)
    img = img + strength * 0.30 * (mean_ch - img)

    img = 0.5 + (img - 0.5) * (1.0 - 0.18 * strength)
    img = img + rng.normal(0.0, float(rng.uniform(0.002, 0.014)), size=img.shape).astype(np.float32)
    img = np.clip(img, 0.0, 1.0)
    # Generated images are shared, saved and re-encoded like any other. If only
    # the real class carried compression blocking, its absence would itself
    # become a giveaway and the detector would be learning our pipeline rather
    # than a generative signature.
    return np.clip(_blockiness(img, float(rng.beta(1.4, 2.0))), 0.0, 1.0)


class FaceSource:
    """Interface shared by the corpus and procedural backends."""

    kind: str = "abstract"

    def sample(
        self,
        *,
        synthetic: bool,
        identity_vec: np.ndarray | None = None,
        artifact_strength: float | None = None,
    ) -> np.ndarray:
        raise NotImplementedError


@dataclass
class ProceduralFaceSource(FaceSource):
    """Seeded procedural faces with controllable identity similarity."""

    seed: int = 0
    artifact_strength: float = 1.0
    kind: str = field(default="procedural", init=False)

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)

    def random_identity_vec(self) -> np.ndarray:
        return self._rng.random(IDENTITY_DIM).astype(np.float32)

    def near_duplicate_vec(self, base: np.ndarray, jitter: float = 0.06) -> np.ndarray:
        """A face from the same generator run -- what ring reuse actually looks like."""
        return np.clip(base + self._rng.normal(0.0, jitter, size=base.shape), 0.0, 1.0).astype(
            np.float32
        )

    def sample(
        self,
        *,
        synthetic: bool,
        identity_vec: np.ndarray | None = None,
        artifact_strength: float | None = None,
    ) -> np.ndarray:
        """Render one face.

        ``artifact_strength`` is the generator-quality dial and is normally set
        by the ring preset: a sloppy operator uses a generator that leaves loud
        upsampling artifacts, a sophisticated one leaves almost none. Left
        unset, it is drawn per image so that a mixed population does not become
        trivially separable -- a uniform strength flatters the detector and does
        not survive contact with a real corpus.
        """
        vec = self.random_identity_vec() if identity_vec is None else np.asarray(identity_vec)
        img = render_face(
            FaceParams.from_vector(vec),
            texture_vec=np.asarray(vec).ravel()[_N_GEOMETRY:],
        )
        if synthetic:
            if artifact_strength is None:
                artifact_strength = float(self._rng.beta(1.6, 1.9) * 2.0)
            strength = float(np.clip(artifact_strength * self.artifact_strength, 0.0, 2.0))
            return apply_generator_artifacts(img, self._rng, strength)
        return apply_capture_realism(img, self._rng)


@dataclass
class CorpusFaceSource(FaceSource):
    """Loads real and generated faces from a downloaded research corpus."""

    root: Path
    seed: int = 0
    kind: str = field(default="corpus", init=False)

    def __post_init__(self) -> None:
        from PIL import Image  # imported lazily so the procedural path needs no PIL

        self._Image = Image
        self._rng = np.random.default_rng(self.seed)
        self._real = sorted((self.root / "real").glob("**/*.*"))
        self._gen = sorted((self.root / "generated").glob("**/*.*"))
        if not self._real or not self._gen:
            raise FileNotFoundError(
                f"corpus at {self.root} needs non-empty real/ and generated/ subdirectories"
            )

    def _load(self, path: Path) -> np.ndarray:
        img = self._Image.open(path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
        return np.asarray(img, dtype=np.float32) / 255.0

    def sample(
        self,
        *,
        synthetic: bool,
        identity_vec: np.ndarray | None = None,
        artifact_strength: float | None = None,
    ) -> np.ndarray:
        """Draw from the corpus. Identity and artifact strength are properties of
        the corpus image itself, so both arguments are accepted and ignored."""
        pool = self._gen if synthetic else self._real
        return self._load(pool[int(self._rng.integers(len(pool)))])


def get_face_source(
    corpus_root: str | Path = "data/corpus", seed: int = 0, prefer_corpus: bool = True
) -> FaceSource:
    """Return the corpus source when it is available, else the procedural one."""
    root = Path(corpus_root)
    if prefer_corpus and root.exists():
        try:
            src = CorpusFaceSource(root=root, seed=seed)
            print(f"[biohash.images] using corpus at {root}")
            return src
        except (FileNotFoundError, ImportError) as exc:
            print(f"[biohash.images] corpus unusable ({exc}); falling back to procedural")
    return ProceduralFaceSource(seed=seed)
