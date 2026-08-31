"""Statistical detector for AI-generated face images.

These features run at **onboarding time only** -- never in the auth path, where
the 50 ms budget rules out image work entirely.

Each feature targets a documented property of generative image stacks rather
than anything about the person depicted, which keeps the detector well away from
proxying for appearance:

``spectral_peak_ratio``
    Transposed-convolution and pixel-shuffle upsampling leave a periodic
    checkerboard, which appears as regular spikes in the Fourier spectrum of the
    high-pass residual. Camera images have smoothly decaying spectra.

``residual_kurtosis``
    Sensor noise is broadband and close to Gaussian; generated high-pass
    residuals are structured and heavier-tailed.

``color_corr_anomaly``
    A generator produces all three channels from one latent, so channels are
    over-correlated relative to the demosaiced output of a Bayer sensor.

``saturation_clip_ratio``
    Real captures clip highlights routinely. Generators trained on normalised
    data rarely reach the rails.

None of these is decisive alone; together they separate the classes well, and
they hand the L4 model interpretable inputs rather than an opaque score.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from contracts.schemas import VerificationSignals


def _high_pass_residual(grey: np.ndarray) -> np.ndarray:
    """3x3 Laplacian residual -- strips content, keeps the generation fingerprint."""
    k = np.array([[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]], dtype=np.float32)
    h, w = grey.shape
    out = np.zeros_like(grey)
    for dy in range(3):
        for dx in range(3):
            if k[dy, dx] == 0.0:
                continue
            out[1 : h - 1, 1 : w - 1] += k[dy, dx] * grey[dy : h - 3 + dy + 1, dx : w - 3 + dx + 1]
    return out[1 : h - 1, 1 : w - 1]


def spectral_peak_ratio(grey: np.ndarray) -> float:
    """Energy at the upsampling grid frequencies over local background energy.

    Checkerboard artifacts concentrate at the Nyquist quarter-points, so we
    compare energy there against a ring of neighbouring frequencies. A ratio
    near 1 means no periodic structure; well above 1 means a regular grid.
    """
    resid = _high_pass_residual(grey)
    spec = np.abs(np.fft.fftshift(np.fft.fft2(resid)))
    h, w = spec.shape
    cy, cx = h // 2, w // 2

    peaks: list[float] = []
    background: list[float] = []
    for fy, fx in ((h // 4, w // 4), (h // 4, 0), (0, w // 4), (h // 4, -(w // 4))):
        py, px = (cy + fy) % h, (cx + fx) % w
        peaks.append(float(spec[py, px]))
        ring = []
        for dy in (-3, -2, 2, 3):
            for dx in (-3, -2, 2, 3):
                ring.append(float(spec[(py + dy) % h, (px + dx) % w]))
        background.append(float(np.median(ring)))

    bg = float(np.mean(background))
    return float(np.mean(peaks) / bg) if bg > 1e-9 else 0.0


def residual_kurtosis(grey: np.ndarray) -> float:
    resid = _high_pass_residual(grey).ravel()
    if resid.std() < 1e-9:
        return 0.0
    return float(stats.kurtosis(resid, fisher=True, bias=False))


def color_corr_anomaly(img: np.ndarray) -> float:
    """How far channel cross-correlation sits above natural-image expectation."""
    if img.ndim != 3 or img.shape[2] < 3:
        return 0.0
    r, g, b = (img[..., i].ravel() for i in range(3))
    resid = []
    for a, c in ((r, g), (g, b), (r, b)):
        if a.std() < 1e-9 or c.std() < 1e-9:
            resid.append(0.0)
        else:
            resid.append(abs(float(np.corrcoef(a, c)[0, 1])))
    #: Typical channel correlation for natural images sits near 0.85; generated
    #: images push toward 1.0.
    return float(max(0.0, np.mean(resid) - 0.85) / 0.15)


def saturation_clip_ratio(img: np.ndarray) -> float:
    """Fraction of pixels pinned at the rails."""
    return float(np.mean((img <= 0.004) | (img >= 0.996)))


def extract_artifact_features(img: np.ndarray) -> dict[str, float]:
    """All four artifact features for one RGB image in ``[0, 1]``."""
    grey = (
        img[..., :3] @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
        if img.ndim == 3
        else img.astype(np.float32)
    )
    return {
        "spectral_peak_ratio": spectral_peak_ratio(grey),
        "residual_kurtosis": residual_kurtosis(grey),
        "color_corr_anomaly": color_corr_anomaly(img),
        "saturation_clip_ratio": saturation_clip_ratio(img),
    }


def build_verification_signals(
    img: np.ndarray,
    *,
    is_synthetic: bool,
    rng: np.random.Generator,
    doc_quality: float = 1.0,
) -> VerificationSignals:
    """Combine measured artifact features with simulated vendor signals.

    The artifact features are genuinely *measured* from the image. The three
    vendor signals are simulated, because we are modelling what a verification
    vendor would report rather than reimplementing one -- drawn from separate
    genuine and synthetic distributions with deliberate overlap, so the task
    stays hard.
    """
    feats = extract_artifact_features(img)

    if is_synthetic:
        template_match = float(np.clip(rng.beta(6.0, 2.2) * doc_quality, 0, 1))
        exif = float(np.clip(rng.beta(1.8, 4.0), 0, 1))
        liveness = float(np.clip(rng.beta(4.5, 2.5), 0, 1))
    else:
        template_match = float(np.clip(rng.beta(8.0, 1.6), 0, 1))
        exif = float(np.clip(rng.beta(6.0, 1.8), 0, 1))
        liveness = float(np.clip(rng.beta(9.0, 1.5), 0, 1))

    return VerificationSignals(
        template_match_score=template_match,
        exif_consistency=exif,
        liveness_score=liveness,
        spectral_peak_ratio=feats["spectral_peak_ratio"],
        residual_kurtosis=feats["residual_kurtosis"],
        color_corr_anomaly=feats["color_corr_anomaly"],
        saturation_clip_ratio=feats["saturation_clip_ratio"],
    )


ARTIFACT_FEATURE_NAMES: tuple[str, ...] = (
    "spectral_peak_ratio",
    "residual_kurtosis",
    "color_corr_anomaly",
    "saturation_clip_ratio",
)
