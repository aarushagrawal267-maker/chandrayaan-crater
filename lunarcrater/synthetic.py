"""Procedural lunar-surface generator with exact crater ground truth.

Real, expert-labelled Chandrayaan-2 crater masks are not openly redistributable, so the
model in this project is trained on *simulated* orbital images whose crater positions are
known exactly.  Each image is produced in three steps:

1. **Terrain** - a fractal height field made by spectral synthesis (power spectrum
   ``P(f) ~ f^-beta``), which mimics the rolling regolith background of the lunar maria.
2. **Craters** - simple bowl-shaped craters are stamped onto the height field.  Radii follow
   a power-law size-frequency distribution (many small, few large craters, as on the real
   Moon).  The profile uses the classic morphometry of fresh simple lunar craters
   (depth ~ 0.2 D, rim height ~ 0.04 D, ejecta decaying as r^-3) and a random
   *freshness* factor that degrades older craters.
3. **Illumination** - the height field is hill-shaded with a Lambertian model for a random
   sun azimuth and a low sun elevation (5-45 deg), similar to the grazing light in
   Chandrayaan-2 OHRC/TMC-2 images of the south-polar region.  Albedo variation, sensor
   noise and gamma are added on top.

The generator returns the image, a crater-rim ring mask (the training target, following
the DeepMoon approach of Silburt et al., 2019) and the crater list ``[(x, y, r), ...]``.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class Crater:
    x: float
    y: float
    r: float
    freshness: float = 1.0
    visible: bool = True

    def as_tuple(self) -> tuple[float, float, float]:
        return (float(self.x), float(self.y), float(self.r))


@dataclass
class SyntheticScene:
    image: np.ndarray  # uint8, HxW
    rim_mask: np.ndarray  # uint8 {0,1}, HxW
    craters: list[Crater] = field(default_factory=list)
    sun_azimuth: float = 0.0
    sun_elevation: float = 30.0


def fractal_terrain(size: int, rng: np.random.Generator, beta: float = 2.6, amplitude: float = 6.0) -> np.ndarray:
    """Gaussian random field with a power-law spectrum; returns heights in pixel units."""
    fy = np.fft.fftfreq(size)[:, None]
    fx = np.fft.fftfreq(size)[None, :]
    f = np.sqrt(fx**2 + fy**2)
    f[0, 0] = 1.0
    amp = f ** (-beta / 2.0)
    amp[0, 0] = 0.0
    phase = rng.uniform(0, 2 * np.pi, (size, size))
    spectrum = amp * np.exp(1j * phase)
    field_ = np.real(np.fft.ifft2(spectrum))
    field_ -= field_.mean()
    std = field_.std() or 1.0
    return field_ / std * amplitude


def sample_radii(rng: np.random.Generator, n: int, r_min: float, r_max: float, slope: float = 2.0) -> np.ndarray:
    """Draw radii from a truncated power law with cumulative slope ``-slope``."""
    u = rng.uniform(0, 1, n)
    a = r_min ** (-slope)
    b = r_max ** (-slope)
    return (a - u * (a - b)) ** (-1.0 / slope)


def crater_profile(d: np.ndarray, r: float, freshness: float) -> np.ndarray:
    """Height change for normalised radial distance ``d`` (0 at centre, 1 at rim)."""
    depth = 0.2 * 2 * r * freshness
    rim = 0.04 * 2 * r * freshness
    inside = -depth + (depth + rim) * np.clip(d, 0, 1) ** 2
    outside = rim * np.clip(d, 1, None) ** -3.0
    # flatten the floor slightly for degraded craters
    return np.where(d < 1.0, inside, outside)


def stamp_crater(height: np.ndarray, c: Crater) -> None:
    h, w = height.shape
    reach = c.r * 3.0
    x0, x1 = int(max(0, np.floor(c.x - reach))), int(min(w, np.ceil(c.x + reach) + 1))
    y0, y1 = int(max(0, np.floor(c.y - reach))), int(min(h, np.ceil(c.y + reach) + 1))
    if x0 >= x1 or y0 >= y1:
        return
    yy, xx = np.mgrid[y0:y1, x0:x1]
    d = np.sqrt((xx - c.x) ** 2 + (yy - c.y) ** 2) / c.r
    prof = crater_profile(d, c.r, c.freshness)
    patch = height[y0:y1, x0:x1]
    # A new crater excavates what was there: inside the bowl, blend the old surface
    # towards the local mean so that the new crater overprints older ones.
    inside = d < 1.0
    if inside.any():
        local_mean = float(patch[inside].mean())
        wgt = np.clip(1.15 - d, 0, 1) * c.freshness
        patch[:] = patch * (1 - wgt) + local_mean * wgt
    patch += prof


def crater_visible(c: Crater, base_slope2: np.ndarray, ratio: float = 1.7) -> bool:
    """A crater is labelled only if its wall slope clearly exceeds the background roughness
    around it - otherwise even a human expert could not see it in the rendered image."""
    h, w = base_slope2.shape
    reach = max(2.0, 1.5 * c.r)
    x0, x1 = int(max(0, c.x - reach)), int(min(w, c.x + reach + 1))
    y0, y1 = int(max(0, c.y - reach)), int(min(h, c.y + reach + 1))
    if x0 >= x1 or y0 >= y1:
        return False
    local = float(np.sqrt(base_slope2[y0:y1, x0:x1].mean()))
    wall = 2 * (0.4 + 0.08) * c.freshness  # rim slope of the profile in crater_profile()
    return wall >= ratio * local


def hillshade(height: np.ndarray, azimuth_deg: float, elevation_deg: float) -> np.ndarray:
    gy, gx = np.gradient(height)
    az = np.deg2rad(azimuth_deg)
    el = np.deg2rad(elevation_deg)
    lx, ly, lz = np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)
    norm = np.sqrt(gx**2 + gy**2 + 1.0)
    shade = (-gx * lx - gy * ly + lz) / norm
    return np.clip(shade, 0, None)


def rim_mask_for(craters: list[Crater], shape: tuple[int, int], min_thickness: int = 1) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    for c in craters:
        thick = max(min_thickness, int(round(c.r * 0.12)))
        cv2.circle(mask, (int(round(c.x * 4)), int(round(c.y * 4))), int(round(c.r * 4)), 1, thick, lineType=cv2.LINE_8, shift=2)
    return mask


def generate_scene(
    size: int = 256,
    seed: int | None = None,
    n_craters: tuple[int, int] = (12, 40),
    r_min: float = 3.0,
    r_max: float | None = None,
    sun_elevation: float | None = None,
) -> SyntheticScene:
    rng = np.random.default_rng(seed)
    r_max = r_max or size / 5.0
    height = fractal_terrain(size, rng, beta=rng.uniform(3.0, 3.6), amplitude=rng.uniform(0.4, 2.0))
    height += fractal_terrain(size, rng, beta=2.2, amplitude=rng.uniform(0.01, 0.07))  # fine regolith texture
    gy, gx = np.gradient(height)
    base_slope2 = gx**2 + gy**2  # background roughness, used to decide which craters are visible

    n = int(rng.integers(n_craters[0], n_craters[1] + 1))
    radii = np.sort(sample_radii(rng, n, r_min, r_max, slope=rng.uniform(1.6, 2.4)))[::-1]
    craters: list[Crater] = []
    margin = 0.5
    for r in radii:  # big (old) craters first, small (young) craters overprint them
        cx = rng.uniform(-margin * r, size + margin * r)
        cy = rng.uniform(-margin * r, size + margin * r)
        fresh = float(rng.uniform(0.4, 1.0))
        c = Crater(cx, cy, float(r), fresh)
        c.visible = crater_visible(c, base_slope2)
        stamp_crater(height, c)
        craters.append(c)

    # sub-resolution "noise" craters that are too small to label
    for r in sample_radii(rng, int(rng.integers(20, 80)), 0.8, r_min * 0.8):
        stamp_crater(height, Crater(rng.uniform(0, size), rng.uniform(0, size), float(r), float(rng.uniform(0.3, 1.0))))

    az = float(rng.uniform(0, 360))
    el = float(rng.uniform(5, 45))
    if sun_elevation is not None:
        el = float(sun_elevation)
    shade = hillshade(height, az, el)
    albedo = 0.75 + 0.25 * np.tanh(fractal_terrain(size, rng, beta=3.2, amplitude=1.0))
    img = shade * albedo
    img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.4, 1.0))
    img = img / (np.percentile(img, 99.5) + 1e-6)
    img = np.clip(img, 0, 1) ** rng.uniform(0.7, 1.3)
    img = img + rng.normal(0, rng.uniform(0.01, 0.05), img.shape)
    image = (np.clip(img, 0, 1) * 255).astype(np.uint8)

    # Ground truth: craters at least r_min with centre inside the frame
    labelled = [c for c in craters if c.visible and 0 <= c.x < size and 0 <= c.y < size and c.r >= r_min]
    # The mask also contains partially visible craters so the network is not punished
    # for seeing rims of craters whose centre is just outside the image.
    visible = [c for c in craters if c.visible and -c.r < c.x < size + c.r and -c.r < c.y < size + c.r]
    mask = rim_mask_for(visible, (size, size))
    return SyntheticScene(image=image, rim_mask=mask, craters=labelled, sun_azimuth=az, sun_elevation=el)
