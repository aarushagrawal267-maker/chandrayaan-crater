#!/usr/bin/env python
"""Turn a Chandrayaan-2 TMC-2 / OHRC product (or any large lunar image) into browser-sized PNG tiles.

Chandrayaan-2 data is distributed by ISRO's Indian Space Science Data Centre through
PRADAN (https://pradan.issdc.gov.in, free registration required) as PDS4 products: an
``.xml`` label plus a raw ``.img`` data file.  Strips are very long (tens of thousands of
lines), so this script

1. reads the PDS4 label (or a PNG/JPG/TIFF),
2. optionally averages ``--downsample`` x ``--downsample`` pixel blocks (e.g. OHRC 0.25 m -> 1 m),
3. cuts the image into ``--tile`` px squares, contrast-stretches each to 8-bit PNG and
4. writes ``tiles.json`` with each tile's offset and its resolution in metres/pixel.

Examples::

    python scripts/prepare_chandrayaan.py ch2_tmc_ndn_2021xxxx_d_img_d18.xml --mpp 5 --out data/tmc_tiles
    python scripts/prepare_chandrayaan.py ohrc_strip.xml --mpp 0.25 --downsample 4 --tile 1024
    python -m lunarcrater detect data/tmc_tiles/tile_r000_c000.png --mpp 5 --csv craters.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lunarcrater.imageio import ImageLoadError, read_pds4, to_uint8  # noqa: E402


def load_raw(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".xml":
        return read_pds4(path)
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ImageLoadError(f"Cannot read {path}")
    if img.ndim == 3:
        img = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY)
    return img


def block_mean(a: np.ndarray, k: int) -> np.ndarray:
    if k <= 1:
        return a
    h, w = (a.shape[0] // k) * k, (a.shape[1] // k) * k
    return a[:h, :w].astype(np.float32).reshape(h // k, k, w // k, k).mean(axis=(1, 3))


def prepare(src: Path, out: Path, mpp: float, tile: int = 1024, downsample: int = 1, min_valid: float = 0.5) -> list[dict]:
    raw = load_raw(src)
    arr = block_mean(raw, downsample)
    eff_mpp = mpp * downsample
    out.mkdir(parents=True, exist_ok=True)
    tiles = []
    h, w = arr.shape
    for r, y in enumerate(range(0, h, tile)):
        for c, x in enumerate(range(0, w, tile)):
            t = arr[y : y + tile, x : x + tile]
            if t.shape[0] < 64 or t.shape[1] < 64:
                continue
            valid = np.isfinite(t) & (t != 0)
            if valid.mean() < min_valid:  # skip no-data borders
                continue
            name = f"tile_r{r:03d}_c{c:03d}.png"
            cv2.imwrite(str(out / name), to_uint8(t))
            tiles.append({"file": name, "row_offset_px": y, "col_offset_px": x, "height": int(t.shape[0]), "width": int(t.shape[1]), "meters_per_pixel": eff_mpp})
    (out / "tiles.json").write_text(json.dumps({"source": src.name, "meters_per_pixel": eff_mpp, "tiles": tiles}, indent=1))
    return tiles


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", type=Path, help="PDS4 .xml label, or PNG/JPG/TIFF")
    ap.add_argument("--mpp", type=float, required=True, help="native resolution in metres/pixel (TMC-2: 5, OHRC: 0.25)")
    ap.add_argument("--out", type=Path, default=Path("data/tiles"))
    ap.add_argument("--tile", type=int, default=1024)
    ap.add_argument("--downsample", type=int, default=1)
    args = ap.parse_args(argv)
    try:
        tiles = prepare(args.source, args.out, args.mpp, args.tile, args.downsample)
    except (ImageLoadError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"wrote {len(tiles)} tiles at {args.mpp * args.downsample:g} m/px to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
