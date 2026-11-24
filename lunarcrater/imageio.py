"""Loading orbital images (PNG/JPG/TIFF, 8/16-bit or float) and Chandrayaan-2 PDS4 products."""
from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np

MAX_SIDE = 2048  # images larger than this are downscaled for interactive analysis


class ImageLoadError(ValueError):
    pass


def to_uint8(arr: np.ndarray, low_pct: float = 0.5, high_pct: float = 99.5) -> np.ndarray:
    """Convert any single-band array to uint8 with a robust percentile stretch."""
    arr = np.asarray(arr)
    if arr.ndim == 3:
        if arr.shape[2] == 4:
            arr = arr[:, :, :3]
        if arr.shape[2] == 3:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY) if arr.dtype == np.uint8 else arr.mean(axis=2)
        else:
            arr = arr[:, :, 0]
    if arr.ndim != 2:
        raise ImageLoadError(f"Expected a 2-D image, got shape {arr.shape}")
    if arr.dtype == np.uint8:
        return arr
    a = arr.astype(np.float64)
    finite = np.isfinite(a)
    if not finite.any():
        raise ImageLoadError("Image contains no valid pixels")
    lo, hi = np.percentile(a[finite], [low_pct, high_pct])
    if hi <= lo:
        hi = lo + 1.0
    a = np.where(finite, a, lo)
    return (np.clip((a - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)


def decode_image_bytes(data: bytes) -> np.ndarray:
    if not data:
        raise ImageLoadError("The uploaded file is empty")
    buf = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ImageLoadError("Could not decode the file. Use PNG, JPG or TIFF (8/16-bit, single band or RGB).")
    return to_uint8(img)


def load_image(path: str | os.PathLike) -> np.ndarray:
    path = Path(path)
    if path.suffix.lower() == ".xml":
        return to_uint8(read_pds4(path))
    if path.suffix.lower() == ".img":
        xml = path.with_suffix(".xml")
        if xml.exists():
            return to_uint8(read_pds4(xml))
        raise ImageLoadError(f"{path.name} is a raw PDS image; its .xml label must sit next to it")
    if not path.exists():
        raise ImageLoadError(f"File not found: {path}")
    return decode_image_bytes(path.read_bytes())


def limit_size(img: np.ndarray, max_side: int = MAX_SIDE) -> tuple[np.ndarray, float]:
    """Downscale so the longest side is ``max_side``. Returns (image, scale) with scale <= 1."""
    h, w = img.shape[:2]
    side = max(h, w)
    if side <= max_side:
        return img, 1.0
    scale = max_side / side
    out = cv2.resize(img, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)
    return out, scale


# --------------------------------------------------------------------------- PDS4
_PDS_DTYPES = {
    "UnsignedByte": "u1",
    "SignedByte": "i1",
    "UnsignedLSB2": "<u2",
    "SignedLSB2": "<i2",
    "UnsignedMSB2": ">u2",
    "SignedMSB2": ">i2",
    "UnsignedLSB4": "<u4",
    "SignedLSB4": "<i4",
    "IEEE754LSBSingle": "<f4",
    "IEEE754MSBSingle": ">f4",
    "IEEE754LSBDouble": "<f8",
    "IEEE754MSBDouble": ">f8",
}


def _strip(tag: str) -> str:
    return tag.split("}", 1)[-1]


def _find(el: ET.Element, name: str) -> ET.Element | None:
    for child in el.iter():
        if _strip(child.tag) == name:
            return child
    return None


def read_pds4(label_path: str | os.PathLike) -> np.ndarray:
    """Read the first ``Array_2D_Image`` from a PDS4 label (the format used by ISRO PRADAN
    for Chandrayaan-2 OHRC/TMC-2 calibrated products)."""
    label_path = Path(label_path)
    root = ET.parse(label_path).getroot()
    file_area = _find(root, "File_Area_Observational")
    if file_area is None:
        raise ImageLoadError("PDS4 label has no File_Area_Observational")
    fname = _find(file_area, "file_name")
    arr_el = _find(file_area, "Array_2D_Image")
    if arr_el is None:
        arr_el = _find(file_area, "Array_3D_Image")
    if fname is None or arr_el is None:
        raise ImageLoadError("PDS4 label has no image array")
    offset_el = _find(arr_el, "offset")
    offset = int(offset_el.text) if offset_el is not None and offset_el.text else 0
    dtype_el = _find(arr_el, "data_type")
    if dtype_el is None or dtype_el.text not in _PDS_DTYPES:
        raise ImageLoadError(f"Unsupported PDS4 data_type: {None if dtype_el is None else dtype_el.text}")
    dims: list[tuple[int, int]] = []
    for ax in arr_el:
        if _strip(ax.tag) == "Axis_Array":
            elements = int(_find(ax, "elements").text)
            seq = int(_find(ax, "sequence_number").text)
            dims.append((seq, elements))
    dims.sort()
    shape = tuple(e for _, e in dims)
    data_file = label_path.parent / fname.text.strip()
    if not data_file.exists():
        raise ImageLoadError(f"Data file {data_file.name} referenced by the label was not found")
    dtype = np.dtype(_PDS_DTYPES[dtype_el.text])
    count = int(np.prod(shape))
    arr = np.fromfile(data_file, dtype=dtype, count=count, offset=offset)
    if arr.size != count:
        raise ImageLoadError("Data file is shorter than the label describes")
    arr = arr.reshape(shape)
    if arr.ndim == 3:
        arr = arr[0]
    return arr
