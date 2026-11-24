import cv2
import numpy as np
import pytest

from lunarcrater.imageio import ImageLoadError, decode_image_bytes, limit_size, load_image, read_pds4, to_uint8

PDS4_LABEL = """<?xml version="1.0" encoding="UTF-8"?>
<Product_Observational xmlns="http://pds.nasa.gov/pds4/pds/v1">
  <File_Area_Observational>
    <File><file_name>{name}</file_name></File>
    <Array_2D_Image>
      <offset unit="byte">{offset}</offset>
      <axes>2</axes>
      <axis_index_order>Last Index Fastest</axis_index_order>
      <Element_Array><data_type>UnsignedLSB2</data_type></Element_Array>
      <Axis_Array><axis_name>Line</axis_name><elements>{h}</elements><sequence_number>1</sequence_number></Axis_Array>
      <Axis_Array><axis_name>Sample</axis_name><elements>{w}</elements><sequence_number>2</sequence_number></Axis_Array>
    </Array_2D_Image>
  </File_Area_Observational>
</Product_Observational>
"""


def write_pds4(tmp_path, arr, offset=16):
    img = tmp_path / "strip.img"
    img.write_bytes(b"\0" * offset + arr.astype("<u2").tobytes())
    xml = tmp_path / "strip.xml"
    xml.write_text(PDS4_LABEL.format(name=img.name, offset=offset, h=arr.shape[0], w=arr.shape[1]))
    return xml


def test_to_uint8_16bit_and_rgb():
    a = (np.arange(100 * 80).reshape(100, 80) * 7).astype(np.uint16)
    u = to_uint8(a)
    assert u.dtype == np.uint8 and u.min() == 0 and u.max() == 255
    rgb = np.dstack([np.full((10, 10), 100, np.uint8)] * 3)
    assert to_uint8(rgb).shape == (10, 10)


def test_decode_png_bytes():
    ok, buf = cv2.imencode(".png", np.full((40, 50), 128, np.uint16))
    assert decode_image_bytes(buf.tobytes()).shape == (40, 50)


@pytest.mark.parametrize("data", [b"", b"not an image at all"])
def test_decode_rejects_garbage(data):
    with pytest.raises(ImageLoadError):
        decode_image_bytes(data)


def test_read_pds4(tmp_path):
    arr = (np.random.default_rng(0).random((64, 96)) * 4000).astype(np.uint16)
    xml = write_pds4(tmp_path, arr)
    assert np.array_equal(read_pds4(xml), arr)
    assert load_image(xml).dtype == np.uint8


def test_read_pds4_missing_data(tmp_path):
    xml = write_pds4(tmp_path, np.zeros((8, 8), np.uint16))
    (tmp_path / "strip.img").unlink()
    with pytest.raises(ImageLoadError):
        read_pds4(xml)


def test_limit_size():
    img, s = limit_size(np.zeros((4000, 1000), np.uint8), 2000)
    assert img.shape == (2000, 500) and s == 0.5
    img, s = limit_size(np.zeros((100, 100), np.uint8), 2000)
    assert s == 1.0
