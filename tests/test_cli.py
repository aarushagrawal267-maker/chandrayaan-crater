import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

from lunarcrater.cli import main
from lunarcrater.synthetic import generate_scene

from .test_imageio import write_pds4

ROOT = Path(__file__).resolve().parent.parent


def test_cli_detect_writes_outputs(tmp_path):
    img = tmp_path / "moon.png"
    cv2.imwrite(str(img), generate_scene(256, seed=11).image)
    rc = main(["detect", str(img), "--mpp", "5", "--method", "hough", "--csv", str(tmp_path / "c.csv"), "--json", str(tmp_path / "a.json"), "--annotated", str(tmp_path / "a.png")])
    assert rc == 0
    assert (tmp_path / "c.csv").read_text().startswith("id,")
    assert "stats" in json.loads((tmp_path / "a.json").read_text())
    assert cv2.imread(str(tmp_path / "a.png")).shape == (256, 256, 3)


def test_cli_detect_missing_file(tmp_path):
    assert main(["detect", str(tmp_path / "missing.png")]) == 2


def test_prepare_script_tiles_pds4(tmp_path):
    arr = (np.random.default_rng(1).random((300, 400)) * 3000 + 10).astype(np.uint16)
    xml = write_pds4(tmp_path, arr)
    out = tmp_path / "tiles"
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "prepare_chandrayaan.py"), str(xml), "--mpp", "0.25", "--downsample", "2", "--tile", "128", "--out", str(out)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    meta = json.loads((out / "tiles.json").read_text())
    assert meta["meters_per_pixel"] == 0.5 and len(meta["tiles"]) == 2  # 150x200 -> 128x128 + 128x72 tiles; 22-row strip skipped


def test_cli_detect_broken_model(tmp_path, capsys):
    img = tmp_path / "moon.png"
    cv2.imwrite(str(img), generate_scene(128, seed=3).image)
    bad = tmp_path / "bad.pt"
    bad.write_text("not a model")
    assert main(["detect", str(img), "--model", str(bad)]) == 2
    assert "could not be loaded" in capsys.readouterr().err
