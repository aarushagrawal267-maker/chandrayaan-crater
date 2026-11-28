"""Build (and optionally publish) the browser demo of Chandrayaan Crater Lens as a static site.

    pip install onnx onnxruntime huggingface_hub          # export, parity check, publish
    python deploy/browser/build.py --out build/browser-demo
    python deploy/browser/build.py --out build/browser-demo --space <hf-user>/chandrayaan-crater-lens

The page is the normal web UI (app/static). A small script answers its /api/* requests inside the browser:
the U-Net runs as an ONNX export of models/crater_unet.pt in onnxruntime-web, and everything else (tiling,
ring extraction, merging, statistics, landing-site screening) is the same lunarcrater Python code, run by
Pyodide. Before anything is written, the ONNX model is checked against PyTorch on every tile the detector
feeds it for the bundled samples; the build stops if they disagree.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from lunarcrater import __version__  # noqa: E402
from lunarcrater.detect import DEFAULT_MODEL, CraterDetector  # noqa: E402
from lunarcrater.imageio import load_image  # noqa: E402

PY_MODULES = ["__init__.py", "analysis.py", "classical.py", "detect.py", "extract.py", "imageio.py", "metrics.py"]
MAX_ABS_DIFF = 1e-4
GITHUB = "https://github.com/aarushagrawal267-maker/chandrayaan-crater"
BANNER = f"""<div class="demo-banner" role="note">
  <strong>Browser demo &mdash; the full app runs locally.</strong>
  The U-Net runs in your browser (ONNX export, onnxruntime-web) and the analysis is the project's own Python code
  running in Pyodide. Nothing is uploaded. Images are analysed at up to 1024&nbsp;px here.
  <span id="demo-status" class="demo-status">Loading the in-browser engine&hellip;</span>
  <a href="{GITHUB}" target="_blank" rel="noopener">Source and full app</a>
</div>"""
DEMO_CSS = """<style>
.demo-banner { padding: 8px 24px; font-size: 13px; line-height: 1.45; color: var(--muted); background: #15130d;
  border-bottom: 1px solid var(--line); overflow-wrap: anywhere; }
.demo-banner strong { color: var(--accent); font-weight: 600; }
.demo-status { display: inline-block; margin: 0 6px; color: var(--text); }
.demo-status.error { color: var(--danger); }
@media (max-width: 640px) { .demo-banner { padding: 8px 16px; } }
</style>"""


def export_onnx(path: Path) -> None:
    """Export the shipped U-Net with the sigmoid included: (N, 1, H, W) tiles -> (N, H, W) probabilities."""
    import torch

    from lunarcrater.model import load_model

    model, _ = load_model(DEFAULT_MODEL, "cpu")
    model.eval()

    class Prob(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            return torch.sigmoid(self.m(x))[:, 0]

    torch.onnx.export(Prob(model), torch.zeros(1, 1, 256, 256), str(path), input_names=["input"], output_names=["prob"],
                      opset_version=17, dynamic_axes={"input": {0: "n", 2: "h", 3: "w"}, "prob": {0: "n", 1: "h", 2: "w"}},
                      dynamo=False)


def check_parity(onnx_path: Path) -> dict:
    """Compare ONNX and PyTorch on every batch the detector produces for the samples, and the final craters."""
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    det = CraterDetector()
    report = {"max_abs_diff": 0.0, "samples": []}
    for s in json.loads((ROOT / "samples" / "samples.json").read_text()):
        img = load_image(ROOT / "samples" / s["file"])
        worst = 0.0

        def drive(use_onnx: bool):
            nonlocal worst
            steps = det.detect_steps(img, "unet", 0.6)
            try:
                batch = next(steps)
                while True:
                    ref = det._infer(batch)
                    got = sess.run(["prob"], {"input": batch[:, None]})[0]
                    worst = max(worst, float(np.abs(ref - got).max()))
                    batch = steps.send(got if use_onnx else ref)
            except StopIteration as stop:
                return stop.value[0]

        torch_craters, onnx_craters = drive(False), drive(True)
        same = torch_craters == onnx_craters
        report["samples"].append({"file": s["file"], "craters_torch": len(torch_craters), "craters_onnx": len(onnx_craters),
                                  "identical_craters": same, "max_abs_diff": worst})
        report["max_abs_diff"] = max(report["max_abs_diff"], worst)
    return report


def build(out: Path) -> dict:
    if out.exists():
        shutil.rmtree(out)
    (out / "model").mkdir(parents=True)
    onnx_path = out / "model" / "crater_unet.onnx"
    export_onnx(onnx_path)
    parity = check_parity(onnx_path)
    for s in parity["samples"]:
        print(f"  {s['file']}: max |onnx - torch| = {s['max_abs_diff']:.2e}, craters {s['craters_torch']} / {s['craters_onnx']}"
              f" ({'identical' if s['identical_craters'] else 'DIFFERENT'})")
    if parity["max_abs_diff"] > MAX_ABS_DIFF:
        raise SystemExit(f"ONNX export disagrees with PyTorch (max abs diff {parity['max_abs_diff']:.2e})")

    shutil.copytree(ROOT / "app" / "static", out / "static", ignore=shutil.ignore_patterns("index.html"))
    shutil.copy2(HERE / "browser-api.js", out / "static" / "browser-api.js")
    html = (ROOT / "app" / "static" / "index.html").read_text(encoding="utf-8")
    html = html.replace('<link rel="stylesheet" href="/static/style.css">',
                        '<link rel="stylesheet" href="/static/style.css">\n  ' + DEMO_CSS, 1)
    html = html.replace("</header>", "</header>\n" + BANNER, 1)
    html = html.replace('<script src="/static/app.js"></script>',
                        '<script src="/static/browser-api.js"></script>\n<script src="/static/app.js"></script>', 1)
    assert "browser-api.js" in html and "demo-banner" in html, "index.html layout changed; update build.py"
    (out / "index.html").write_text(html, encoding="utf-8")

    (out / "api" / "samples").mkdir(parents=True)
    for p in (ROOT / "samples").glob("*.png"):
        shutil.copy2(p, out / "api" / "samples" / p.name)
    (out / "data").mkdir()
    for name in ("metrics.json", "training_history.json"):
        shutil.copy2(ROOT / "models" / name, out / "data" / name)
    shutil.copy2(ROOT / "samples" / "samples.json", out / "data" / "samples.json")
    (out / "py" / "lunarcrater").mkdir(parents=True)
    for name in PY_MODULES:
        shutil.copy2(ROOT / "lunarcrater" / name, out / "py" / "lunarcrater" / name)
    shutil.copy2(HERE / "web_api.py", out / "py" / "web_api.py")
    files = [f"lunarcrater/{n}" for n in PY_MODULES] + ["web_api.py"]
    (out / "data" / "manifest.json").write_text(json.dumps({"version": __version__, "python_files": files,
                                                            "parity": parity}, indent=1))
    shutil.copy2(HERE / "README.md", out / "README.md")
    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    print(f"wrote {out} ({size / 1e6:.1f} MB, {sum(1 for p in out.rglob('*') if p.is_file())} files)")
    return parity


def publish(out: Path, space: str) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(space, repo_type="space", space_sdk="static", exist_ok=True)
    api.upload_folder(repo_id=space, repo_type="space", folder_path=str(out), commit_message="Publish browser demo",
                      delete_patterns=["*"])
    owner, name = space.split("/", 1)
    print(f"Space page: https://huggingface.co/spaces/{space}")
    print(f"App URL:    https://{owner.lower()}-{name.lower().replace('_', '-').replace('.', '-')}.static.hf.space")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(ROOT / "build" / "browser-demo"))
    ap.add_argument("--space", help="also publish to this Hugging Face Space (static SDK, free)")
    a = ap.parse_args()
    out = Path(a.out)
    build(out)
    if a.space:
        publish(out, a.space)


if __name__ == "__main__":
    main()
