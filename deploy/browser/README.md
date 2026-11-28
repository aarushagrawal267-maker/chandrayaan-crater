---
title: Chandrayaan Crater Lens
colorFrom: gray
colorTo: yellow
sdk: static
app_file: index.html
pinned: false
license: mit
short_description: Lunar crater detection, running in your browser
---

# Chandrayaan Crater Lens: browser demo

Automatic lunar crater detection, crater statistics and landing-site screening for Chandrayaan-2 style orbital
images. Pick a built-in sample or upload a PNG/JPG image: a small U-Net marks crater rims, every crater becomes a
circle with a position, a diameter in metres and a confidence, and the page reports crater density, coverage, the
size-frequency distribution and candidate landing sites.

This is a **browser demo; the full app runs locally**. The trained U-Net runs in your browser as an ONNX export
(onnxruntime-web, checked against the PyTorch model when the demo is built), and the rest of the pipeline is the
project's own Python code running in Pyodide. Nothing is uploaded. Images are analysed at up to 1024 px here
(2048 px in the full app).

Independent student project, not affiliated with or endorsed by ISRO. The network is trained on *simulated* lunar
terrain; the accuracy numbers on the *Model & evaluation* tab were measured on held-out simulated scenes, not on real
Chandrayaan images.

Source code and the full app: https://github.com/aarushagrawal267-maker/chandrayaan-crater
