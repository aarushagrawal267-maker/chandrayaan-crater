/* Browser demo: answers the UI's /api/* requests inside the page, so the same app.js works without a server.
 * - the U-Net runs in onnxruntime-web (ONNX export of models/crater_unet.pt, checked against PyTorch at build time)
 * - tiling, ring extraction, statistics and landing-site screening run as the project's own Python code in Pyodide
 * Built by deploy/browser/build.py. */
"use strict";
(() => {
  const PYODIDE_URL = "https://cdn.jsdelivr.net/pyodide/v0.27.7/full/";
  const ORT_URL = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";
  const realFetch = window.fetch.bind(window);
  const base = new URL("./", document.baseURI);
  const local = (p) => new URL(p, base).href;

  function setStatus(text, error = false) {
    const el = document.getElementById("demo-status");
    if (el) { el.textContent = text; el.classList.toggle("error", error); }
  }
  function loadScript(src) {
    return new Promise((resolve, reject) => {
      const s = document.createElement("script");
      s.src = src; s.crossOrigin = "anonymous";
      s.onload = resolve; s.onerror = () => reject(new Error("Could not load " + src));
      document.head.appendChild(s);
    });
  }
  const getJson = async (p) => { const r = await realFetch(local(p)); if (!r.ok) throw new Error(`${p}: HTTP ${r.status}`); return r.json(); };

  const manifest = getJson("data/manifest.json");
  const samples = getJson("data/samples.json");

  // ---- U-Net (onnxruntime-web, single thread: a static host cannot enable cross-origin isolation)
  let session = null;
  const modelReady = (async () => {
    await loadScript(ORT_URL + "ort.min.js");
    const ort = window.ort;
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.wasmPaths = ORT_URL;
    ort.env.logLevel = "error";
    const buf = await (await realFetch(local("model/crater_unet.onnx"))).arrayBuffer();
    session = await ort.InferenceSession.create(buf, { executionProviders: ["wasm"] });
  })();

  async function infer(bytes, shape) {
    const [n, h, w] = shape.toJs ? shape.toJs() : shape;
    const u8 = bytes.toJs ? bytes.toJs() : bytes;
    const x = new Float32Array(u8.byteLength / 4);
    new Uint8Array(x.buffer).set(u8);
    const out = await session.run({ input: new window.ort.Tensor("float32", x, [n, 1, h, w]) });
    return out.prob.data;
  }

  // ---- Python (Pyodide + numpy + OpenCV + the lunarcrater package)
  const pythonReady = (async () => {
    await loadScript(PYODIDE_URL + "pyodide.js");
    const py = await window.loadPyodide({ indexURL: PYODIDE_URL });
    await py.loadPackage(["numpy", "opencv-python"], { messageCallback: () => {} });
    const m = await manifest;
    py.FS.mkdirTree("/home/pyodide/demo/lunarcrater");
    for (const f of m.python_files) {
      const r = await realFetch(local("py/" + f));
      if (!r.ok) throw new Error(`py/${f}: HTTP ${r.status}`);
      py.FS.writeFile("/home/pyodide/demo/" + f, await r.text());
    }
    py.registerJsModule("crater_js", { infer });
    py.runPython("import sys; sys.path.insert(0, '/home/pyodide/demo')");
    return py.pyimport("web_api");
  })();

  const engineReady = Promise.all([modelReady, pythonReady]);
  engineReady
    .then(() => setStatus("Ready: everything runs on this device."))
    .catch((e) => { console.warn(e); setStatus("The in-browser engine could not load (" + e.message + "). Try reloading, or run the full app locally.", true); });
  setStatus("Loading the in-browser engine (about 40 MB on the first visit)…");

  const json = (body, status = 200) => new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json" } });

  async function handle(path, init) {
    if (path === "/api/health") {
      let error = null;
      try { await modelReady; } catch (e) { error = "The U-Net could not be loaded in this browser: " + e.message; }
      return json({ status: "ok", version: (await manifest).version, model_available: !error, model_error: error, methods: ["unet", "hough"] });
    }
    if (path === "/api/samples") {
      return json((await samples).map(({ ground_truth, ...s }) => ({ ...s, ground_truth_count: (ground_truth || []).length })));
    }
    if (path === "/api/model") {
      const [evaluation, history] = await Promise.all([getJson("data/metrics.json"), getJson("data/training_history.json")]);
      return json({ model_available: session !== null, evaluation, training_history: history });
    }
    if (path === "/api/detect" && init && (init.method || "GET").toUpperCase() === "POST") {
      const fd = init.body;
      const file = fd.get("file"), sample = fd.get("sample");
      let bytes, source, gt = null;
      if (file && file.name) {
        if (file.size > 40 * 1024 * 1024) return json({ detail: "File too large (max 40 MB)." }, 413);
        bytes = new Uint8Array(await file.arrayBuffer()); source = file.name;
      } else if (sample) {
        const s = (await samples).find((x) => x.file === sample);
        if (!s) return json({ detail: "Unknown sample" }, 404);
        bytes = new Uint8Array(await (await realFetch(local("api/samples/" + encodeURIComponent(sample)))).arrayBuffer());
        source = sample; gt = JSON.stringify(s.ground_truth || []);
      } else return json({ detail: "Upload an image or choose a sample" }, 422);
      const method = fd.get("method") || "unet";
      try {
        if (method === "unet") await engineReady; else await pythonReady;
      } catch (e) {
        return json({ detail: "The in-browser engine could not load: " + e.message }, 503);
      }
      const web = await pythonReady;
      const text = await web.detect_json(bytes, source, gt, method, fd.get("meters_per_pixel") || "5",
        fd.get("min_confidence") || "0.6", fd.get("lander_footprint_m") || "10");
      const res = JSON.parse(text);
      if (res.detail !== undefined && res.status !== undefined && !res.catalogue) return json({ detail: res.detail }, res.status);
      return json(res);
    }
    return json({ detail: "Not available in the browser demo" }, 404);
  }

  window.fetch = async (input, init) => {
    const url = new URL(typeof input === "string" ? input : input.url, location.href);
    if (url.origin === location.origin && url.pathname.startsWith("/api/") && !url.pathname.startsWith("/api/samples/")) {
      try { return await handle(url.pathname, init); }
      catch (e) { console.warn(e); return json({ detail: "Browser demo error: " + e.message }, 500); }
    }
    return realFetch(input, init);
  };
})();
