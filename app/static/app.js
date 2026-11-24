"use strict";
const $ = (s) => document.querySelector(s);
const state = { sample: null, file: null, result: null, baseImg: null, safetyImg: null, hover: null };

// ------------------------------------------------------------------ tabs
document.querySelectorAll(".tab").forEach((b) =>
  b.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t === b));
    document.querySelectorAll(".tabpanel").forEach((p) => p.classList.toggle("active", p.id === "tab-" + b.dataset.tab));
    if (b.dataset.tab === "model") loadModel();
  })
);

// ------------------------------------------------------------------ status + samples
async function init() {
  try {
    const h = await (await fetch("/api/health")).json();
    const pill = $("#status");
    if (h.model_available) { pill.textContent = "U-Net model loaded"; pill.className = "pill ok"; }
    else {
      pill.textContent = "No model: Hough only"; pill.className = "pill warn";
      $("#method").value = "hough";
      $("#method option[value=unet]").disabled = true;
    }
  } catch { $("#status").textContent = "backend offline"; }

  try {
    const samples = await (await fetch("/api/samples")).json();
    const box = $("#samples");
    if (!samples.length) box.innerHTML = '<p class="hint">No samples found. Run <code>python -m lunarcrater make-samples</code>.</p>';
    samples.forEach((s) => {
      const b = document.createElement("button");
      b.type = "button"; b.className = "sample"; b.title = s.description;
      b.innerHTML = `<img src="/api/samples/${encodeURIComponent(s.file)}" alt=""><span>${escapeHtml(s.description.replace("Simulated ", ""))}</span>`;
      b.addEventListener("click", () => selectSample(s, b));
      box.appendChild(b);
    });
  } catch { setMsg("Could not load samples", true); }
}

function selectSample(s, el) {
  state.sample = s.file; state.file = null;
  document.querySelectorAll(".sample").forEach((x) => x.classList.toggle("selected", x === el));
  $("#drop").classList.remove("selected");
  $("#drop-text").innerHTML = "<strong>Upload your own</strong><br>PNG, JPG or TIFF (8/16-bit), max 40 MB";
  $("#mpp").value = s.meters_per_pixel;
  $("#run").disabled = false;
  setMsg(`Sample selected (${s.ground_truth_count} ground-truth craters).`);
}

function selectFile(f) {
  if (!f) return;
  if (f.size > 40 * 1024 * 1024) { setMsg("File is larger than 40 MB.", true); return; }
  state.file = f; state.sample = null;
  document.querySelectorAll(".sample").forEach((x) => x.classList.remove("selected"));
  $("#drop").classList.add("selected");
  $("#drop-text").innerHTML = `<strong>${escapeHtml(f.name)}</strong><br>${(f.size / 1024 / 1024).toFixed(2)} MB - set its resolution below`;
  $("#show-gt").checked = false;
  $("#run").disabled = false;
  setMsg("");
}
$("#file").addEventListener("change", (e) => selectFile(e.target.files[0]));
const drop = $("#drop");
["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", (e) => selectFile(e.dataTransfer.files[0]));
document.querySelectorAll("[data-mpp]").forEach((b) => b.addEventListener("click", () => ($("#mpp").value = b.dataset.mpp)));
$("#conf").addEventListener("input", (e) => ($("#conf-out").textContent = Number(e.target.value).toFixed(2)));

function setMsg(t, err = false) { const m = $("#msg"); m.textContent = t; m.className = "msg" + (err ? " error" : ""); }
function escapeHtml(s) { return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])); }
function loadImage(src) { return new Promise((res, rej) => { const i = new Image(); i.onload = () => res(i); i.onerror = rej; i.src = src; }); }
const fmt = (v, d = 2) => (v === null || v === undefined ? "-" : Number(v).toLocaleString(undefined, { maximumFractionDigits: d }));

// ------------------------------------------------------------------ run detection
$("#run").addEventListener("click", async () => {
  const mpp = parseFloat($("#mpp").value);
  const foot = parseFloat($("#footprint").value);
  if (!(mpp > 0)) return setMsg("Enter the image resolution in metres per pixel.", true);
  if (!(foot > 0)) return setMsg("Enter a lander footprint in metres.", true);
  const fd = new FormData();
  if (state.file) fd.append("file", state.file); else if (state.sample) fd.append("sample", state.sample); else return;
  fd.append("method", $("#method").value);
  fd.append("meters_per_pixel", mpp);
  fd.append("min_confidence", $("#conf").value);
  fd.append("lander_footprint_m", foot);
  const btn = $("#run"); btn.disabled = true; btn.textContent = "Detecting...";
  setMsg("Running the detector...");
  try {
    const r = await fetch("/api/detect", { method: "POST", body: fd });
    const data = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(data.detail || `Request failed (${r.status})`);
    state.result = data;
    state.baseImg = await loadImage("data:image/png;base64," + data.image_png);
    state.safetyImg = await loadImage("data:image/png;base64," + data.landing.overlay_png);
    if (data.ground_truth) $("#show-gt").disabled = false; else { $("#show-gt").checked = false; $("#show-gt").disabled = true; }
    render();
    let note = `Found ${data.stats.count} craters in ${fmt(data.detect_ms, 0)} ms.`;
    if (data.downscale_factor < 1) note += ` Image was downscaled by ${fmt(1 / data.downscale_factor, 2)}x; resolution adjusted to ${fmt(data.effective_meters_per_pixel, 3)} m/px.`;
    setMsg(note);
  } catch (e) {
    setMsg(e.message, true);
  } finally { btn.disabled = false; btn.textContent = "Detect craters"; }
});

["#show-craters", "#show-gt", "#show-safety", "#show-sites"].forEach((id) => $(id).addEventListener("change", drawCanvas));

function render() {
  const d = state.result;
  $("#empty").hidden = true;
  $("#results").hidden = false;
  ["#dl-csv", "#dl-json", "#dl-png"].forEach((id) => ($(id).disabled = false));
  drawCanvas();
  renderCards(d);
  renderSfd(d.sfd);
  renderSites(d.landing);
  renderTable(d.catalogue);
}

function drawCanvas() {
  const d = state.result; if (!d) return;
  const c = $("#canvas"), ctx = c.getContext("2d");
  const W = d.stats.image_width_px, H = d.stats.image_height_px;
  // draw at least 640 px wide for crisp vector overlays
  const k = Math.max(1, Math.min(4, 768 / Math.max(W, H)));
  c.width = Math.round(W * k); c.height = Math.round(H * k);
  ctx.imageSmoothingEnabled = k < 1.5;
  ctx.drawImage(state.baseImg, 0, 0, c.width, c.height);
  if ($("#show-safety").checked) { ctx.globalAlpha = 0.55; ctx.drawImage(state.safetyImg, 0, 0, c.width, c.height); ctx.globalAlpha = 1; }
  const lw = Math.max(1.2, k * 0.8);
  if ($("#show-gt").checked && d.ground_truth) {
    ctx.strokeStyle = "#4fc3f7"; ctx.lineWidth = lw; ctx.setLineDash([4, 3]);
    d.ground_truth.circles.forEach(([x, y, r]) => { ctx.beginPath(); ctx.arc(x * k, y * k, r * k, 0, 2 * Math.PI); ctx.stroke(); });
    ctx.setLineDash([]);
  }
  if ($("#show-craters").checked) {
    d.catalogue.forEach((row) => {
      const hl = state.hover === row.id;
      ctx.strokeStyle = hl ? "#ffffff" : "#ffb020"; ctx.lineWidth = hl ? lw * 2 : lw;
      ctx.beginPath(); ctx.arc(row.x_px * k, row.y_px * k, row.radius_px * k, 0, 2 * Math.PI); ctx.stroke();
    });
  }
  if ($("#show-sites").checked) {
    const fr = (d.landing.footprint_m / d.effective_meters_per_pixel / 2) * k;
    d.landing.sites.forEach((s) => {
      ctx.strokeStyle = "#5ad17a"; ctx.fillStyle = "rgba(90,209,122,.25)"; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.arc(s.x_px * k, s.y_px * k, Math.max(fr, 6), 0, 2 * Math.PI); ctx.fill(); ctx.stroke();
      ctx.fillStyle = "#0c0d10"; ctx.beginPath(); ctx.arc(s.x_px * k, s.y_px * k, 9, 0, 2 * Math.PI); ctx.fill();
      ctx.fillStyle = "#5ad17a"; ctx.font = "bold 12px system-ui"; ctx.textAlign = "center"; ctx.textBaseline = "middle";
      ctx.fillText(String(s.rank), s.x_px * k, s.y_px * k + 0.5);
    });
  }
  c.dataset.scale = k;
}

$("#canvas").addEventListener("mousemove", (e) => {
  const d = state.result; if (!d) return;
  const c = e.currentTarget, rect = c.getBoundingClientRect();
  const k = parseFloat(c.dataset.scale || 1);
  const x = ((e.clientX - rect.left) * (c.width / rect.width)) / k;
  const y = ((e.clientY - rect.top) * (c.height / rect.height)) / k;
  let best = null, bestD = Infinity;
  d.catalogue.forEach((row) => {
    const dist = Math.hypot(row.x_px - x, row.y_px - y);
    if (dist <= row.radius_px * 1.1 && dist / row.radius_px < bestD) { best = row; bestD = dist / row.radius_px; }
  });
  const tip = $("#tooltip");
  const id = best ? best.id : null;
  if (id !== state.hover) { state.hover = id; drawCanvas(); highlightRow(id); }
  if (best) {
    tip.hidden = false;
    tip.innerHTML = `Crater #${best.id}<br>D = ${fmt(best.diameter_m)} m<br>confidence ${fmt(best.confidence, 2)}`;
    const vr = $("#viewer").getBoundingClientRect();
    tip.style.left = Math.min(e.clientX - vr.left + 14, vr.width - 140) + "px";
    tip.style.top = e.clientY - vr.top + 14 + "px";
  } else tip.hidden = true;
});
$("#canvas").addEventListener("mouseleave", () => { $("#tooltip").hidden = true; state.hover = null; drawCanvas(); highlightRow(null); });

function highlightRow(id) {
  document.querySelectorAll("#table tbody tr").forEach((tr) => tr.classList.toggle("hl", Number(tr.dataset.id) === id));
  if (id !== null) { const tr = document.querySelector(`#table tbody tr[data-id="${id}"]`); if (tr) tr.scrollIntoView({ block: "nearest" }); }
}

function renderCards(d) {
  const s = d.stats, sfd = d.sfd, gt = d.ground_truth;
  const cards = [
    ["Craters detected", fmt(s.count, 0), d.method === "unet" ? "U-Net" : "Hough baseline"],
    ["Density", fmt(s.density_per_km2, 1), "craters per km²"],
    ["Area covered", fmt(s.coverage_percent, 1) + "%", "inside crater rims"],
    ["Median diameter", s.diameter_median_m === null ? "-" : fmt(s.diameter_median_m, 1) + " m", `range ${fmt(s.diameter_min_m, 1)}-${fmt(s.diameter_max_m, 1)} m`],
    ["SFD slope", sfd.slope === null ? "-" : fmt(sfd.slope, 2), sfd.slope === null ? "need >= 5 craters" : `fit on ${sfd.fit_n} craters`],
    ["Image area", fmt(s.area_km2, 4) + " km²", `${s.image_width_px}x${s.image_height_px} px @ ${fmt(s.meters_per_pixel, 3)} m`],
  ];
  if (gt) cards.push(["Vs ground truth", `F1 ${fmt(gt.scores.f1, 2)}`, `P ${fmt(gt.scores.precision, 2)} / R ${fmt(gt.scores.recall, 2)} of ${gt.count}`]);
  $("#cards").innerHTML = cards.map(([k, v, s2]) => `<div class="card"><div class="k">${k}</div><div class="v">${v}</div><div class="s">${s2}</div></div>`).join("");
}

// ------------------------------------------------------------------ charts (plain SVG)
function svgEl(w, h) { return `<svg class="chart" viewBox="0 0 ${w} ${h}" width="100%" preserveAspectRatio="xMidYMid meet">`; }

function renderSfd(sfd) {
  const pts = sfd.points;
  if (pts.length < 2) { $("#sfd").innerHTML = '<p class="hint">Not enough craters for a distribution.</p>'; return; }
  const W = 460, H = 260, m = { l: 52, r: 12, t: 10, b: 36 };
  const xs = pts.map((p) => Math.log10(p.diameter_m)), ys = pts.map((p) => Math.log10(p.n_cum_per_km2));
  const x0 = Math.floor(Math.min(...xs) * 2) / 2, x1 = Math.ceil(Math.max(...xs) * 2) / 2 || x0 + 0.5;
  const y0 = Math.floor(Math.min(...ys)), y1 = Math.ceil(Math.max(...ys)) || y0 + 1;
  const X = (v) => m.l + ((v - x0) / Math.max(1e-9, x1 - x0)) * (W - m.l - m.r);
  const Y = (v) => H - m.b - ((v - y0) / Math.max(1e-9, y1 - y0)) * (H - m.t - m.b);
  let s = svgEl(W, H);
  for (let v = Math.ceil(y0); v <= y1; v++) s += `<line class="axis" x1="${m.l}" x2="${W - m.r}" y1="${Y(v)}" y2="${Y(v)}"/><text x="${m.l - 6}" y="${Y(v) + 4}" text-anchor="end">1e${v}</text>`;
  for (let v = Math.ceil(x0); v <= x1; v++) s += `<line class="axis" y1="${m.t}" y2="${H - m.b}" x1="${X(v)}" x2="${X(v)}"/><text x="${X(v)}" y="${H - m.b + 16}" text-anchor="middle">${fmt(10 ** v, 3)} m</text>`;
  s += `<text x="${(W + m.l) / 2}" y="${H - 4}" text-anchor="middle">crater diameter D</text>`;
  s += `<text transform="translate(12 ${(H - m.b) / 2}) rotate(-90)" text-anchor="middle">N(&ge;D) per km&sup2;</text>`;
  if (sfd.completeness_m > 0) { const cx = X(Math.log10(sfd.completeness_m)); if (cx > m.l) s += `<line x1="${cx}" x2="${cx}" y1="${m.t}" y2="${H - m.b}" stroke="#555" stroke-dasharray="2 3"/>`; }
  pts.forEach((p, i) => (s += `<circle cx="${X(xs[i])}" cy="${Y(ys[i])}" r="2.6" fill="#ffb020"/>`));
  if (sfd.slope !== null) {
    const a = Math.max(Math.log10(sfd.completeness_m), x0), b = Math.max(...xs);
    s += `<line x1="${X(a)}" y1="${Y(sfd.intercept + sfd.slope * a)}" x2="${X(b)}" y2="${Y(sfd.intercept + sfd.slope * b)}" stroke="#e8e6e1" stroke-width="1.5" stroke-dasharray="5 4"/>`;
    s += `<text x="${W - m.r - 4}" y="${m.t + 14}" text-anchor="end" style="fill:#e8e6e1">slope ${fmt(sfd.slope, 2)}</text>`;
  }
  $("#sfd").innerHTML = s + "</svg>";
}

function renderSites(l) {
  if (!l.sites.length) { $("#sites").innerHTML = '<p class="hint">No hazard-free spot found for this footprint.</p>'; return; }
  $("#sites").innerHTML = `<p class="hint">${fmt(l.safe_fraction * 100, 1)}% of the image is at least ${fmt(l.footprint_m / 2, 1)} m from every detected crater.</p><ul class="site-list">` +
    l.sites.map((s) => `<li><span><strong>Site ${s.rank}</strong> at (${s.x_px}, ${s.y_px}) px</span><span>clearance ${fmt(s.clearance_m, 1)} m · smoother than ${Math.min(99.9, 100 - s.roughness_percentile).toFixed(1)}% of image <span class="badge ${s.fits_footprint ? "good" : "bad"}">${s.fits_footprint ? "fits" : "too tight"}</span></span></li>`).join("") + "</ul>";
}

function renderTable(cat) {
  $("#cat-count").textContent = cat.length > 500 ? `(largest 500 of ${cat.length}; full list in CSV)` : `(${cat.length})`;
  $("#table tbody").innerHTML = cat.slice(0, 500).map((r) => `<tr data-id="${r.id}"><td>${r.id}</td><td>${fmt(r.x_px, 1)}</td><td>${fmt(r.y_px, 1)}</td><td>${fmt(r.diameter_m, 2)}</td><td>${fmt(r.confidence, 2)}</td></tr>`).join("");
  document.querySelectorAll("#table tbody tr").forEach((tr) => {
    tr.addEventListener("mouseenter", () => { state.hover = Number(tr.dataset.id); drawCanvas(); });
    tr.addEventListener("mouseleave", () => { state.hover = null; drawCanvas(); });
  });
}

// ------------------------------------------------------------------ downloads
function download(name, blob) { const a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = name; a.click(); setTimeout(() => URL.revokeObjectURL(a.href), 2000); }
const baseName = () => (state.result.source || "image").replace(/\.[^.]+$/, "");
$("#dl-csv").addEventListener("click", () => {
  const cols = ["id", "x_px", "y_px", "radius_px", "diameter_m", "confidence"];
  const csv = [cols.join(",")].concat(state.result.catalogue.map((r) => cols.map((c) => r[c]).join(","))).join("\n") + "\n";
  download(baseName() + "_craters.csv", new Blob([csv], { type: "text/csv" }));
});
$("#dl-json").addEventListener("click", () => {
  const r = { ...state.result, image_png: undefined, landing: { ...state.result.landing, overlay_png: undefined } };
  download(baseName() + "_analysis.json", new Blob([JSON.stringify(r, null, 2)], { type: "application/json" }));
});
$("#dl-png").addEventListener("click", () => $("#canvas").toBlob((b) => download(baseName() + "_annotated.png", b)));

// ------------------------------------------------------------------ model tab
let modelLoaded = false;
async function loadModel() {
  if (modelLoaded) return;
  try {
    const d = await (await fetch("/api/model")).json();
    modelLoaded = true;
    const ev = d.evaluation;
    if (!ev) { $("#eval-desc").textContent = "No evaluation report yet. Run: python -m lunarcrater evaluate"; }
    else {
      const t = ev.test_set;
      $("#eval-desc").textContent = `${t.images} held-out simulated scenes (${t.image_size_px} px, ${t.ground_truth_craters} ground-truth craters with radius >= ${t.min_radius_px} px). Match rule: ${t.match_rule}.`;
      const names = { unet: "U-Net + ring matching", hough: "Hough transform baseline" };
      const rows = Object.entries(ev.methods);
      $("#eval-table").innerHTML = "<thead><tr><th>Method</th><th>Precision</th><th>Recall</th><th>F1</th><th>TP / FP / FN</th><th>Centre error</th><th>Radius error</th><th>Time / image</th></tr></thead><tbody>" +
        rows.map(([k, m]) => `<tr><td>${names[k] || k}</td><td>${fmt(m.precision, 3)}</td><td>${fmt(m.recall, 3)}</td><td><strong>${fmt(m.f1, 3)}</strong></td><td>${m.tp} / ${m.fp} / ${m.fn}</td><td>${m.mean_frac_err_xy === null ? "-" : fmt(m.mean_frac_err_xy * 100, 1) + "% of r"}</td><td>${m.mean_frac_err_r === null ? "-" : fmt(m.mean_frac_err_r * 100, 1) + "%"}</td><td>${fmt(m.mean_inference_ms, 0)} ms</td></tr>`).join("") + "</tbody>";
      renderRecall(rows);
    }
    renderTraining(d.training_history);
  } catch { $("#eval-desc").textContent = "Could not load model information."; }
}

function renderRecall(rows) {
  const bins = Object.keys(rows[0][1].recall_by_size);
  const W = 460, H = 240, m = { l: 40, r: 10, t: 24, b: 40 };
  const colors = { unet: "#ffb020", hough: "#4fc3f7" };
  const gw = (W - m.l - m.r) / bins.length, bw = (gw - 24) / rows.length;
  const Y = (v) => H - m.b - v * (H - m.t - m.b);
  let s = svgEl(W, H);
  [0, 0.25, 0.5, 0.75, 1].forEach((v) => (s += `<line class="axis" x1="${m.l}" x2="${W - m.r}" y1="${Y(v)}" y2="${Y(v)}"/><text x="${m.l - 6}" y="${Y(v) + 4}" text-anchor="end">${v * 100}%</text>`));
  bins.forEach((b, i) => {
    rows.forEach(([k, mm], j) => {
      const v = mm.recall_by_size[b] || 0, x = m.l + i * gw + 12 + j * bw;
      s += `<rect x="${x}" y="${Y(v)}" width="${bw - 4}" height="${Y(0) - Y(v)}" fill="${colors[k] || "#999"}" rx="2"/><text x="${x + (bw - 4) / 2}" y="${Y(v) - 4}" text-anchor="middle">${Math.round(v * 100)}</text>`;
    });
    s += `<text x="${m.l + i * gw + gw / 2}" y="${H - m.b + 16}" text-anchor="middle">${b}</text>`;
  });
  rows.forEach(([k], j) => (s += `<rect x="${m.l + j * 120}" y="4" width="10" height="10" fill="${colors[k]}"/><text x="${m.l + 14 + j * 120}" y="13">${k === "unet" ? "U-Net" : "Hough"}</text>`));
  $("#recall-chart").innerHTML = s + "</svg>";
}

function renderTraining(hist) {
  if (!hist || !hist.length) { $("#train-chart").innerHTML = '<p class="hint">No training history found.</p>'; return; }
  const W = 460, H = 240, m = { l: 44, r: 44, t: 24, b: 36 };
  const n = hist.length, maxL = Math.max(...hist.map((h) => Math.max(h.train_loss, h.val_loss)));
  const X = (i) => m.l + (n === 1 ? 0 : (i / (n - 1)) * (W - m.l - m.r));
  const Y = (v) => H - m.b - (v / maxL) * (H - m.t - m.b);
  const Yi = (v) => H - m.b - v * (H - m.t - m.b);
  const line = (f, col, dash = "") => `<polyline fill="none" stroke="${col}" stroke-width="2" ${dash} points="${hist.map((h, i) => `${X(i)},${f(h)}`).join(" ")}"/>`;
  let s = svgEl(W, H);
  [0, 0.5, 1].forEach((v) => (s += `<line class="axis" x1="${m.l}" x2="${W - m.r}" y1="${Yi(v)}" y2="${Yi(v)}"/><text x="${m.l - 6}" y="${Yi(v) + 4}" text-anchor="end">${fmt(v * maxL, 2)}</text><text x="${W - m.r + 6}" y="${Yi(v) + 4}">${v.toFixed(1)}</text>`));
  hist.forEach((h, i) => { if (n <= 20 || i % 2 === 0) s += `<text x="${X(i)}" y="${H - m.b + 16}" text-anchor="middle">${h.epoch}</text>`; });
  s += line((h) => Y(h.train_loss), "#9a9890") + line((h) => Y(h.val_loss), "#ffb020") + line((h) => Yi(h.val_rim_iou), "#5ad17a", 'stroke-dasharray="5 4"');
  s += `<text x="${m.l}" y="13" style="fill:#9a9890">train loss</text><text x="${m.l + 80}" y="13" style="fill:#ffb020">val loss</text><text x="${m.l + 150}" y="13" style="fill:#5ad17a">val rim IoU (right axis)</text>`;
  s += `<text x="${(W + m.l - m.r) / 2}" y="${H - 4}" text-anchor="middle">epoch</text>`;
  $("#train-chart").innerHTML = s + "</svg>";
}

init();
