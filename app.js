const $ = s => document.querySelector(s), $$ = s => [...document.querySelectorAll(s)];
const S = 1024, CONF = 0.3;
const MODEL_BYTES = { diagnosis: 19166356, fdi: 19188438 }; // fallback when no content-length
let sessions = {}, cases = [], lastFindings = [], lastImg = null, lang = "en", sel = -1, sampleIdx = 0, lastRun = null;
const esc = s => String(s).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const hexA = (h, a) => { const n = parseInt(h.slice(1), 16); return `rgba(${n >> 16},${(n >> 8) & 255},${n & 255},${a})`; };
function toast(t) { const el = $("#toast"); el.textContent = t; el.classList.add("on"); clearTimeout(toast.t); toast.t = setTimeout(() => el.classList.remove("on"), 1800); }

// ---------- service worker (offline after first load) ----------
if ("serviceWorker" in navigator && location.protocol !== "file:") addEventListener("load", () => navigator.serviceWorker.register("sw.js").catch(() => {}));

// ---------- views (hash routing) ----------
function show(v) {
  if (!["home", "look", "learn"].includes(v)) v = "home";
  $$(".view").forEach(s => s.classList.toggle("on", s.id === v));
  $$(".dock a").forEach(a => a.classList.toggle("on", a.dataset.v === v));
  if (v === "learn" && !learn.c) loadLearn(dailyIndex());
  scrollTo({ top: 0, behavior: "instant" });
}
addEventListener("hashchange", () => show(location.hash.slice(1)));

// ---------- home hero ----------
async function hero() {
  await loadCases(); const c = cases[0];
  $("#heroBoxes").innerHTML = c.findings.map(f => { const [x, y, w, h] = f.box, col = FINDINGS[f.finding].color, d = (x / c.w) * 2.7;
    return `<i style="left:${100 * x / c.w}%;top:${100 * y / c.h}%;width:${100 * w / c.w}%;height:${100 * h / c.h}%;color:${col};animation-delay:${d.toFixed(2)}s" data-l="${f.fdi} ${f.finding}"></i>`; }).join("");
}

// ---------- models ----------
if (self.ort) { ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";
  ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 2) : 1; }
const dl = { diagnosis: 0, fdi: 0 };
function setProgress() {
  const got = dl.diagnosis + dl.fdi, tot = MODEL_BYTES.diagnosis + MODEL_BYTES.fdi, pct = Math.min(100, Math.round(100 * got / tot));
  $("#pPct").textContent = pct + "%"; $("#pBar").style.width = pct + "%";
  $("#pSub").textContent = `${(got / 1048576).toFixed(1)} of ${(tot / 1048576).toFixed(1)} MB · one-time download, cached for offline use`;
}
async function fetchModel(name) {
  const r = await fetch(`models/${name}.onnx`);
  if (!r.ok) throw new Error(`model download failed (HTTP ${r.status})`);
  const len = +r.headers.get("content-length"); if (len > 1e6) MODEL_BYTES[name] = len;
  if (!r.body || !r.body.getReader) { const b = new Uint8Array(await r.arrayBuffer()); dl[name] = b.length; setProgress(); return b; }
  const rd = r.body.getReader(), parts = []; let got = 0;
  for (;;) { const { done, value } = await rd.read(); if (done) break; parts.push(value); got += value.length; dl[name] = got; setProgress(); }
  const buf = new Uint8Array(got); let o = 0; for (const p of parts) { buf.set(p, o); o += p.length; } return buf;
}
function model(name) {
  if (!self.ort) return Promise.reject(new Error("the AI engine (onnxruntime-web) did not load"));
  if (!sessions[name]) sessions[name] = fetchModel(name).then(b => ort.InferenceSession.create(b, { executionProviders: ["wasm"] }))
    .catch(e => { delete sessions[name]; dl[name] = 0; throw e; });
  return sessions[name];
}
function letterbox(img) {
  const r = S / Math.max(img.width, img.height), nw = Math.round(img.width * r), nh = Math.round(img.height * r);
  const px = Math.floor((S - nw) / 2), py = Math.floor((S - nh) / 2);
  const c = document.createElement("canvas"); c.width = c.height = S; const g = c.getContext("2d", { willReadFrequently: true });
  g.fillStyle = "rgb(114,114,114)"; g.fillRect(0, 0, S, S); g.drawImage(img, px, py, nw, nh);
  const d = g.getImageData(0, 0, S, S).data, f = new Float32Array(3 * S * S);
  for (let i = 0; i < S * S; i++) { f[i] = d[i * 4] / 255; f[i + S * S] = d[i * 4 + 1] / 255; f[i + 2 * S * S] = d[i * 4 + 2] / 255; }
  return { t: new ort.Tensor("float32", f, [1, 3, S, S]), r, px, py };
}
const iou = (a, b) => { const x1 = Math.max(a[0], b[0]), y1 = Math.max(a[1], b[1]), x2 = Math.min(a[2], b[2]), y2 = Math.min(a[3], b[3]);
  const i = Math.max(0, x2 - x1) * Math.max(0, y2 - y1); return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-9); };
async function detect(name, lb) {
  const s = await model(name), out = (await s.run({ images: lb.t })).output0, [, ch, n] = out.dims, o = out.data, nc = ch - 4, cand = [];
  for (let i = 0; i < n; i++) {
    let best = 0, cl = 0; for (let c = 0; c < nc; c++) { const v = o[(4 + c) * n + i]; if (v > best) { best = v; cl = c; } }
    if (best < CONF) continue;
    const cx = o[i], cy = o[n + i], w = o[2 * n + i], h = o[3 * n + i];
    cand.push({ box: [(cx - w / 2 - lb.px) / lb.r, (cy - h / 2 - lb.py) / lb.r, (cx + w / 2 - lb.px) / lb.r, (cy + h / 2 - lb.py) / lb.r], cl, p: best });
  }
  cand.sort((a, b) => b.p - a.p); const keep = [];
  for (const c of cand) if (keep.every(k => iou(k.box, c.box) < 0.5)) keep.push(c);
  return keep;
}
const tick = () => new Promise(r => requestAnimationFrame(() => setTimeout(r, 0)));
async function analyse(img) {
  const P = $("#progress"); P.hidden = false; $("#pBar").parentElement.classList.remove("indet");
  $("#pLabel").textContent = "Downloading AI models"; setProgress();
  const lb = letterbox(img);
  await Promise.all([model("diagnosis"), model("fdi")]);
  $("#pLabel").textContent = "Reading the OPG on your device"; $("#pPct").textContent = ""; $("#pSub").textContent = "Finding caries, deep caries, periapical lesions and impacted teeth, then numbering the teeth.";
  $("#pBar").parentElement.classList.add("indet"); $("#lookBeam").hidden = false; await tick();
  const t0 = performance.now();
  const d = await detect("diagnosis", lb), t = await detect("fdi", lb);
  const res = d.map(f => { let bt = null, bi = 0; for (const k of t) { const v = iou(f.box, k.box); if (v > bi) { bi = v; bt = k; } }
    return { finding: DIS[f.cl], fdi: bt && bi > 0.2 ? FDI[bt.cl] : "?", p: f.p, box: f.box }; });
  P.hidden = true; $("#lookBeam").hidden = true;
  const st = $("#status"); st.hidden = false;
  st.innerHTML = `<i class="okdot"></i>Done in ${((performance.now() - t0) / 1000).toFixed(1)} s, on your device. ${res.length} finding${res.length === 1 ? "" : "s"} to double-check.`;
  return res;
}

// ---------- drawing ----------
function rr(g, x, y, w, h, r) { g.beginPath(); g.moveTo(x + r, y); g.arcTo(x + w, y, x + w, y + h, r); g.arcTo(x + w, y + h, x, y + h, r); g.arcTo(x, y + h, x, y, r); g.arcTo(x, y, x + w, y, r); g.closePath(); }
function draw(cv, img, items, opts = {}) {
  cv.width = img.width; cv.height = img.height; const g = cv.getContext("2d"); g.drawImage(img, 0, 0);
  const U = img.width / 400, lw = Math.max(2.5, U * 1.25), fs = Math.round(img.width / 40), hasSel = opts.sel >= 0;
  g.font = `700 ${fs}px "JetBrains Mono", ui-monospace, monospace`; g.textBaseline = "middle";
  if (hasSel) { g.fillStyle = "rgba(2,6,14,.45)"; g.fillRect(0, 0, img.width, img.height); }
  items.forEach((it, i) => {
    const c = it.color || FINDINGS[it.finding].color, [x1, y1, x2, y2] = it.box, on = !hasSel || opts.sel === i, w = x2 - x1, h = y2 - y1;
    g.save(); g.globalAlpha = on ? 1 : 0.35;
    if (hasSel && on) { g.save(); rr(g, x1, y1, w, h, U * 2); g.clip(); g.drawImage(img, 0, 0); g.restore(); g.fillStyle = hexA(c, .12); rr(g, x1, y1, w, h, U * 2); g.fill(); }
    g.shadowColor = c; g.shadowBlur = on ? U * 6 : 0; g.strokeStyle = c; g.lineWidth = on && hasSel ? lw * 1.8 : lw;
    g.setLineDash(it.dash ? [lw * 3, lw * 2] : []); rr(g, x1, y1, w, h, U * 2); g.stroke(); g.shadowBlur = 0; g.setLineDash([]);
    const lab = it.label ?? `${i + 1}${it.fdi && it.fdi !== "?" ? " · " + it.fdi : ""}`;
    const tw = g.measureText(lab).width + fs * 0.8, th = fs * 1.5, ly = y1 - th - U < 0 ? y2 + U : y1 - th - U;
    g.fillStyle = c; rr(g, x1, ly, tw, th, th * 0.3); g.fill(); g.fillStyle = "#03101f"; g.fillText(lab, x1 + fs * 0.4, ly + th / 2 + 1);
    g.restore();
  });
  (opts.dots || []).forEach((d, i) => { const r = img.width / 75; g.setLineDash([]);
    g.fillStyle = "rgba(255,255,255,.18)"; g.beginPath(); g.arc(d.x, d.y, r * 1.8, 0, 7); g.fill();
    g.fillStyle = "#fff"; g.strokeStyle = "#03101f"; g.lineWidth = U * 0.8; g.beginPath(); g.arc(d.x, d.y, r, 0, 7); g.fill(); g.stroke();
    g.fillStyle = "#03101f"; g.textAlign = "center"; g.fillText(String(i + 1), d.x, d.y + 1); g.textAlign = "left"; });
}
const loadImg = src => new Promise((ok, no) => { const i = new Image(); i.onload = () => ok(i); i.onerror = () => no(new Error("That file could not be opened as an image. Try a JPG or PNG of the OPG.")); i.src = src; });

// ---------- zoom / pan / tap ----------
function zoomable(wrap, onTap, dblZoom) {
  const inner = wrap.querySelector(".zin"); let s = 1, tx = 0, ty = 0, pts = new Map(), st = null, pin = null, moved = false, multi = false, lastTap = 0;
  const apply = () => { const W = wrap.clientWidth, H = wrap.clientHeight; tx = Math.min(0, Math.max(W * (1 - s), tx)); ty = Math.min(0, Math.max(H * (1 - s), ty));
    inner.style.transform = `translate(${tx}px,${ty}px) scale(${s})`; wrap.classList.toggle("zoomed", s > 1.01); };
  const zoomAt = (ns, cx, cy) => { const r = wrap.getBoundingClientRect(), px = cx - r.left, py = cy - r.top; ns = Math.min(5, Math.max(1, ns));
    tx = px - (px - tx) * ns / s; ty = py - (py - ty) * ns / s; s = ns; apply(); };
  wrap.addEventListener("pointerdown", e => { pts.set(e.pointerId, e); if (pts.size === 1) { st = { x: e.clientX, y: e.clientY, tx, ty, t: performance.now() }; moved = false; multi = false; }
    if (pts.size === 2) { const [a, b] = [...pts.values()]; pin = { d: Math.hypot(a.clientX - b.clientX, a.clientY - b.clientY), s }; multi = true; } });
  wrap.addEventListener("pointermove", e => { if (!pts.has(e.pointerId)) return; pts.set(e.pointerId, e);
    if (pts.size === 2 && pin) { const [a, b] = [...pts.values()]; inner.classList.add("drag"); zoomAt(pin.s * Math.hypot(a.clientX - b.clientX, a.clientY - b.clientY) / pin.d, (a.clientX + b.clientX) / 2, (a.clientY + b.clientY) / 2); moved = true; }
    else if (st) { const dx = e.clientX - st.x, dy = e.clientY - st.y; if (Math.hypot(dx, dy) > 8) moved = true; if (moved && s > 1.01) { inner.classList.add("drag"); tx = st.tx + dx; ty = st.ty + dy; apply(); } } });
  const up = e => { const had = pts.has(e.pointerId); pts.delete(e.pointerId); if (pts.size < 2) pin = null; inner.classList.remove("drag");
    if (!had || pts.size || e.type === "pointercancel") return; const now = performance.now();
    if (!moved && !multi && st && now - st.t < 450) {
      if (dblZoom && now - lastTap < 320) { zoomAt(s > 1.4 ? 1 : 2.5, e.clientX, e.clientY); lastTap = 0; return; }
      lastTap = now; onTap(e.clientX, e.clientY);
    } st = null; };
  wrap.addEventListener("pointerup", up); wrap.addEventListener("pointercancel", up);
  wrap.addEventListener("wheel", e => { if (!e.ctrlKey && !e.metaKey && s <= 1.01) return; e.preventDefault(); zoomAt(s * (e.deltaY < 0 ? 1.15 : 1 / 1.15), e.clientX, e.clientY); }, { passive: false });
  const ctl = wrap.parentElement.querySelector(".zctl");
  ctl && ctl.addEventListener("click", e => { const b = e.target.closest("button"); if (!b) return; const r = wrap.getBoundingClientRect(), cx = r.left + r.width / 2, cy = r.top + r.height / 2;
    if (b.dataset.z === "in") zoomAt(s * 1.6, cx, cy); else if (b.dataset.z === "out") zoomAt(s / 1.6, cx, cy); else { s = 1; tx = ty = 0; apply(); } });
  return { reset() { s = 1; tx = ty = 0; apply(); } };
}
function canvasPoint(cv, img, cx, cy) { const r = cv.getBoundingClientRect(), k = img.width / r.width; return { x: (cx - r.left) * k, y: (cy - r.top) * k }; }

// ---------- second look ----------
const lookZ = zoomable($("#lookZoom"), (cx, cy) => {
  if (!lastImg || !lastFindings.length) return; const p = canvasPoint($("#cv"), lastImg, cx, cy);
  let hit = -1, area = 1e18; lastFindings.forEach((f, i) => { const [x1, y1, x2, y2] = f.box, a = (x2 - x1) * (y2 - y1); if (p.x >= x1 && p.x <= x2 && p.y >= y1 && p.y <= y2 && a < area) { area = a; hit = i; } });
  select(hit === sel ? -1 : hit, true);
}, true);
function select(i, fromCanvas) {
  sel = i; draw($("#cv"), lastImg, lastFindings, { sel });
  $$("#list .item[data-i]").forEach(el => { const k = +el.dataset.i; el.classList.toggle("sel", k === i); el.classList.toggle("dim", i >= 0 && k !== i); });
  if (fromCanvas && i >= 0) $(`#list .item[data-i="${i}"]`).scrollIntoView({ behavior: "smooth", block: "nearest" });
  if (!fromCanvas && i >= 0) $("#stageWrap").scrollIntoView({ behavior: "smooth", block: "nearest" });
}
function fail(msg) { $("#progress").hidden = true; $("#lookBeam").hidden = true; $("#errBox").hidden = false; $("#errMsg").textContent = msg; }
async function runLook(src) {
  lastRun = src; $("#errBox").hidden = true; $("#empty").hidden = true; $("#status").hidden = true; $("#findWrap").hidden = true; $("#reportBox").hidden = true;
  $("#sampleBtn").disabled = true; sel = -1;
  try {
    const img = await loadImg(src); lastImg = img; lastFindings = []; $("#stageWrap").hidden = false; lookZ.reset(); draw($("#cv"), img, []);
    try { lastFindings = await analyse(img); } catch (e) { fail("Could not run the AI on this device: " + (e.message || e) + ". Check your connection for the first download, then try again."); return; }
    reveal(img);
  } catch (e) { fail(e.message); } finally { $("#sampleBtn").disabled = false; }
}
function reveal(img) {
  // boxes pop in one by one
  lastFindings.forEach((_, k) => setTimeout(() => { if (lastImg === img) draw($("#cv"), img, lastFindings.slice(0, k + 1)); }, 120 * k));
  $("#fCount").textContent = `${lastFindings.length} flagged`; $("#findWrap").hidden = false;
  $("#list").innerHTML = lastFindings.length ? lastFindings.map((f, i) => { const k = FINDINGS[f.finding], pc = Math.round(f.p * 100);
    return `<div class="item" data-i="${i}" style="--c:${k.color};animation-delay:${i * 70}ms"><span class="num">${i + 1}</span>
      <div class="i-main"><b>${f.finding}</b><small>${f.fdi === "?" ? "Tooth number unclear" : "Tooth " + f.fdi + " · " + toothName(f.fdi)}</small></div>
      <div class="conf">${pc}%<span class="cb"><i style="width:${pc}%"></i></span></div><button class="why" data-x="${i}" aria-label="Explain ${f.finding}">i</button></div>`; }).join("")
    : `<div class="item none">No caries, deep caries, periapical lesion or impaction flagged. That does not rule them out, so still read the whole OPG.</div>`;
  $$("#list .item[data-i]").forEach(el => el.onclick = e => { const i = +el.dataset.i; if (e.target.closest(".why")) { select(i); explain(lastFindings[i]); return; } select(sel === i ? -1 : i); });
  $("#reportBox").hidden = false; renderReport();
}
$("#file").onchange = e => { const f = e.target.files[0]; if (!f) return; runLook(URL.createObjectURL(f)); e.target.value = ""; };
$("#sampleBtn").onclick = async () => { try { await loadCases(); } catch { return fail("Could not load the sample cases. Check your connection and try again."); }
  const c = cases[sampleIdx++ % cases.length]; runLook(c.img); };
$("#retryBtn").onclick = () => lastRun ? runLook(lastRun) : $("#sampleBtn").click();
document.addEventListener("dragover", e => { if ($("#look").classList.contains("on")) e.preventDefault(); });
document.addEventListener("drop", e => { if (!$("#look").classList.contains("on")) return; e.preventDefault(); const f = e.dataTransfer.files[0]; if (f) runLook(URL.createObjectURL(f)); });

function reportText(l) {
  const today = new Date().toLocaleDateString(l === "hi" ? "hi-IN" : "en-IN");
  if (l === "hi") {
    const lines = lastFindings.map((f, i) => `${i + 1}. ${f.fdi !== "?" ? "दाँत " + f.fdi + " (" + QUAD_HI[f.fdi[0]] + ")" : "दाँत"}: ${FINDINGS[f.finding].patientHi}`);
    return `OPG सेकंड-लुक रिपोर्ट (${today})\n\n${lines.length ? "AI ने इन जगहों पर ध्यान देने को कहा है:\n" + lines.join("\n") : "AI को कोई खास बदलाव नहीं दिखा।"}\n\nयह निदान (diagnosis) नहीं है। कृपया अपने डेंटिस्ट से जाँच करवाएँ।`;
  }
  const lines = lastFindings.map((f, i) => `${i + 1}. ${f.fdi !== "?" ? "Tooth " + f.fdi + " (" + toothName(f.fdi) + ")" : "Unnumbered tooth"}: ${FINDINGS[f.finding].patient}. Confidence ${Math.round(f.p * 100)}%.`);
  return `OPG second-look report (${today})\n\n${lines.length ? "Areas the AI suggests checking:\n" + lines.join("\n") : "The AI did not flag caries, deep caries, periapical lesions or impacted teeth."}\n\nThis is a second-look aid, not a diagnosis. Please confirm with a clinical exam and your dentist.`;
}
function renderReport() { const p = $("#report"); p.textContent = reportText(lang); p.classList.toggle("hi", lang === "hi"); p.lang = lang; }
$$(".seg button").forEach(b => b.onclick = () => { lang = b.dataset.l; $$(".seg button").forEach(x => x.classList.toggle("on", x === b)); $(".seg").classList.toggle("hi", lang === "hi"); renderReport(); });
const copy = async t => { try { await navigator.clipboard.writeText(t); toast("Report copied"); } catch { toast("Copy is blocked here. Select the text instead."); } };
$("#copyBtn").onclick = () => copy($("#report").textContent);
$("#shareBtn").onclick = async () => { const text = $("#report").textContent; if (navigator.share) { try { await navigator.share({ title: "OPG second-look report", text }); } catch {} } else copy(text); };

// ---------- explanations + BYOK tutor ----------
let exCtx = null;
function explain(f) {
  const k = FINDINGS[f.finding]; exCtx = f;
  $("#exBody").innerHTML = `<div class="ex-head" style="--c:${k.color}"><span class="ex-badge">${f.fdi && f.fdi !== "?" ? f.fdi : "?"}</span><div><h3>${f.finding}</h3><span class="hi" lang="hi">${k.hi}</span></div></div>
  <p class="muted">${f.fdi && f.fdi !== "?" ? "Tooth " + f.fdi + ", " + toothName(f.fdi) : "Tooth number unclear"}${f.p ? " · model confidence " + Math.round(f.p * 100) + "%" : " · expert label"}</p>
  <div style="--c:${k.color}"><div class="ex-sec"><b>What it looks like</b><p>${k.looks}</p></div><div class="ex-sec"><b>Why</b><p>${k.why}</p></div>
  <div class="ex-sec"><b>Don't confuse with</b><p>${k.mimics}</p></div><div class="ex-sec"><b>Next step</b><p>${k.next}</p></div></div>`;
  $("#tutor").hidden = !byok().key; $("#tAns").textContent = ""; $("#exDlg").showModal();
}
const byok = () => { try { return JSON.parse(localStorage.getItem("byok") || "{}"); } catch { return {}; } };
$("#setBtn").onclick = () => { const b = byok(); $("#sUrl").value = b.url || ""; $("#sKey").value = b.key || ""; $("#sModel").value = b.model || ""; $("#sVision").checked = !!b.vision; $("#setDlg").showModal(); };
$("#sSave").onclick = () => { localStorage.setItem("byok", JSON.stringify({ url: $("#sUrl").value.trim().replace(/\/$/, ""), key: $("#sKey").value.trim(), model: $("#sModel").value.trim(), vision: $("#sVision").checked })); toast("Saved in this browser only"); };
async function askLLM(question, withImage) {
  const b = byok(); const content = [{ type: "text", text: question }];
  if (withImage && b.vision && lastImg) { const c = document.createElement("canvas"), r = 1280 / lastImg.width; c.width = 1280; c.height = lastImg.height * r; c.getContext("2d").drawImage(lastImg, 0, 0, c.width, c.height); content.push({ type: "image_url", image_url: { url: c.toDataURL("image/jpeg", 0.8) } }); }
  const res = await fetch(`${b.url || "https://api.openai.com/v1"}/chat/completions`, { method: "POST", headers: { "Content-Type": "application/json", Authorization: `Bearer ${b.key}` },
    body: JSON.stringify({ model: b.model || "gpt-4o-mini", messages: [{ role: "system", content: "You are an oral radiology tutor for dental students in India. Be accurate, concise, and say when an OPG is not enough. Never give a definitive diagnosis." }, { role: "user", content }] }) });
  const j = await res.json(); if (!res.ok) throw new Error(j.error?.message || res.status); return j.choices[0].message.content;
}
$("#tAsk").onclick = async () => { $("#tAns").textContent = "Thinking…";
  try { $("#tAns").textContent = await askLLM(`Finding: ${exCtx.finding} on tooth ${exCtx.fdi} seen on a panoramic radiograph. Student question: ${$("#tq").value || "Explain this finding and how to confirm it."}`, true); }
  catch (e) { $("#tAns").textContent = "Your AI endpoint returned an error: " + e.message; } };

// ---------- learn mode ----------
const learn = { c: null, img: null, picks: [], revealed: false };
async function loadCases() { if (!cases.length) { const r = await fetch("samples/cases.json"); if (!r.ok) throw new Error("cases"); cases = await r.json(); } }
const dayKey = () => new Date().toLocaleDateString("en-CA");
const dailyIndex = () => Math.floor(Date.now() / 864e5);
const getStats = () => { try { return JSON.parse(localStorage.getItem("learn") || "{}"); } catch { return {}; } };
async function loadLearn(idx) {
  try { await loadCases(); } catch { $("#lresult").innerHTML = `<div class="error"><b>Could not load today's case</b><p>Check your connection and try again.</p></div>`; return; }
  learn.c = cases[idx % cases.length]; learn.img = await loadImg(learn.c.img); learn.picks = []; learn.revealed = false;
  learn.daily = idx === dailyIndex(); $("#dayLbl").textContent = learn.daily ? "OPG of the Day" : "Practice case";
  $("#caseHint").textContent = "Tap the X-ray where you see a finding, then name it."; $("#revealBtn").disabled = false;
  $("#lresult").innerHTML = ""; renderPicks(); learnZ.reset(); draw($("#lcv"), learn.img, []); stats();
}
function stats() {
  const s = getStats(), y = new Date(Date.now() - 864e5).toLocaleDateString("en-CA"), alive = s.last === dayKey() || s.last === y, st = alive ? s.streak || 0 : 0, done = s.last === dayKey();
  $("#streak").textContent = st; $("#homeStreak").textContent = st; $("#best").textContent = s.best || 0;
  $(".stat.flame").classList.toggle("cold", !st); $("#bestRing").style.strokeDashoffset = 94.25 * (1 - (s.best || 0) / 100);
  $("#todayIco").classList.toggle("done", done); $("#todayIco").textContent = done ? "✓" : "○"; $("#todayLbl").textContent = done ? "Done" : "To do";
}
let pending = null, pickF = null, pickFdi = "";
$("#fOpts").innerHTML = DIS.map(d => `<button type="button" data-f="${d}" style="--c:${FINDINGS[d].color}"><i></i>${d}</button>`).join("");
$("#fOpts").onclick = e => { const b = e.target.closest("button"); if (!b) return; pickF = b.dataset.f; $$("#fOpts button").forEach(x => x.classList.toggle("on", x === b)); };
const row = (a) => a.map(n => `<button type="button" data-n="${n}">${n}</button>`).join("");
$("#fdiGrid").innerHTML = `<span class="lbl">upper right</span><span class="lbl">upper left</span>${row([18, 17, 16, 15, 14, 13, 12, 11])}${row([21, 22, 23, 24, 25, 26, 27, 28])}${row([48, 47, 46, 45, 44, 43, 42, 41])}${row([31, 32, 33, 34, 35, 36, 37, 38])}<span class="lbl">lower right</span><span class="lbl">lower left</span>`;
$("#fdiGrid").onclick = e => { const b = e.target.closest("button"); if (!b) return; pickFdi = pickFdi === b.dataset.n ? "" : b.dataset.n; $("#fdiIn").value = pickFdi; $("#fdiShow").textContent = pickFdi || "–"; $$("#fdiGrid button").forEach(x => x.classList.toggle("on", x.dataset.n === pickFdi)); };
const learnZ = zoomable($("#learnZoom"), (cx, cy) => {
  if (learn.revealed || !learn.img) return; pending = canvasPoint($("#lcv"), learn.img, cx, cy); pickF = null; pickFdi = ""; $("#fdiIn").value = ""; $("#fdiShow").textContent = "–";
  $$("#fOpts button,#fdiGrid button").forEach(x => x.classList.remove("on")); $("#pickDlg").showModal();
}, false);
$("#pickOk").onclick = e => { if (!pickF) { e.preventDefault(); const c = $("#fOpts"); c.classList.remove("shake"); void c.offsetWidth; c.classList.add("shake"); return; }
  learn.picks.push({ ...pending, finding: pickF, fdi: $("#fdiIn").value.trim() }); renderPicks(); draw($("#lcv"), learn.img, [], { dots: learn.picks }); };
function renderPicks() { $("#picks").innerHTML = learn.picks.map((p, i) => `<div class="pick" style="--c:${FINDINGS[p.finding].color}"><span class="num">${i + 1}</span><span><b style="color:var(--c)">●</b> ${p.finding}${p.fdi ? " · " + p.fdi : ""}</span><button class="x" data-del="${i}" aria-label="Remove mark">✕</button></div>`).join("");
  $$("[data-del]").forEach(b => b.onclick = () => { learn.picks.splice(+b.dataset.del, 1); renderPicks(); draw($("#lcv"), learn.img, [], { dots: learn.picks }); }); }
$("#revealBtn").onclick = () => {
  if (learn.revealed || !learn.c) return; learn.revealed = true; $("#revealBtn").disabled = true; const gt = learn.c.findings, used = new Set(); let score = 0; const rows = [];
  gt.forEach((g, gi) => { const [x, y, w, h] = g.box, m = w * 0.15; const hit = learn.picks.findIndex((p, pi) => !used.has(pi) && p.x > x - m && p.x < x + w + m && p.y > y - m && p.y < y + h + m);
    let pts = 0, note; if (hit < 0) note = `<span class="bad">Missed</span>`;
    else { used.add(hit); const p = learn.picks[hit]; pts = p.finding === g.finding ? 1 : (p.finding.includes("Caries") && g.finding.includes("Caries") ? 0.5 : 0.25); if (p.fdi === g.fdi) pts += 0.25;
      note = p.finding === g.finding ? `<span class="ok">Correct</span>` : `<span class="warn">Right tooth, you said ${p.finding}</span>`; if (p.fdi && p.fdi !== g.fdi) note += `<br><span class="warn">FDI ${p.fdi} ≠ ${g.fdi}</span>`; else if (p.fdi === g.fdi) note += `<br><span class="ok">FDI ✓</span>`; }
    score += pts; rows.push(`<div class="item res-row" data-g="${gi}" style="--c:${FINDINGS[g.finding].color};animation-delay:${300 + gi * 90}ms"><span class="num">${g.fdi}</span><div class="i-main"><b>${g.finding}</b><small>${toothName(g.fdi)}</small></div><div class="verdict">${note}</div><button class="why" aria-label="Explain">i</button></div>`); });
  const extra = learn.picks.length - used.size, max = gt.length * 1.25, pct = Math.round(100 * Math.max(0, score - extra * 0.25) / max);
  draw($("#lcv"), learn.img, gt.map(g => ({ ...g, box: [g.box[0], g.box[1], g.box[0] + g.box[2], g.box[1] + g.box[3]], label: g.fdi })), { dots: learn.picks });
  const msg = pct >= 90 ? "Examiner-level read!" : pct >= 70 ? "Sharp eyes. Great read." : pct >= 40 ? "Good start. Check what you missed." : "Every expert started here. Study the answers.";
  $("#lresult").innerHTML = `<div class="result"><div class="score"><div class="ring"><svg viewBox="0 0 100 100"><defs><linearGradient id="rg" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#7ce7ff"/><stop offset="1" stop-color="#8a9bff"/></linearGradient></defs><circle cx="50" cy="50" r="44" class="trk"/><circle cx="50" cy="50" r="44" class="val" id="scoreRing"/></svg><b><span id="scoreNum">0</span><small>%</small></b></div>
    <div><h3>${msg}</h3><p>${used.size} of ${gt.length} expert finding${gt.length > 1 ? "s" : ""} found${extra > 0 ? ` · ${extra} extra mark${extra > 1 ? "s" : ""} (−0.25 each)` : ""}</p></div></div>
    <div class="list-head"><h3>Expert answer key</h3><span class="count">DENTEX</span></div>${rows.join("")}
    <p class="tiny">Answer key: expert labels from DENTEX. They mark only these 4 finding types, so other pathology isn't scored. Tap a finding to learn it.</p></div>`;
  requestAnimationFrame(() => requestAnimationFrame(() => { $("#scoreRing").style.strokeDashoffset = 276.46 * (1 - pct / 100); }));
  const t0 = performance.now(); (function cnt() { const k = Math.min(1, (performance.now() - t0) / 1300), e = 1 - Math.pow(1 - k, 3); $("#scoreNum").textContent = Math.round(pct * e); if (k < 1) requestAnimationFrame(cnt); })();
  $$("[data-g]").forEach(el => el.onclick = () => explain(learn.c.findings[+el.dataset.g]));
  if (pct >= 70) setTimeout(confetti, 500);
  if (learn.daily) { const s = getStats(); if (s.last !== dayKey()) { const y = new Date(Date.now() - 864e5).toLocaleDateString("en-CA"); s.streak = s.last === y ? (s.streak || 0) + 1 : 1; s.last = dayKey(); }
    s.best = Math.max(s.best || 0, pct); localStorage.setItem("learn", JSON.stringify(s)); stats(); }
  setTimeout(() => $("#lresult").scrollIntoView({ behavior: "smooth", block: "start" }), 250);
};
$("#nextBtn").onclick = () => { let i; do i = Math.floor(Math.random() * 1000); while (learn.c && cases[i % cases.length] === learn.c); loadLearn(i); };

// ---------- confetti ----------
function confetti() {
  const cv = $("#confetti"), g = cv.getContext("2d"), dpr = devicePixelRatio || 1; cv.width = innerWidth * dpr; cv.height = innerHeight * dpr; g.scale(dpr, dpr);
  const cols = ["#7ce7ff", "#8a9bff", "#facc15", "#fb923c", "#f472b6", "#4ade80"], P = [];
  for (let i = 0; i < 160; i++) P.push({ x: innerWidth / 2 + (Math.random() - .5) * 80, y: innerHeight * .45, vx: (Math.random() - .5) * 14, vy: -Math.random() * 15 - 5, r: Math.random() * 6.28, vr: (Math.random() - .5) * .4, w: 6 + Math.random() * 6, h: 3 + Math.random() * 5, c: cols[i % cols.length] });
  const t0 = performance.now(); (function f(t) { const k = (t - t0) / 1000; g.clearRect(0, 0, innerWidth, innerHeight);
    P.forEach(p => { p.vy += .42; p.vx *= .985; p.x += p.vx; p.y += p.vy; p.r += p.vr; g.save(); g.globalAlpha = Math.max(0, 1 - k / 2.6); g.translate(p.x, p.y); g.rotate(p.r); g.fillStyle = p.c; g.fillRect(-p.w / 2, -p.h / 2, p.w, p.h); g.restore(); });
    if (k < 2.7) requestAnimationFrame(f); else g.clearRect(0, 0, innerWidth, innerHeight); })(t0);
}

// ---------- boot ----------
stats(); hero().catch(() => {}); show(location.hash.slice(1) || "home");
window.__opg = { learn, get findings() { return lastFindings; }, select, explain, confetti, loadLearn, runLook };
