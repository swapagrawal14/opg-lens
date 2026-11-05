const $ = s => document.querySelector(s);
const S = 1024, CONF = 0.3;
let sessions = {}, cases = [], lastFindings = [], lastImg = null, lang = "en";

// ---------- tabs ----------
document.querySelectorAll(".tabs button").forEach(b => b.onclick = () => {
  document.querySelectorAll(".tabs button").forEach(x => x.classList.toggle("on", x === b));
  $("#look").hidden = b.dataset.t !== "look"; $("#learn").hidden = b.dataset.t !== "learn";
  if (b.dataset.t === "learn" && !learn.c) loadLearn(dailyIndex());
});

// ---------- models ----------
ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";
ort.env.wasm.numThreads = Math.min(4, navigator.hardwareConcurrency || 2);
async function model(name) {
  if (!sessions[name]) sessions[name] = ort.InferenceSession.create(`models/${name}.onnx`, { executionProviders: ["wasm"] });
  return sessions[name];
}
function letterbox(img) {
  const r = S / Math.max(img.width, img.height), nw = Math.round(img.width * r), nh = Math.round(img.height * r);
  const px = Math.floor((S - nw) / 2), py = Math.floor((S - nh) / 2);
  const c = document.createElement("canvas"); c.width = c.height = S; const g = c.getContext("2d");
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
async function analyse(img) {
  $("#status").textContent = "Loading models (first time ~40 MB, then cached)…";
  const lb = letterbox(img); await model("diagnosis"); await model("fdi");
  $("#status").textContent = "Analysing…"; const t0 = performance.now();
  const d = await detect("diagnosis", lb), t = await detect("fdi", lb);
  const res = d.map(f => { let bt = null, bi = 0; for (const k of t) { const v = iou(f.box, k.box); if (v > bi) { bi = v; bt = k; } }
    return { finding: DIS[f.cl], fdi: bt && bi > 0.2 ? FDI[bt.cl] : "?", p: f.p, box: f.box }; });
  $("#status").textContent = `Done in ${((performance.now() - t0) / 1000).toFixed(1)} s, on your device. ${res.length} finding${res.length === 1 ? "" : "s"} to double-check.`;
  return res;
}

// ---------- drawing ----------
function draw(cv, img, items, opts = {}) {
  cv.width = img.width; cv.height = img.height; const g = cv.getContext("2d"); g.drawImage(img, 0, 0);
  const lw = Math.max(2, img.width / 400); g.lineWidth = lw; g.font = `bold ${Math.round(img.width / 55)}px system-ui`;
  items.forEach((it, i) => {
    const c = it.color || FINDINGS[it.finding].color, [x1, y1, x2, y2] = it.box; g.strokeStyle = c; g.setLineDash(it.dash ? [lw * 3, lw * 2] : []);
    g.strokeRect(x1, y1, x2 - x1, y2 - y1); const lab = it.label ?? `${i + 1}${it.fdi && it.fdi !== "?" ? " · " + it.fdi : ""}`;
    const tw = g.measureText(lab).width + 10, th = img.width / 45; g.fillStyle = c; g.fillRect(x1, y1 - th, tw, th); g.fillStyle = "#04121f"; g.fillText(lab, x1 + 5, y1 - th * 0.22);
  });
  (opts.dots || []).forEach((d, i) => { g.setLineDash([]); g.fillStyle = "#fff"; g.strokeStyle = "#000"; g.beginPath(); g.arc(d.x, d.y, img.width / 90, 0, 7); g.fill(); g.stroke();
    g.fillStyle = "#000"; g.fillText(String(i + 1), d.x - img.width / 220, d.y + img.width / 160); });
}
function legend() { $("#legend").innerHTML = DIS.map(k => `<span><i style="background:${FINDINGS[k].color}"></i>${k}</span>`).join(""); }
const loadImg = src => new Promise((ok, no) => { const i = new Image(); i.onload = () => ok(i); i.onerror = no; i.src = src; });

// ---------- second look ----------
async function runLook(img) {
  lastImg = img; draw($("#cv"), img, []); legend();
  try { lastFindings = await analyse(img); } catch (e) { $("#status").textContent = "Could not run the model on this device: " + e.message; return; }
  draw($("#cv"), img, lastFindings);
  $("#list").innerHTML = lastFindings.length ? lastFindings.map((f, i) => `<div class="item" data-i="${i}"><div><b>${i + 1}. ${f.finding}</b><div class="muted">${f.fdi === "?" ? "Tooth number unclear" : "Tooth " + f.fdi + ", " + toothName(f.fdi)}</div></div><span class="pill">${Math.round(f.p * 100)}% sure</span></div>`).join("")
    : `<div class="item">No caries, deep caries, periapical lesion or impaction flagged. That does not rule them out.</div>`;
  document.querySelectorAll("#list .item[data-i]").forEach(el => el.onclick = () => explain(lastFindings[+el.dataset.i]));
  $("#reportBox").hidden = false; renderReport();
}
$("#file").onchange = async e => { const f = e.target.files[0]; if (!f) return; runLook(await loadImg(URL.createObjectURL(f))); };
$("#sampleBtn").onclick = async () => { await loadCases(); const c = cases[Math.floor(Math.random() * cases.length)]; runLook(await loadImg(c.img)); };

function reportText(l) {
  const today = new Date().toLocaleDateString(l === "hi" ? "hi-IN" : "en-IN");
  if (l === "hi") {
    const lines = lastFindings.map((f, i) => `${i + 1}. ${f.fdi !== "?" ? "दाँत " + f.fdi + " (" + QUAD_HI[f.fdi[0]] + ")" : "दाँत"}: ${FINDINGS[f.finding].patientHi}`);
    return `OPG सेकंड-लुक रिपोर्ट (${today})\n\n${lines.length ? "AI ने इन जगहों पर ध्यान देने को कहा है:\n" + lines.join("\n") : "AI को कोई खास बदलाव नहीं दिखा।"}\n\nयह निदान (diagnosis) नहीं है। कृपया अपने डेंटिस्ट से जाँच करवाएँ।`;
  }
  const lines = lastFindings.map((f, i) => `${i + 1}. ${f.fdi !== "?" ? "Tooth " + f.fdi + " (" + toothName(f.fdi) + ")" : "Unnumbered tooth"}: ${FINDINGS[f.finding].patient}. Confidence ${Math.round(f.p * 100)}%.`);
  return `OPG second-look report (${today})\n\n${lines.length ? "Areas the AI suggests checking:\n" + lines.join("\n") : "The AI did not flag caries, deep caries, periapical lesions or impacted teeth."}\n\nThis is a second-look aid, not a diagnosis. Please confirm with a clinical exam and your dentist.`;
}
function renderReport() { $("#report").textContent = reportText(lang); }
document.querySelectorAll(".seg button").forEach(b => b.onclick = () => { lang = b.dataset.l; document.querySelectorAll(".seg button").forEach(x => x.classList.toggle("on", x === b)); renderReport(); });
$("#copyBtn").onclick = () => navigator.clipboard.writeText($("#report").textContent);
$("#shareBtn").onclick = () => navigator.share ? navigator.share({ title: "OPG report", text: $("#report").textContent }) : navigator.clipboard.writeText($("#report").textContent);

// ---------- explanations + BYOK tutor ----------
let exCtx = null;
function explain(f) {
  const k = FINDINGS[f.finding]; exCtx = f;
  $("#exBody").innerHTML = `<h3 style="color:${k.color}">${f.finding} <span class="muted">(${k.hi})</span></h3>
  <p class="muted">${f.fdi && f.fdi !== "?" ? "Tooth " + f.fdi + ", " + toothName(f.fdi) : ""}</p>
  <p><b>What it looks like:</b> ${k.looks}</p><p><b>Why:</b> ${k.why}</p><p><b>Don't confuse with:</b> ${k.mimics}</p><p><b>Next step:</b> ${k.next}</p>`;
  $("#tutor").hidden = !byok().key; $("#tAns").textContent = ""; $("#exDlg").showModal();
}
const byok = () => JSON.parse(localStorage.getItem("byok") || "{}");
$("#setBtn").onclick = () => { const b = byok(); $("#sUrl").value = b.url || ""; $("#sKey").value = b.key || ""; $("#sModel").value = b.model || ""; $("#sVision").checked = !!b.vision; $("#setDlg").showModal(); };
$("#sSave").onclick = () => localStorage.setItem("byok", JSON.stringify({ url: $("#sUrl").value.trim().replace(/\/$/, ""), key: $("#sKey").value.trim(), model: $("#sModel").value.trim(), vision: $("#sVision").checked }));
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
async function loadCases() { if (!cases.length) cases = await (await fetch("samples/cases.json")).json(); }
const dayKey = () => new Date().toLocaleDateString("en-CA");
const dailyIndex = () => Math.floor(Date.now() / 864e5);
async function loadLearn(idx) {
  await loadCases(); learn.c = cases[idx % cases.length]; learn.img = await loadImg(learn.c.img); learn.picks = []; learn.revealed = false;
  learn.daily = idx === dailyIndex(); $("#dayLbl").textContent = learn.daily ? "OPG of the Day" : "Practice case";
  $("#lresult").innerHTML = ""; renderPicks(); draw($("#lcv"), learn.img, []); stats();
}
function stats() { const s = JSON.parse(localStorage.getItem("learn") || "{}"); $("#streak").textContent = s.streak || 0; $("#best").textContent = s.best || 0; }
let pending = null, pickF = null;
$("#fOpts").innerHTML = DIS.map(d => `<button type="button" data-f="${d}">${d}</button>`).join("");
$("#fOpts").onclick = e => { const b = e.target.closest("button"); if (!b) return; pickF = b.dataset.f; $("#fOpts").querySelectorAll("button").forEach(x => x.classList.toggle("on", x === b)); };
$("#lcv").onclick = e => { if (learn.revealed) return; const r = e.target.getBoundingClientRect(), sx = learn.img.width / r.width;
  pending = { x: (e.clientX - r.left) * sx, y: (e.clientY - r.top) * sx }; pickF = null; $("#fdiIn").value = ""; $("#fOpts").querySelectorAll("button").forEach(x => x.classList.remove("on")); $("#pickDlg").showModal(); };
$("#pickOk").onclick = e => { if (!pickF) { e.preventDefault(); return; } learn.picks.push({ ...pending, finding: pickF, fdi: $("#fdiIn").value.trim() }); renderPicks(); draw($("#lcv"), learn.img, [], { dots: learn.picks }); };
function renderPicks() { $("#picks").innerHTML = learn.picks.map((p, i) => `<div class="item"><span>${i + 1}. ${p.finding}${p.fdi ? " · " + p.fdi : ""}</span><button class="ghost" data-del="${i}">✕</button></div>`).join("");
  document.querySelectorAll("[data-del]").forEach(b => b.onclick = () => { learn.picks.splice(+b.dataset.del, 1); renderPicks(); draw($("#lcv"), learn.img, [], { dots: learn.picks }); }); }
$("#revealBtn").onclick = () => {
  if (learn.revealed) return; learn.revealed = true; const gt = learn.c.findings, used = new Set(); let score = 0; const rows = [];
  gt.forEach((g, gi) => { const [x, y, w, h] = g.box, m = w * 0.15; const hit = learn.picks.findIndex((p, pi) => !used.has(pi) && p.x > x - m && p.x < x + w + m && p.y > y - m && p.y < y + h + m);
    let pts = 0, note; if (hit < 0) note = `<span class="bad">Missed</span>`;
    else { used.add(hit); const p = learn.picks[hit]; pts = p.finding === g.finding ? 1 : (p.finding.includes("Caries") && g.finding.includes("Caries") ? 0.5 : 0.25); if (p.fdi === g.fdi) pts += 0.25;
      note = p.finding === g.finding ? `<span class="ok">Correct</span>` : `<span class="warn">Right tooth, you said ${p.finding}</span>`; if (p.fdi && p.fdi !== g.fdi) note += ` <span class="warn">· FDI ${p.fdi}≠${g.fdi}</span>`; }
    score += pts; rows.push(`<div class="item" data-g="${gi}"><div><b>${g.finding}</b> · ${g.fdi}<div class="muted">${toothName(g.fdi)}</div></div><div>${note}</div></div>`); });
  const extra = learn.picks.length - used.size, max = gt.length * 1.25, pct = Math.round(100 * Math.max(0, score - extra * 0.25) / max);
  draw($("#lcv"), learn.img, gt.map(g => ({ ...g, box: [g.box[0], g.box[1], g.box[0] + g.box[2], g.box[1] + g.box[3]], label: g.fdi })), { dots: learn.picks });
  $("#lresult").innerHTML = `<h3>Score ${pct}%</h3>${extra > 0 ? `<p class="muted">${extra} mark${extra > 1 ? "s" : ""} where the experts saw nothing (−0.25 each).</p>` : ""}${rows.join("")}<p class="tiny">Answer key: expert labels from DENTEX. They mark only these 4 finding types, so other pathology isn't scored. Tap a finding to learn it.</p>`;
  document.querySelectorAll("[data-g]").forEach(el => el.onclick = () => explain(learn.c.findings[+el.dataset.g]));
  if (learn.daily) { const s = JSON.parse(localStorage.getItem("learn") || "{}"); if (s.last !== dayKey()) { const y = new Date(Date.now() - 864e5).toLocaleDateString("en-CA"); s.streak = s.last === y ? (s.streak || 0) + 1 : 1; s.last = dayKey(); }
    s.best = Math.max(s.best || 0, pct); localStorage.setItem("learn", JSON.stringify(s)); stats(); }
};
$("#nextBtn").onclick = () => loadLearn(Math.floor(Math.random() * 1000));
