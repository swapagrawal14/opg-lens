// OPG Lens service worker: app shell + samples precached, models and CDN files cached on first use.
const V = "opg-lens-v2.1.0";            // bump on every app file change
const MODELS = "opg-lens-models-v0";    // bump only when models/*.onnx change
const SHELL = ["./", "index.html", "style.css", "app.js", "content.js", "manifest.json", "icon.svg", "docs/icon-512.png", "samples/cases.json",
  ...[1, 2, 3, 4, 5, 6, 7, 8].map(i => `samples/case${i}.jpg`)];
self.addEventListener("install", e => { e.waitUntil(caches.open(V).then(c => c.addAll(SHELL)).then(() => self.skipWaiting())); });
self.addEventListener("activate", e => { e.waitUntil(caches.keys().then(ks => Promise.all(ks.filter(k => k !== V && k !== MODELS).map(k => caches.delete(k)))).then(() => self.clients.claim())); });
async function cacheFirst(req, name) {
  const c = await caches.open(name), hit = await c.match(req); if (hit) return hit;
  const res = await fetch(req); if (res.ok || res.type === "opaque") c.put(req, res.clone()); return res;
}
async function networkFirst(req) {
  const c = await caches.open(V);
  try { const res = await fetch(req); if (res.ok) c.put(req, res.clone()); return res; }
  catch (err) { const hit = await c.match(req, { ignoreSearch: true }); if (hit) return hit; if (req.mode === "navigate") { const idx = await c.match("index.html"); if (idx) return idx; } throw err; }
}
self.addEventListener("fetch", e => {
  const req = e.request; if (req.method !== "GET") return;
  const u = new URL(req.url);
  if (u.origin === location.origin) {
    if (u.pathname.includes("/models/")) return e.respondWith(cacheFirst(req, MODELS));
    if (u.pathname.includes("/samples/")) return e.respondWith(cacheFirst(req, V));
    if (/\/(docs|assets)\//.test(u.pathname) && !u.pathname.endsWith("icon-512.png")) return; // README media: no caching
    return e.respondWith(networkFirst(req));
  }
  if (/cdn\.jsdelivr\.net|fonts\.(googleapis|gstatic)\.com/.test(u.host)) return e.respondWith(cacheFirst(req, V));
});
