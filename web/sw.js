/* Offline support. Network-first, so a new version of the app or data shows up on the next open; on a weak signal
   the saved copy is used after a few seconds; offline, the last good copy is served. A data file that was never
   saved answers 503 (not an empty file), so the app says "not saved for offline use" instead of "no data". A data
   file answered from the saved copy carries an X-SMR-From-Cache header, so the app can say it's showing a saved
   copy and check again next time. Bump VERSION when the shell changes. */
const VERSION = "v10";
const SHELL = `smr-shell-${VERSION}`;
const DATA = "smr-data-v1";
const TIMEOUT_MS = 3500;
const ASSETS = ["./", "index.html", "styles.css", "theme.js", "charts.js", "app.js", "stock.js", "accuracy.js", "results.js", "chat.js",
  "manifest.webmanifest", "icons/icon-192.png", "icons/icon-512.png", "fonts/ibm-plex-sans-latin-400-normal.woff2",
  "fonts/ibm-plex-sans-latin-500-normal.woff2", "fonts/ibm-plex-sans-latin-600-normal.woff2"];
self.addEventListener("install", (e) => {
  // "reload" skips the browser's HTTP cache, so a new version never installs ten-minute-old files
  e.waitUntil(caches.open(SHELL).then((c) => c.addAll(ASSETS.map((u) => new Request(u, { cache: "reload" })))).then(() => self.skipWaiting()));
});
self.addEventListener("activate", (e) => {
  e.waitUntil(caches.keys().then((ks) => Promise.all(ks.filter((k) => ![SHELL, DATA].includes(k)).map((k) => caches.delete(k))))
    .then(() => self.clients.claim()));
});
const saved = (cacheName, key) => caches.open(cacheName).then((c) => c.match(key, { ignoreSearch: true }))
  .then((r) => (r && cacheName === DATA ? marked(r) : r));
function marked(r) {
  const h = new Headers(r.headers);
  h.set("X-SMR-From-Cache", "1");
  return new Response(r.body, { status: r.status, statusText: r.statusText, headers: h });
}
function networkFirst(fetchIt, cacheName, key) {
  return new Promise((resolve) => {
    let done = false;
    const finish = (r) => { if (!done) { done = true; resolve(r || null); } };
    fetchIt().then((r) => {
      if (r.ok) { const copy = r.clone(); caches.open(cacheName).then((c) => c.put(key, copy)); }
      finish(r);
    }).catch(() => saved(cacheName, key).then(finish));
    setTimeout(() => saved(cacheName, key).then((r) => { if (r) finish(r); }), TIMEOUT_MS);   // slow network: use the copy
  });
}
self.addEventListener("fetch", (e) => {
  const req = e.request, url = new URL(req.url);
  if (req.method !== "GET" || url.origin !== location.origin) return;
  if (url.pathname.includes("/data/")) {
    const key = url.origin + url.pathname;            // ignore the cache-busting query
    e.respondWith(networkFirst(() => fetch(req), DATA, key).then((r) => r ||
      new Response(JSON.stringify({ offline: true }), { status: 503, statusText: "Offline, not saved", headers: { "Content-Type": "application/json" } })));
    return;
  }
  // app files: revalidate with the server every time (GitHub Pages lets browsers keep files for 10 minutes)
  const fresh = () => fetch(req.mode === "navigate" ? url.href : req, { cache: "no-cache" });   // a navigate Request can't take options
  e.respondWith(networkFirst(fresh, SHELL, req.mode === "navigate" ? url.origin + url.pathname : req.url)
    .then((r) => r || caches.match("index.html")));
});
