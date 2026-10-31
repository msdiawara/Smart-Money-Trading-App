/* Smart Money Radar — small SVG chart kit (no dependencies; the page's CSP allows only its own scripts).
   fan(): price history + forecast cone (+ Claude markers, event ticks), with a crosshair and tooltip.
   lines(): month-by-month trends with an optional target band. bars(): simple one-series columns.
   Colours come from CSS custom properties, so light and dark themes restyle every chart. */
(() => {
  "use strict";
  const NS = "http://www.w3.org/2000/svg";
  const el = (tag, attrs = {}, parent) => {
    const e = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs)) if (v != null) e.setAttribute(k, v);
    if (parent) parent.appendChild(e);
    return e;
  };
  const text = (parent, x, y, s, attrs = {}) => { const t = el("text", { x, y, ...attrs }, parent); t.textContent = s; return t; };
  // dates are trading days ("2026-10-30"): placed at noon UTC and always formatted in UTC, so a phone far east or
  // west of New York never shows the day before or after
  const day = (s) => new Date(s + "T12:00:00Z").getTime();
  const fmtDate = (ms, withYear) => new Date(ms).toLocaleDateString(undefined, withYear ? { month: "short", day: "numeric", year: "numeric", timeZone: "UTC" } : { month: "short", day: "numeric", timeZone: "UTC" });
  const fmtMonth = (ms) => new Date(ms).toLocaleDateString(undefined, { month: "short", year: "numeric", timeZone: "UTC" });   // "Nov 2025", never "Nov 25"
  const decimalsFor = (a) => (a >= 1000 ? 0 : a >= 1 ? 2 : a >= 0.01 ? 4 : 6);
  const money = (v) => {
    if (v == null || !isFinite(v)) return "—";
    const d = decimalsFor(Math.abs(v));
    return "$" + Number(v).toLocaleString(undefined, { minimumFractionDigits: Math.min(2, d), maximumFractionDigits: d });
  };
  const decOf = (step) => { let d = 0; while (d < 6 && Math.abs(Math.round(step * 10 ** d) - step * 10 ** d) > 1e-6 * 10 ** d) d++; return d; };
  const good = (v) => v != null && isFinite(v) && v > 0;
  function liveRegion(box) {
    let r = box.querySelector(".viz-live");
    if (!r) { r = document.createElement("div"); r.className = "viz-live sr-only"; r.setAttribute("aria-live", "polite"); box.appendChild(r); }
    return r;
  }
  const tipText = (title, rows) => [title].concat(rows.map((r) => `${r.label ? r.label + ": " : ""}${r.value}`)).join(". ");

  function niceTicks(lo, hi, n = 4) {
    if (!(hi > lo)) { hi = lo + 1; }
    const span = hi - lo, raw = span / n, mag = Math.pow(10, Math.floor(Math.log10(raw)));
    const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => span / s <= n + 0.5) || 10 * mag;
    const out = [];
    for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) out.push(+v.toFixed(10));
    return out;
  }

  function tooltip(box) {
    let t = box.querySelector(".viz-tip");
    if (!t) { t = document.createElement("div"); t.className = "viz-tip"; t.hidden = true; box.appendChild(t); }
    return t;
  }
  // rows: [{key: "line"|"dot"|"band"|"none", color, value, label}] — values lead, labels follow (textContent only)
  function fillTip(t, title, rows) {
    t.replaceChildren();
    const h = document.createElement("div"); h.className = "viz-tip-h"; h.textContent = title; t.appendChild(h);
    for (const r of rows) {
      const row = document.createElement("div"); row.className = "viz-tip-r";
      const k = document.createElement("span"); k.className = "viz-key " + (r.key || "line"); if (r.color) k.style.setProperty("--k", r.color);
      const v = document.createElement("b"); v.textContent = r.value;
      const l = document.createElement("span"); l.className = "viz-tip-l"; l.textContent = r.label;
      row.append(k, v, l); t.appendChild(row);
    }
  }
  function placeTip(box, t, x, y) {
    t.hidden = false;
    const bw = box.clientWidth, tw = t.offsetWidth, th = t.offsetHeight;
    let left = x + 14; if (left + tw > bw - 4) left = x - tw - 14; if (left < 4) left = 4;
    let top = y - th - 10; if (top < 2) top = y + 14;
    t.style.left = left + "px"; t.style.top = top + "px";
  }

  // ------------------------------------------------------------------ forecast fan
  const RANGES = { "1d": { hist: 10, h: 1 }, "1w": { hist: 21, h: 5 }, "1m": { hist: 63, h: 21 }, "3m": { hist: 126, h: 63 }, "1y": { hist: 504, h: 252 } };

  function fan(box, cfg) {
    const { history, fan: F, claude, events, range = "3m", labels = {} } = cfg;
    box.replaceChildren();
    box.classList.add("viz");
    const R = RANGES[range] || RANGES["3m"];
    const hd = (history && history.d) || [], hc = (history && history.c) || [];
    const hi0 = Math.max(0, hd.length - R.hist);
    const H = hd.slice(hi0).map((d, i) => ({ t: day(d), d, v: hc[hi0 + i] })).filter((p) => good(p.v));   // a missing close is a gap, not $0
    // fan points within the horizon: the fan's own dates carry the trading-day count implicitly (index 0 = today)
    const fd = (F && F.dates) || [];
    const hDate = cfg.horizonDate;            // the target date for the chosen horizon
    const P = [];
    for (let i = 0; i < fd.length; i++) {
      if (hDate && fd[i] > hDate) break;
      const q = ["05", "10", "25", "50", "75", "90", "95"].map((k) => (F.p[k] || [])[i]);
      if (q.every(good)) P.push({ t: day(fd[i]), d: fd[i], q });
    }
    // Claude's targets are counted from the session after its forecast, so they can sit a day or two past the
    // model's horizon date; keep those (the x range stretches to fit them)
    const hLim = hDate ? new Date(day(hDate) + 4 * 864e5).toISOString().slice(0, 10) : null;
    const C = (claude || []).filter((c) => c && c.date && (!hLim || c.date <= hLim) && good(c.lo) && good(c.mid) && good(c.hi));
    const width = Math.max(280, box.clientWidth || 320), ph = cfg.height || 236, ax = 26, padL = 6, padT = 12;
    const t0 = H.length ? H[0].t : P.length ? P[0].t : Date.now();
    const t1 = Math.max(P.length ? P[P.length - 1].t : H.length ? H[H.length - 1].t : t0 + 864e5, ...C.map((c) => day(c.date)));
    const span = Math.max(864e5, t1 - t0);
    let lo = Infinity, hi = -Infinity;
    for (const p of H) { lo = Math.min(lo, p.v); hi = Math.max(hi, p.v); }
    for (const p of P) { lo = Math.min(lo, p.q[0]); hi = Math.max(hi, p.q[6]); }
    for (const c of C) { lo = Math.min(lo, c.lo); hi = Math.max(hi, c.hi); }
    if (!isFinite(lo)) { lo = 0; hi = 1; }
    const pad = (hi - lo) * 0.06 || hi * 0.05 || 1;
    lo = Math.max(0, lo - pad); hi += pad;
    // y labels on the right (finance convention), with as many decimals as the step needs ($112.5 stays $112.5)
    const ticks = niceTicks(lo, hi, 4), stepV = ticks.length > 1 ? ticks[1] - ticks[0] : 1, dec = decOf(stepV);
    const tickFmt = (v) => "$" + v.toLocaleString(undefined, { minimumFractionDigits: dec, maximumFractionDigits: dec });
    const padR = Math.max(54, 12 + Math.max(...ticks.map((v) => tickFmt(v).length)) * 6.8);     // room for $120,000
    const x = (t) => padL + ((t - t0) / span) * (width - padL - padR);
    const svg = el("svg", { width, height: ph + ax + padT, viewBox: `0 0 ${width} ${ph + ax + padT}`, role: "img", "aria-label": cfg.aria || "Price history and forecast range", class: "viz-svg" }, box);
    const y = (v) => padT + ph - ((v - lo) / (hi - lo)) * ph;
    const g = el("g", {}, svg);
    for (const v of ticks) {
      el("line", { x1: padL, x2: width - padR + 4, y1: y(v), y2: y(v), class: "viz-grid" }, g);
      text(g, width - padR + 8, y(v) + 4, tickFmt(v), { class: "viz-tick" });
    }
    // x labels: 3-5 evenly spaced dates
    const nx = width < 380 ? 3 : 4;
    for (let k = 0; k <= nx; k++) {
      const tt = t0 + (span * k) / nx;
      const anchor = k === 0 ? "start" : k === nx ? "end" : "middle";
      text(g, x(tt), padT + ph + 18, span > 200 * 864e5 ? fmtMonth(tt) : fmtDate(tt), { class: "viz-tick", "text-anchor": anchor });
    }
    el("line", { x1: padL, x2: width - padR + 4, y1: padT + ph, y2: padT + ph, class: "viz-axis" }, g);
    // bands (nested washes: 90 → 80 → 50), median line
    if (P.length > 1) {
      const band = (a, b, cls) => {
        const top = P.map((p) => `${x(p.t).toFixed(1)},${y(p.q[b]).toFixed(1)}`);
        const bot = P.slice().reverse().map((p) => `${x(p.t).toFixed(1)},${y(p.q[a]).toFixed(1)}`);
        el("polygon", { points: top.concat(bot).join(" "), class: cls }, g);
      };
      band(0, 6, "viz-band b90"); band(1, 5, "viz-band b80"); band(2, 4, "viz-band b50");
      el("polyline", { points: P.map((p) => `${x(p.t).toFixed(1)},${y(p.q[3]).toFixed(1)}`).join(" "), class: "viz-median" }, g);
    }
    // today marker
    const today = P.length ? P[0] : H[H.length - 1];
    if (today) {
      el("line", { x1: x(today.t), x2: x(today.t), y1: padT, y2: padT + ph, class: "viz-today" }, g);
      text(g, x(today.t) - 4, padT + 10, labels.today || "Last close", { class: "viz-note", "text-anchor": "end" });
    }
    // history
    if (H.length > 1) el("polyline", { points: H.map((p) => `${x(p.t).toFixed(1)},${y(p.v).toFixed(1)}`).join(" "), class: "viz-price" }, g);
    // events (ticks along the bottom of the plot)
    const evs = (events || []).filter((e) => e.date && day(e.date) >= t0 && day(e.date) <= t1);
    const evG = el("g", { class: "viz-events" }, g);
    // labels only where they don't collide; earnings first (they carry the most risk); every tick stays
    const placed = [];
    for (const e of evs.slice().sort((a, b) => (a.kind === "earnings" ? 0 : 1) - (b.kind === "earnings" ? 0 : 1))) {
      const ex = x(day(e.date));
      el("line", { x1: ex, x2: ex, y1: padT + ph - 7, y2: padT + ph, class: "viz-evtick " + e.kind }, evG);
      const lab = e.kind === "earnings" ? "E" : e.kind === "fomc" ? "Fed" : "CPI", w = lab.length * 6 + 6;
      if (placed.every((p) => Math.abs(p - ex) > (w + 18) / 2)) { placed.push(ex); text(evG, ex, padT + ph - 10, lab, { class: "viz-evlabel" + (e.kind === "earnings" ? " strong" : ""), "text-anchor": "middle" }); }
    }
    // Claude markers with 10–90 whiskers
    for (const c of C) {
      const cx = x(day(c.date));
      el("line", { x1: cx, x2: cx, y1: y(c.hi), y2: y(c.lo), class: "viz-whisker" }, g);
      el("line", { x1: cx - 4, x2: cx + 4, y1: y(c.hi), y2: y(c.hi), class: "viz-whisker" }, g);
      el("line", { x1: cx - 4, x2: cx + 4, y1: y(c.lo), y2: y(c.lo), class: "viz-whisker" }, g);
      el("circle", { cx, cy: y(c.mid), r: 4.5, class: "viz-claude" }, g);
    }
    // direct label: median at the end of the cone
    if (P.length > 1) {
      const e = P[P.length - 1];
      el("circle", { cx: x(e.t), cy: y(e.q[3]), r: 4, class: "viz-enddot" }, g);
    }
    // crosshair layer
    const pts = H.map((p) => ({ t: p.t, d: p.d, hist: p.v })).concat(P.slice(1).map((p) => ({ t: p.t, d: p.d, q: p.q })));
    const cross = el("line", { y1: padT, y2: padT + ph, class: "viz-cross", visibility: "hidden" }, g);
    const dotH = el("circle", { r: 4, class: "viz-dot", visibility: "hidden" }, g);
    const tip = tooltip(box), live = liveRegion(box);
    let cur = -1;
    const show = (i, announce) => {
      if (i < 0 || i >= pts.length) return;
      cur = i;
      const p = pts[i], px = x(p.t);
      cross.setAttribute("x1", px); cross.setAttribute("x2", px); cross.setAttribute("visibility", "visible");
      const rows = [];
      if (p.hist != null) {
        dotH.setAttribute("cx", px); dotH.setAttribute("cy", y(p.hist)); dotH.setAttribute("visibility", "visible");
        rows.push({ key: "line", color: "var(--ink)", value: money(p.hist), label: "Close" });
      } else {
        dotH.setAttribute("cx", px); dotH.setAttribute("cy", y(p.q[3])); dotH.setAttribute("visibility", "visible");
        rows.push({ key: "line", color: "var(--series-1)", value: money(p.q[3]), label: "Median forecast" });
        rows.push({ key: "band", color: "var(--series-1)", value: `${money(p.q[2])}–${money(p.q[4])}`, label: "50% range" });
        rows.push({ key: "band", color: "var(--series-1)", value: `${money(p.q[1])}–${money(p.q[5])}`, label: "80% range" });
      }
      const c = C.find((c) => Math.abs(day(c.date) - p.t) < 2.5 * 864e5);
      if (c) rows.push({ key: "dot", color: "var(--series-2)", value: `${money(c.mid)} (${money(c.lo)}–${money(c.hi)})`, label: "Claude" });
      const ev = evs.find((e) => Math.abs(day(e.date) - p.t) < 1.5 * 864e5);
      if (ev) rows.push({ key: "none", value: ev.label, label: "" });
      fillTip(tip, fmtDate(p.t, true), rows);
      if (announce) live.textContent = tipText(fmtDate(p.t, true), rows);
      const r = svg.getBoundingClientRect(), b = box.getBoundingClientRect();
      placeTip(box, tip, px * (r.width / width) + (r.left - b.left), (p.hist != null ? y(p.hist) : y(p.q[3])) * (r.height / (ph + ax + padT)) + (r.top - b.top));
    };
    const hide = () => { cross.setAttribute("visibility", "hidden"); dotH.setAttribute("visibility", "hidden"); tip.hidden = true; cur = -1; };
    const nearest = (clientX) => {
      const r = svg.getBoundingClientRect();
      const tt = t0 + ((clientX - r.left) * (width / r.width) - padL) / (width - padL - padR) * span;
      let best = 0, bd = Infinity;
      for (let i = 0; i < pts.length; i++) { const d = Math.abs(pts[i].t - tt); if (d < bd) { bd = d; best = i; } }
      return best;
    };
    const hit = el("rect", { x: 0, y: 0, width, height: ph + padT, fill: "transparent", class: "viz-hit" }, svg);
    hit.addEventListener("pointermove", (ev) => show(nearest(ev.clientX)));
    hit.addEventListener("pointerdown", (ev) => show(nearest(ev.clientX)));
    hit.addEventListener("pointerleave", (ev) => { if (ev.pointerType === "mouse") hide(); });
    box.tabIndex = 0;
    box.setAttribute("role", "group");
    box.setAttribute("aria-label", (cfg.aria || "Forecast chart") + ". Use the left and right arrow keys to read values.");
    box.onkeydown = (ev) => {
      if (ev.key === "ArrowRight" || ev.key === "ArrowLeft") { ev.preventDefault(); show(Math.max(0, Math.min(pts.length - 1, (cur < 0 ? Math.max(0, H.length - 1) : cur) + (ev.key === "ArrowRight" ? 1 : -1))), true); }
      else if (ev.key === "Escape") hide();
    };
    box.onblur = hide;
    return { svg };
  }

  // ------------------------------------------------------------------ trend lines (one shared axis)
  function lines(box, cfg) {
    // fmt(value, decimals): decimals come from the tick step, so close ticks never print the same label twice
    const { x: labels, series, yDomain, band, ref, fmt = (v, d = 0) => `${(v * 100).toFixed(d)}%`, height = 170 } = cfg;
    box.replaceChildren(); box.classList.add("viz");
    const width = Math.max(260, box.clientWidth || 300), padL = 42, padR = 10, padT = 10, ax = 22;
    const svg = el("svg", { width, height: height + ax + padT, viewBox: `0 0 ${width} ${height + ax + padT}`, role: "img", "aria-label": cfg.aria || "Trend chart", class: "viz-svg" }, box);
    const n = labels.length;
    const xs = (i) => padL + (n <= 1 ? (width - padL - padR) / 2 : (i / (n - 1)) * (width - padL - padR));
    // the domain always fits the data (a suggested domain is a minimum, never a crop); rates stay within 0-100%
    let lo = Infinity, hi = -Infinity;
    for (const s of series) for (const v of s.values) if (v != null) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
    if (ref) { lo = Math.min(lo, ref.y); hi = Math.max(hi, ref.y); }
    if (band) { lo = Math.min(lo, band.lo); hi = Math.max(hi, band.hi); }
    if (!isFinite(lo)) { lo = 0; hi = 1; }
    const pd = (hi - lo) * 0.12 || 0.05; lo -= pd; hi += pd;
    if (yDomain) { lo = Math.min(lo, yDomain[0]); hi = Math.max(hi, yDomain[1]); if (cfg.clamp01 !== false && yDomain[0] >= 0 && yDomain[1] <= 1) { lo = Math.max(0, lo); hi = Math.min(1, hi); } }
    const y = (v) => padT + height - ((v - lo) / (hi - lo)) * height;
    const g = el("g", {}, svg);
    const ticks = niceTicks(lo, hi, 4), dec = Math.min(2, decOf(+((ticks.length > 1 ? ticks[1] - ticks[0] : 1) * 100).toFixed(8)));
    for (const v of ticks) { el("line", { x1: padL, x2: width - padR, y1: y(v), y2: y(v), class: "viz-grid" }, g); text(g, padL - 6, y(v) + 4, fmt(v, dec), { class: "viz-tick", "text-anchor": "end" }); }
    if (band) el("rect", { x: padL, width: width - padL - padR, y: y(band.hi), height: Math.max(1, y(band.lo) - y(band.hi)), class: "viz-target" }, g);
    if (ref) { el("line", { x1: padL, x2: width - padR, y1: y(ref.y), y2: y(ref.y), class: "viz-ref" }, g); if (ref.label) text(g, padL + 4, y(ref.y) - 5, ref.label, { class: "viz-note" }); }
    const step = Math.max(1, Math.ceil(n / (width < 380 ? 4 : 6)));
    for (let i = 0; i < n; i += step) text(g, xs(i), padT + height + 16, labels[i], { class: "viz-tick", "text-anchor": n === 1 ? "middle" : i === 0 ? "start" : i === n - 1 ? "end" : "middle" });
    el("line", { x1: padL, x2: width - padR, y1: padT + height, y2: padT + height, class: "viz-axis" }, g);
    for (const s of series) {
      let seg = [];
      const flush = () => { if (seg.length > 1) el("polyline", { points: seg.join(" "), class: "viz-line", style: `stroke:${s.color}` }, g); seg = []; };
      s.values.forEach((v, i) => { if (v == null) flush(); else seg.push(`${xs(i).toFixed(1)},${y(v).toFixed(1)}`); });
      flush();
      s.values.forEach((v, i) => { if (v != null && (s.values[i - 1] == null && s.values[i + 1] == null || i === s.values.length - 1)) el("circle", { cx: xs(i), cy: y(v), r: 4, class: "viz-pt", style: `fill:${s.color}` }, g); });
    }
    const cross = el("line", { y1: padT, y2: padT + height, class: "viz-cross", visibility: "hidden" }, g);
    const tip = tooltip(box), live = liveRegion(box);
    let cur = -1;
    const show = (i, announce) => {
      cur = i;
      cross.setAttribute("x1", xs(i)); cross.setAttribute("x2", xs(i)); cross.setAttribute("visibility", "visible");
      const rows = series.map((s) => ({ key: "line", color: s.color, value: s.values[i] == null ? "—" : fmt(s.values[i], dec), label: s.name + (s.n && s.n[i] != null ? ` (n=${s.n[i]})` : "") }));
      const title = cfg.tipTitle ? cfg.tipTitle(i) : labels[i];
      fillTip(tip, title, rows);
      if (announce) live.textContent = tipText(title, rows);
      const r = svg.getBoundingClientRect(), b = box.getBoundingClientRect();
      placeTip(box, tip, xs(i) * (r.width / width) + (r.left - b.left), padT + 10);
    };
    box.tabIndex = 0;
    box.setAttribute("role", "group");
    box.setAttribute("aria-label", (cfg.aria || "Trend chart") + ". Use the left and right arrow keys to read each month.");
    box.onkeydown = (ev) => {
      if (ev.key === "ArrowRight" || ev.key === "ArrowLeft") { ev.preventDefault(); show(Math.max(0, Math.min(n - 1, (cur < 0 ? n - 1 : cur) + (ev.key === "ArrowRight" ? 1 : -1))), true); }
      else if (ev.key === "Escape") { cross.setAttribute("visibility", "hidden"); tip.hidden = true; cur = -1; }
    };
    box.onblur = () => { cross.setAttribute("visibility", "hidden"); tip.hidden = true; cur = -1; };
    const hit = el("rect", { x: 0, y: 0, width, height: height + padT, fill: "transparent", class: "viz-hit" }, svg);
    const idx = (cx) => { const r = svg.getBoundingClientRect(); const xx = (cx - r.left) * (width / r.width); return Math.max(0, Math.min(n - 1, Math.round((xx - padL) / ((width - padL - padR) / Math.max(1, n - 1))))); };
    hit.addEventListener("pointermove", (ev) => show(idx(ev.clientX)));
    hit.addEventListener("pointerdown", (ev) => show(idx(ev.clientX)));
    hit.addEventListener("pointerleave", (ev) => { if (ev.pointerType === "mouse") { cross.setAttribute("visibility", "hidden"); tip.hidden = true; } });
    if (series.length > 1) {
      const lg = document.createElement("div"); lg.className = "viz-legend";
      for (const s of series) { const it = document.createElement("span"); const k = document.createElement("i"); k.className = "viz-key line"; k.style.setProperty("--k", s.color); it.append(k, document.createTextNode(s.name)); lg.appendChild(it); }
      if (band && band.label) { const it = document.createElement("span"); const k = document.createElement("i"); k.className = "viz-key band"; k.style.setProperty("--k", "var(--good)"); it.append(k, document.createTextNode(band.label)); lg.appendChild(it); }
      box.appendChild(lg);
    }
  }

  // ------------------------------------------------------------------ columns (one series, optional intervals)
  function bars(box, cfg) {
    const { labels, values, lo: los, hi: his, color = "var(--series-1)", fmt = (v) => `${Math.round(v * 100)}%`, ref, height = 140, yMax } = cfg;
    box.replaceChildren(); box.classList.add("viz");
    const width = Math.max(240, box.clientWidth || 300), padL = 34, padR = 8, padT = 12, ax = 22, n = labels.length;
    const svg = el("svg", { width, height: height + ax + padT, viewBox: `0 0 ${width} ${height + ax + padT}`, role: "group", "aria-label": cfg.aria || "Bar chart", class: "viz-svg" }, box);
    const top = yMax != null ? yMax : Math.max(...values.map((v, i) => Math.max(v || 0, his ? his[i] || 0 : 0)), ref ? ref.y : 0) * 1.15 || 1;
    const y = (v) => padT + height - (v / top) * height;
    const g = el("g", {}, svg);
    for (const v of niceTicks(0, top, 3)) { el("line", { x1: padL, x2: width - padR, y1: y(v), y2: y(v), class: "viz-grid" }, g); text(g, padL - 6, y(v) + 4, fmt(v), { class: "viz-tick", "text-anchor": "end" }); }
    const slot = (width - padL - padR) / n, bw = Math.min(24, slot * 0.6);
    const tip = tooltip(box);
    values.forEach((v, i) => {
      const cx = padL + slot * (i + 0.5);
      if (v != null && v > 0) {
        const h = Math.max(1, y(0) - y(v)), r = Math.min(4, bw / 2, h);
        const x0 = cx - bw / 2, y0 = y(v);
        el("path", { d: `M${x0},${y(0)} V${y0 + r} Q${x0},${y0} ${x0 + r},${y0} H${x0 + bw - r} Q${x0 + bw},${y0} ${x0 + bw},${y0 + r} V${y(0)} Z`, class: "viz-bar", style: `fill:${color}` }, g);
      }
      if (los && his && los[i] != null && his[i] != null) {
        el("line", { x1: cx, x2: cx, y1: y(his[i]), y2: y(los[i]), class: "viz-ci" }, g);
        el("line", { x1: cx - 4, x2: cx + 4, y1: y(his[i]), y2: y(his[i]), class: "viz-ci" }, g);
        el("line", { x1: cx - 4, x2: cx + 4, y1: y(los[i]), y2: y(los[i]), class: "viz-ci" }, g);
      }
      text(g, cx, padT + height + 16, labels[i], { class: "viz-tick", "text-anchor": "middle" });
      const said = `${labels[i]}: ${v == null ? "no data" : fmt(v)}${cfg.valueLabel ? ` ${cfg.valueLabel}` : ""}${los && his && los[i] != null && his[i] != null ? ` (95% range ${fmt(los[i])} to ${fmt(his[i])})` : ""}`;
      const hit = el("rect", { x: cx - slot / 2, y: padT, width: slot, height, fill: "transparent", class: "viz-hit", tabindex: 0, role: "img", "aria-label": said }, g);
      const showIt = () => { fillTip(tip, labels[i], [{ key: "band", color, value: v == null ? "—" : fmt(v), label: cfg.valueLabel || "" }].concat(los && his && los[i] != null ? [{ key: "none", value: `${fmt(los[i])}–${fmt(his[i])}`, label: "95% range" }] : []).concat(cfg.extra ? cfg.extra(i) : []));
        const r = svg.getBoundingClientRect(), b = box.getBoundingClientRect(); placeTip(box, tip, cx * (r.width / width) + (r.left - b.left), y(v || 0) * (r.height / (height + ax + padT)) + (r.top - b.top)); };
      hit.addEventListener("pointerenter", showIt); hit.addEventListener("pointerdown", showIt); hit.addEventListener("focus", showIt);
      hit.addEventListener("pointerleave", (ev) => { if (ev.pointerType === "mouse") tip.hidden = true; }); hit.addEventListener("blur", () => (tip.hidden = true));
    });
    el("line", { x1: padL, x2: width - padR, y1: y(0), y2: y(0), class: "viz-axis" }, g);
    if (ref) { el("line", { x1: padL, x2: width - padR, y1: y(ref.y), y2: y(ref.y), class: "viz-ref" }, g); if (ref.label) text(g, width - padR, y(ref.y) - 5, ref.label, { class: "viz-note", "text-anchor": "end" }); }
  }

  // re-draw charts on resize (each chart registers a redraw function)
  const redraws = new Set();
  let rt;
  window.addEventListener("resize", () => { clearTimeout(rt); rt = setTimeout(() => { for (const f of redraws) { try { f(); } catch { redraws.delete(f); } } }, 150); });
  function keep(box, draw) { draw(); const f = () => { if (!box.isConnected) { redraws.delete(f); return; } draw(); }; redraws.add(f); }

  window.SMRCharts = { fan, lines, bars, keep, niceTicks, RANGES };
})();
