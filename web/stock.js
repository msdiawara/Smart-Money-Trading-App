/* Smart Money Radar — the stock page: forecast chart, valuation, bubble risk, news, conviction checklist.
   Data: data/stock/<TICKER>.json (written by the pipeline), plus the feed and quotes already loaded by app.js. */
(() => {
  "use strict";
  const HZ = [["1d", "1 day"], ["1w", "1 week"], ["1m", "1 month"], ["3m", "3 months"], ["1y", "1 year"]];
  const cache = new Map();          // ticker → {version, doc}
  const state = { range: null };

  async function load(t) {
    const S = window.SMR.data(), v = S.version || Date.now();
    const hit = cache.get(t);
    if (hit && hit.version === v) return hit.doc;
    const r = await fetch(`data/stock/${encodeURIComponent(t)}.json?v=${v}`, { cache: "no-store" });
    if (r.status === 404) return null;
    if (!r.ok) {
      let off = false;
      if (r.status === 503) { try { off = !!(await r.json()).offline; } catch { /* a server error page */ } }
      const e = new Error(off ? "offline, not saved" : `HTTP ${r.status}`); e.offline = off; throw e;
    }
    const doc = await r.json();
    if (r.headers.get("X-SMR-From-Cache")) { doc._saved = true; cache.delete(t); }   // a saved copy: fetch again next time
    else cache.set(t, { version: v, doc });
    return doc;
  }

  const RATING = { top: ["Top conviction", "top", "★"], positive: ["Positive", "pos", "▲"], neutral: ["Neutral", "neu", "•"],
    caution: ["Caution", "cau", "!"], red: ["Red flags", "red", "⚑"] };
  const BUB = { Low: ["low", "•"], Moderate: ["mod", "!"], Elevated: ["ele", "!"], High: ["hig", "▲"] };
  const VAL_CLASS = { "Very cheap": "vc", Cheap: "c", "Fairly valued": "f", Expensive: "e", "Very expensive": "ve" };

  function ratingChip(c, big) {
    if (!c || !c.rating) return "";
    const [l, cls, ico] = RATING[c.rating] || RATING.neutral;
    return `<span class="rating ${cls}${big ? " big" : ""}"><i aria-hidden="true">${ico}</i>${l}</span>`;
  }
  function valChip(v) { return v && v.label ? `<span class="tag val ${VAL_CLASS[v.label] || "f"}">${window.SMR.u.esc(v.label)}</span>` : ""; }
  function bubChip(b) { if (!b || !b.level) return ""; const [cls, ico] = BUB[b.level] || BUB.Low; return `<span class="tag bub ${cls}"><i aria-hidden="true">${ico}</i>Bubble risk ${window.SMR.u.esc(String(b.level).toLowerCase())}</span>`; }
  function aiChip(a) { return a && a.level && a.level !== "none" ? `<span class="tag ai">${window.SMR.u.esc(a.label)}</span>` : ""; }
  window.SMRChips = { ratingChip, valChip, bubChip, aiChip };

  // device-side Claude forecasts (made with your own API key from this phone)
  const session = {};               // ticker → forecasts made since the app opened (shown even if storage is full)
  const running = new Map();        // ticker → progress text while a paid forecast is in flight
  const finished = {};              // ticker → how the last one ended (cost included), shown when the page is redrawn
  let onScreen = null;              // the ticker whose page was drawn last
  const showing = (t) => onScreen === t && !!document.getElementById("stock-body");
  const devStored = () => window.SMR.u.LS.get("devfc", {}) || {};
  const devAll = () => {
    const all = devStored();
    for (const [t, list] of Object.entries(session)) {
      const have = new Set((all[t] || []).map((f) => f.made_at));
      all[t] = [...(all[t] || []), ...list.filter((f) => !have.has(f.made_at))];
    }
    return all;
  };
  const devFor = (t) => (devAll()[t] || []).slice().sort((a, b) => (b.made_at || "").localeCompare(a.made_at || ""));

  // the close on (or just before) a date from the stock's published, split-adjusted history
  function closeOn(px, d) {
    if (!px || !px.d || !d) return null;
    let k = -1;
    for (let i = 0; i < px.d.length; i++) if (px.d[i] <= d) k = i;
    return k >= 0 && Date.parse(d) - Date.parse(px.d[k]) <= 4 * 864e5 && px.c[k] > 0 ? px.c[k] : null;
  }
  // A forecast stored as prices on this phone: re-based for any split since (the history is split-adjusted),
  // and without horizons whose date has already passed.
  function devView(f, doc) {
    const out = JSON.parse(JSON.stringify(f)), now = doc.forecast && doc.forecast.made;
    const c0 = closeOn(doc.px, f.last_close_date), r = c0 && f.last_close ? c0 / f.last_close : 1;
    const k = isFinite(r) && r > 0.005 && r < 200 && Math.abs(Math.log(r)) > 0.02 ? r : 1;
    for (const [h, row] of Object.entries(out.horizons || {})) {
      const td = (out.target_dates || {})[h];
      if (now && td && td <= now) { delete out.horizons[h]; continue; }
      if (k !== 1) for (const key of ["low_10", "median", "high_90"]) row[key] = row[key] * k;
    }
    if (k !== 1) { (out.scenarios || []).forEach((x) => { x.price_3m = x.price_3m * k; }); out.last_close = f.last_close * k; out.split_adjusted = k; }
    return out;
  }

  function claudeMarkers(doc) {
    const out = [];
    const dev = devFor(doc.t)[0];
    const srv = doc.claude;
    const pick = dev && (!srv || (dev.made_at || "") > (srv.made_at || "")) ? { ...devView(dev, doc), device: true } : srv;
    if (!pick || !pick.horizons || !Object.keys(pick.horizons).length) return { pick: pick && pick.horizons ? pick : null, marks: [] };
    for (const [h, row] of Object.entries(pick.horizons)) {
      const date = (pick.target_dates || {})[h];
      if (date && row && row.median) out.push({ h, date, lo: row.low_10, mid: row.median, hi: row.high_90 });
    }
    return { pick, marks: out };
  }

  // Claude's 10/50/90 → seven quantiles (split-normal in log price), and the model's quantiles on a given date
  const Z = [-1.6449, -1.2816, -0.6745, 0, 0.6745, 1.2816, 1.6449], QK = ["05", "10", "25", "50", "75", "90", "95"];
  function splitNormal(row, last) {
    const l10 = Math.log(row.low_10 / last), l50 = Math.log(row.median / last), l90 = Math.log(row.high_90 / last);
    const lo = Math.max((l50 - l10) / 1.2816, 1e-4), hi = Math.max((l90 - l50) / 1.2816, 1e-4);
    return Z.map((z) => last * Math.exp(l50 + (z < 0 ? lo : hi) * z));
  }
  function fanAt(fan, date) {
    const ds = (fan && fan.dates) || [];
    if (!ds.length || !date || date <= ds[0] || date > ds[ds.length - 1]) return null;
    const k = ds.findIndex((d) => d >= date), a = Date.parse(ds[k - 1]), b = Date.parse(ds[k]), w = b === a ? 0 : (Date.parse(date) - a) / (b - a);
    const q = QK.map((key) => { const x = fan.p[key][k - 1], y = fan.p[key][k]; return x > 0 && y > 0 ? Math.exp((1 - w) * Math.log(x) + w * Math.log(y)) : null; });
    return q.every((v) => v != null) ? q : null;
  }
  function deviceBlend(pick, fc, h) {
    const row = pick.horizons[h], date = (pick.target_dates || {})[h];
    const mq = fanAt(fc.fan, date);
    if (!row || !mq || !(pick.last_close > 0)) return null;
    const cq = splitNormal(row, pick.last_close);
    const w = ((((window.SMR.data().fcAccuracy || {}).learning || {}).blend || {})[h] || {}).w_claude ?? 0.5;
    const q = {}; QK.forEach((key, i) => { q[key] = Math.exp(w * Math.log(cq[i]) + (1 - w) * Math.log(mq[i])); });
    return { q, w_claude: w, date };
  }

  let seq = 0;                         // a slow load must never paint another stock's page
  function render(view, t) {
    const my = ++seq;
    onScreen = t;
    const U = window.SMR.u, S = window.SMR.data(), q = S.quotes[t] || {};
    view.innerHTML = `<div class="pagebar"><button class="back" data-act="back">‹ Back</button>
      <button class="btn ghost" data-act="watch-t" data-t="${U.esc(t)}">${U.watch().tickers.includes(t) ? "★ Watching" : "☆ Watch"}</button></div>
      <header class="note-head"><div class="grow"><h1 class="ticker">${U.esc(t)}</h1><div class="co">${U.esc(U.niceName(q.name || ""))}</div></div>
        <div class="quote">${q.price ? `<div class="price">${U.px(q.price)}</div>` : ""}<div class="small ${q.ret_1m >= 0 ? "up" : "down"}">${q.ret_1m != null ? U.pct(q.ret_1m) + " in a month" : ""}</div></div></header>
      <div id="stock-body"><p class="muted small loading">Loading the research note…</p></div>`;
    const paint = (doc, err) => { const b = document.getElementById("stock-body"); if (my === seq && b) body(b, t, doc, err); };
    load(t).then((doc) => paint(doc)).catch((err) => paint(null, err));
  }

  // one broken field must not cost the whole page: each section draws on its own
  function safe(fn, ...a) {
    try { return fn(...a); } catch (e) {
      window.dispatchEvent(new ErrorEvent("error", { message: `stock page section: ${e.message}`, filename: "stock.js" }));
      return `<section class="sec"><div class="panel"><p class="small muted">This part couldn't be shown (${window.SMR.u.esc(e.message)}).</p></div></section>`;
    }
  }

  function body(box, t, doc, err) {
    const U = window.SMR.u, S = window.SMR.data();
    const evs = S.feed.filter((e) => e.ticker === t);
    if (!doc) {
      const off = err && (err.offline || !navigator.onLine);
      box.innerHTML = `<div class="notice">${off ? "You're offline and this stock's research wasn't saved for offline use yet. It loads next time you're online." : err ? `Couldn't load this stock's research (${U.esc(err.message)}).` : "No forecast or valuation for this ticker yet. The app builds them for stocks in the feed once it has about six months of their prices."}</div>
        ${U.riskPanel ? `<section class="sec">${U.riskPanel(t)}</section>` : ""}
        ${signals(evs)}`;
      return;
    }
    const fc = doc.forecast && doc.forecast.horizons && Object.keys(doc.forecast.horizons).length ? doc.forecast : null, conv = doc.conviction;
    if (!state.range || (fc && !fc.horizons[state.range])) state.range = fc && !fc.horizons["3m"] ? Object.keys(fc.horizons)[0] : "3m";
    const { pick, marks } = claudeMarkers(doc);
    box.innerHTML = `
      ${safe(chipsRow, doc, conv)}
      ${fc ? safe(forecastSection, doc, fc, pick) : `<section class="sec"><h2>Forecast</h2><div class="panel"><p class="muted small">${U.esc(doc.forecast_note || "Not enough price history for a forecast yet (needs about six months).")}</p></div></section>`}
      ${safe(claudeSection, doc, pick)}
      ${safe(convictionSection, conv)}
      ${safe(valuationSection, doc)}
      ${safe(bubbleSection, doc)}
      ${safe(newsSection, doc)}
      ${safe(signals, evs)}
      <section class="sec">${safe(U.riskPanel, t)}</section>
      ${doc._saved ? `<p class="small muted">The network was slow or offline, so this is the copy of this page saved on this phone (from ${U.esc(U.ago(doc.generated_at))}).</p>` : ""}
      <p class="disc">Forecasts are probabilities from a model and from Claude, graded in the open on the Records tab. Ratings are research, not personal financial advice.</p>`;
    if (fc) { try { drawForecast(box, doc, fc, marks); } catch (e) { const f = box.querySelector("#fan"); if (f) f.innerHTML = `<p class="small muted">The chart couldn't be drawn.</p>`; } }
    box.querySelectorAll("[data-range]").forEach((b) => b.addEventListener("click", () => {
      state.range = b.dataset.range; body(box, t, doc);
      const nb = box.querySelector(`[data-range="${state.range}"]`); if (nb) nb.focus({ preventScroll: true });   // keep the keyboard's place
    }));
    const ask = box.querySelector('[data-act="device-forecast"]'), st = box.querySelector("#devfc-status");
    if (ask) ask.addEventListener("click", () => deviceForecast(t, doc));
    if (ask && running.has(t)) { ask.disabled = true; if (st) st.textContent = running.get(t); }
    else if (st && finished[t]) st.textContent = finished[t];
  }
  function chipsRow(doc, conv) {
    const U = window.SMR.u, fl = doc.news && doc.news.flags && doc.news.flags[0];
    return `<div class="chips-row">${ratingChip(conv, true)}${valChip(doc.valuation)}${bubChip(doc.bubble)}${aiChip(doc.ai)}${fl ? `<span class="tag flag"><i aria-hidden="true">⚑</i>${U.esc(fl.label)}</span>` : ""}</div>`;
  }

  // ------------------------------------------------------------------ forecast
  function forecastSection(doc, fc, pick) {
    const U = window.SMR.u, h = fc.horizons[state.range], name = (HZ.find((x) => x[0] === state.range) || [])[1];
    const A = (window.SMR.data().fcAccuracy || {}).metrics || {};
    const tr = ((A.by || {}).quant || {})[state.range], btr = ((A.by || {}).quant_bt || {})[state.range];
    const live = tr && tr.all && tr.all.n ? tr.all : null, bt = btr && btr.all && btr.all.n ? btr.all : null;
    // live grades once there are enough, else the backtest; with neither past 100, whichever has more (20+)
    const L0 = live && { ...live, from: "live" }, B0 = bt && { ...bt, from: "backtest" };
    const measured = L0 && L0.n >= 100 ? L0 : B0 && B0.n >= 100 ? B0 : [L0, B0].filter((m) => m && m.n >= 20).sort((a, b) => b.n - a.n)[0] || null;
    // "too narrow / too wide" only on solid evidence: an interval resting on 3+ independent periods that excludes
    // 80%, quoted with its own rate (it comes from the kept records, which can differ from the all-time total)
    const iv0 = measured && measured.cov80_ci;
    const iv = iv0 && typeof iv0 === "object" && !Array.isArray(iv0) && iv0.periods >= 3 && iv0.n >= 20 ? iv0 : null;
    const covOff = !iv ? 0 : iv.hi < 0.8 ? -1 : iv.lo > 0.8 ? 1 : 0;
    const touch = h.touch || {};
    const evs = (fc.events || []).filter((e) => e.date <= h.date);
    const big = evs.filter((e) => e.risk_share_3m != null).sort((a, b) => (Math.round(b.risk_share_3m * 100) - Math.round(a.risk_share_3m * 100)) || a.date.localeCompare(b.date))[0];
    const lt = fc.low_timing_3m || {};
    const pct0 = (v) => (v == null ? "—" : `${Math.round(v * 100)}%`);
    const row = pick && pick.horizons && pick.horizons[state.range];
    const bl = pick && pick.device ? (row ? deviceBlend(pick, fc, state.range) : null) : doc.blend && doc.blend[state.range];
    const cDate = row && (pick.target_dates || {})[state.range];
    return `<section class="sec"><div class="sec-h"><h2>Forecast</h2><span class="small muted">from the ${U.dateFmt(fc.made)} close of ${U.px(fc.anchor)}</span></div>
      <div class="panel chart-panel">
        <div class="seg compact" role="group" aria-label="Forecast horizon">${HZ.filter(([k]) => fc.horizons[k]).map(([k, l]) => `<button data-range="${k}" aria-pressed="${state.range === k}">${k.toUpperCase()}<span class="sr-only"> (${l})</span></button>`).join("")}</div>
        <div id="fan" class="fan-box"></div>
        <div class="viz-legend"><span><i class="viz-key line" style="--k:var(--ink)"></i>Price</span><span><i class="viz-key band" style="--k:var(--series-1)"></i>Model range (50 / 80 / 90%)</span>${pick ? `<span><i class="viz-key dot" style="--k:var(--series-2)"></i>Claude (10–90%)</span>` : ""}<span class="muted">Ticks: E earnings, Fed, CPI</span></div>
      </div>
      <div class="panel">
        <div class="hz-title">In ${U.esc(name)}, by ${U.dateFmt(h.date)}</div>
        <div class="kpis">
          <div class="kpi"><span>Likely range (80%)</span><b>${U.pxc(h.q["10"])}–${U.pxc(h.q["90"])}</b>${measured && measured.cov80 != null ? `<small class="kpi-sub">${measured.from === "live" ? "So far" : "Backtest"}: held ${pct0(measured.cov80)} of ${U.num(measured.n)}</small>` : ""}</div>
          <div class="kpi"><span>Middle outcome</span><b>${U.pxc(h.q["50"])}</b></div>
          <div class="kpi"><span>Chance it ends higher</span><b>${pct0(h.p_up)}</b></div>
          <div class="kpi"><span>Wider range (90%)</span><b>${U.pxc(h.q["05"])}–${U.pxc(h.q["95"])}</b></div>
        </div>
        ${covOff ? `<p class="notice small">${covOff < 0
          ? `Read these ranges as too narrow for now: at this horizon, the model's 80% ranges held only <b>${pct0(iv.p)}</b> of ${U.num(iv.n)} recent ${measured.from === "live" ? "graded forecasts" : "backtest forecasts"} (likely between ${pct0(iv.lo)} and ${pct0(iv.hi)}; target 80%). Calibration widens them as grades come in.`
          : `These ranges have been wider than needed: they held <b>${pct0(iv.p)}</b> of ${U.num(iv.n)} recent ${measured.from === "live" ? "graded forecasts" : "backtest forecasts"} (likely between ${pct0(iv.lo)} and ${pct0(iv.hi)}; target 80%).`}</p>` : ""}
        <h3>Peaks and dips along the way</h3>
        <div class="peaks">
          <div><span>Typical high</span><b>${U.pct(h.hi_med * 100, 1)}</b></div><div><span>Typical low</span><b>${U.pct(h.lo_med * 100, 1)}</b></div>
          <div><span>Touches +10% on the way</span><b>${pct0(touch["+10"])}</b></div><div><span>Touches −10% on the way</span><b>${pct0(touch["-10"])}</b></div>
          <div><span>Typical worst drop from a peak</span><b>${U.pct(-h.mdd_med * 100, 1)}</b></div><div><span>20%+ drop from a peak</span><b>${pct0(h.p_dd20)}</b></div>
        </div>
        ${big && big.risk_share_3m > 0.04 ? `<p class="riskday">Riskiest day: <b>${U.esc(big.label.replace(" (estimated date)", ""))}, ${U.dateFmt(big.date)}</b>${big.estimated ? " (date estimated)" : ""}. It carries ${pct0(big.risk_share_3m)} of the next three months' risk${big.p_low_near != null ? `, and the 3-month low falls within a day or two of it ${pct0(big.p_low_near)} of the time` : ""}.</p>` : ""}
        ${state.range === "3m" && lt.now != null ? timingBar(lt) : ""}
        <p class="small muted">A 40% fall within a year: ${pct0((fc.crash || {}).p_dd40_1y)}, judged from its volatility alone. Nobody can name the day of a peak or a dip; for prices that move like a random walk the high is about as likely near either end of the window, so the app shows probabilities and risk windows instead.</p>
        ${row || bl ? `<h3>Model, Claude and combined</h3>
        <div class="scroll-x"><table class="cmp"><tr><th></th><th>Middle</th><th>80% range</th><th>Higher</th></tr>
          <tr><td>Model<div class="small muted">${U.dateFmt(h.date).replace(/, \d{4}$/, "")}</div></td><td>${U.pxc(h.q["50"])}</td><td class="nw">${U.pxc(h.q["10"])}–${U.pxc(h.q["90"])}</td><td>${pct0(h.p_up)}</td></tr>
          ${row ? `<tr><td>Claude${pick.device ? " (this phone)" : ""}${cDate ? `<div class="small muted">${U.dateFmt(cDate).replace(/, \d{4}$/, "")}</div>` : ""}</td><td>${U.pxc(row.median)}</td><td class="nw">${U.pxc(row.low_10)}–${U.pxc(row.high_90)}</td><td>${pct0(row.p_up)}</td></tr>` : ""}
          ${bl ? `<tr><td>Combined${bl.date ? `<div class="small muted">${U.dateFmt(bl.date).replace(/, \d{4}$/, "")}</div>` : ""}</td><td>${U.pxc(bl.q["50"])}</td><td class="nw">${U.pxc(bl.q["10"])}–${U.pxc(bl.q["90"])}</td><td></td></tr>` : ""}</table></div>
        ${bl ? `<p class="small muted">Combined: Claude's forecast and the model's for the same date, Claude weighted ${Math.round(bl.w_claude * 100)}% by each one's measured skill (equal until both are graded).</p>` : ""}` : ""}
        <h3>How reliable is this?</h3>
        <p class="small">${live ? `Graded so far at this horizon: ${U.num(live.n)} forecasts. The 80% range held <b>${pct0(live.cov80)}</b> of outcomes (target 80%); direction was right ${pct0(live.hit)} of the time versus ${pct0(live.up)} for "always say up".`
          : bt ? `Not graded live yet at this horizon. In the backtest (the same model re-run on past dates, ${U.num(bt.n)} forecasts at this horizon), its 80% ranges held <b>${pct0(bt.cov80)}</b> of outcomes (target 80%); direction was right ${pct0(bt.hit)} of the time versus ${pct0(bt.up)} for "always say up".`
          : "Not graded yet at this horizon: forecasts are written down daily and graded when their date arrives (see Records)."}
          ${fc.calib && fc.calib[state.range] && fc.calib[state.range].scale !== 1 ? ` Ranges here are ${fc.calib[state.range].scale > 1 ? "widened" : "narrowed"} ×${U.num(fc.calib[state.range].scale, 2)} by what grading has taught it.` : ""}</p>
        <details class="more"><summary>Method, assumptions and the table view</summary>
          <div class="scroll-x"><table class="cmp"><tr><th>Horizon</th><th>5%</th><th>10%</th><th>50%</th><th>90%</th><th>95%</th><th>Higher</th></tr>
            ${HZ.filter(([k]) => fc.horizons[k]).map(([k, l]) => { const z = fc.horizons[k]; return `<tr><td>${l}</td><td>${U.pxc(z.q["05"])}</td><td>${U.pxc(z.q["10"])}</td><td>${U.pxc(z.q["50"])}</td><td>${U.pxc(z.q["90"])}</td><td>${U.pxc(z.q["95"])}</td><td>${pct0(z.p_up)}</td></tr>`; }).join("")}</table></div>
          <p class="small muted">${U.num(fc.paths)} simulated paths that resample this stock's own past daily shocks (keeping its fat tails), scaled by its current volatility of ${U.pct(fc.vol_now * 100, 0, false)} a year (long-run ${U.pct(fc.vol_lr * 100, 0, false)}), with extra risk on earnings days (${U.num(fc.earn_mult, 1)}× a normal day${fc.earn_samples ? `, from its last ${U.num(fc.earn_samples)} reports` : ", a typical value"}), Fed decisions and CPI releases. Drift ${U.pct(fc.drift_ann * 100, 1)} a year: cash rate plus beta ${U.num(fc.beta, 2)} × a 5% equity premium${fc.div_yield ? `, minus its ${U.pct(fc.div_yield * 100, 1, false)} dividend yield (prices drop by the dividend)` : ""}${fc.tilt && fc.tilt.ret ? `, plus a learned nudge of ${U.pct(fc.tilt.ret * 100, 1)} over the next ${U.num(fc.tilt.days)} sessions${(fc.tilt.parts || []).filter((x) => x && x.label && x.h_ret).length ? ` (mainly ${U.esc(fc.tilt.parts.filter((x) => x && x.label && x.h_ret).slice(0, 2).map((x) => x.label.charAt(0).toLowerCase() + x.label.slice(1)).join(" and "))}; see Records → What it has learned)` : ""}` : ""}. The middle outcome sits below the average one for volatile stocks: that's how compounding works, not pessimism.</p></details>
      </div></section>`;
  }

  function timingBar(lt) {
    const parts = [["now", "Never below today", "t0"], ["early", "First month", "t1"], ["middle", "Middle month", "t2"], ["late", "Last month", "t3"]];
    const pc = (v) => `${Math.round(v * 100)}%`;
    return `<h3>When the 3-month low tends to come</h3>
      <div class="timing" role="img" aria-label="${parts.map(([k, l]) => `${l} ${pc(lt[k] || 0)}`).join(", ")}">${parts.map(([k, , c]) => (+lt[k] > 0 ? `<span class="${c}" style="flex:${Number(lt[k])}">${lt[k] >= 0.12 ? pc(lt[k]) : ""}</span>` : "")).join("")}</div>
      <div class="timing-legend">${parts.map(([k, l, c]) => `<span><i class="${c}"></i>${l} ${pc(lt[k] || 0)}</span>`).join("")}</div>`;
  }

  function drawForecast(box, doc, fc, marks) {
    const fanBox = box.querySelector("#fan");
    if (!fanBox || !window.SMRCharts) return;
    const h = fc.horizons[state.range];
    window.SMRCharts.keep(fanBox, () => window.SMRCharts.fan(fanBox, {
      history: doc.px, fan: fc.fan, claude: marks, events: fc.events, range: state.range, horizonDate: h.date,
      labels: { today: window.SMR.u.dateFmt(fc.made).replace(/, \d{4}$/, "") },
      aria: `${doc.t}: price history and ${state.range} forecast; 80% range ${h.q["10"]} to ${h.q["90"]}`,
    }));
  }

  // ------------------------------------------------------------------ Claude
  function claudeSection(doc, pick) {
    const U = window.SMR.u;
    const hasKey = window.SMRChat && window.SMRChat.hasKey && window.SMRChat.hasKey();
    const btn = hasKey ? `<button class="btn" data-act="device-forecast">Ask Claude for a fresh forecast</button>`
      : `<a class="btn" href="#/ask">Add your API key to ask Claude here</a>`;
    if (!pick) {
      const has = ((window.SMR.data().health || {}).integrations || {}).claude_api;
      return `<section class="sec"><div class="sec-h"><h2>Claude's forecast</h2></div><div class="panel"><p class="small muted">${has ? "Claude forecasts a few stocks a day (your watchlist first). None for this one in the last 45 days." : "With an Anthropic API key, Claude researches the news and forecasts stocks; its calls are graded like the model's."}</p><div class="btns">${btn}</div><div id="devfc-status" class="small muted"></div></div></section>`;
    }
    const sc = pick.scenarios || [];
    const dr = pick.dip_risk || {}, pk = pick.peak_view || {};
    return `<section class="sec"><div class="sec-h"><h2>Claude's forecast</h2><span class="small muted">${pick.device ? "made on this phone" : U.esc(window.SMR.u.modelName(pick.model))}, ${U.esc(U.ago(pick.made_at))}</span></div>
      <div class="panel ai">
        <p class="lead">${U.esc(pick.thesis || "")}</p>
        ${sc.length ? `<h3>Scenarios in about 3 months</h3><div class="scen">${sc.map((s) => `<div class="sc ${U.esc(s.name)}"><span>${s.name === "bull" ? "Bull" : s.name === "bear" ? "Bear" : "Base"}, ${Math.round((s.probability || 0) * 100)}%</span><b>${U.pxc(s.price_3m)}</b><p class="small">${U.esc(s.drivers)}</p></div>`).join("")}</div>` : ""}
        <table class="facts">
          ${dr.window ? `<tr><td>Dip risk</td><td>${U.esc(dr.window)}${dr.p_dip10_3m != null ? `; ${Math.round(dr.p_dip10_3m * 100)}% chance of a 10%+ dip within 3 months` : ""}<div class="small muted">${U.esc(dr.why || "")}</div></td></tr>` : ""}
          ${pk.window ? `<tr><td>Upside window</td><td>${U.esc(pk.window)}<div class="small muted">${U.esc(pk.why || "")}</div></td></tr>` : ""}
          ${pick.path_notes ? `<tr><td>Path</td><td>${U.esc(pick.path_notes)}</td></tr>` : ""}
          ${(pick.key_dates || []).length ? `<tr><td>Key dates</td><td>${pick.key_dates.map((k) => `<div><b>${U.esc(/^\d{4}-\d{2}-\d{2}$/.test(k.date || "") ? U.dateFmt(k.date) : k.date)}</b> ${U.esc(k.event)}${k.impact ? `<div class="small muted">${U.esc(k.impact)}</div>` : ""}</div>`).join("")}</td></tr>` : ""}
          ${(pick.risks || []).length ? `<tr><td>Risks</td><td>${pick.risks.map((r) => `<div>${U.esc(r)}</div>`).join("")}</td></tr>` : ""}
          ${pick.what_would_change_my_mind ? `<tr><td>Would change its mind</td><td>${U.esc(pick.what_would_change_my_mind)}</td></tr>` : ""}
          ${pick.stance ? `<tr><td>Versus the S&amp;P 500</td><td>${window.SMR.u.stanceChip(pick)} <span class="small muted">over ~3 months, ${U.esc(pick.confidence || "low")} confidence</span></td></tr>` : ""}
          ${(pick.lessons_applied || []).length ? `<tr><td>Lessons it applied</td><td class="small">${pick.lessons_applied.map((l) => `<div>${U.esc(l)}</div>`).join("")}</td></tr>` : ""}
        </table>
        ${U.sourcesList(pick.sources)}
        <div class="btns">${btn}</div><div id="devfc-status" class="small muted"></div>
        <p class="small muted">Graded from the close of ${U.dateFmt(pick.ref_date)}, the first session after it was written, so after-hours news can't flatter it.</p>
      </div></section>`;
  }

  function deviceForecast(t, doc) {
    if (running.has(t) || !window.SMRChat || !window.SMRChat.forecast) return;
    const U = window.SMR.u;
    const say = (msg) => {
      running.set(t, msg);
      const st = showing(t) && document.querySelector("#stock-body #devfc-status");
      if (st) st.textContent = msg;
    };
    delete finished[t];
    say("Claude is researching and forecasting… this can take a minute or two. You can leave this page; it carries on.");
    const b0 = document.querySelector('#stock-body [data-act="device-forecast"]'); if (b0) b0.disabled = true;
    window.SMRChat.forecast(t, doc, say).then((res) => {
      const stored = devStored();
      stored[t] = [res, ...(stored[t] || [])].slice(0, 6);
      const saved = U.LS.set("devfc", stored);
      if (!saved) session[t] = [res, ...(session[t] || [])];
      finished[t] = `Done${res.cost != null ? ` (about $${res.cost.toFixed(2)}, billed to your key)` : ""}. ` + (saved
        ? "It will be graded on this phone when its dates arrive."
        : "This phone's storage is full or blocked, so it's kept only until the app closes and won't be graded.");
      if (!saved) U.toast("Couldn't save the forecast on this phone");
      else if (!showing(t)) U.toast(`Claude's forecast for ${t} is ready`);
    }).catch((e) => {
      finished[t] = `Couldn't get a forecast: ${e.message}` + (e.cost > 0 ? ` Up to about $${e.cost.toFixed(2)} may have been billed to your key for the attempt.` : "");
      if (!showing(t)) U.toast(`Claude's forecast for ${t} failed: open the stock to see why`);
    }).finally(() => {
      running.delete(t);
      const box = document.getElementById("stock-body");
      if (showing(t)) {                      // redraw with the new forecast
        const hit = cache.get(t), d = (hit && hit.doc) || doc, had = box.contains(document.activeElement);
        body(box, t, d);
        const nb = had && box.querySelector('[data-act="device-forecast"]'); if (nb) nb.focus({ preventScroll: true });
      }
    });
  }

  // ------------------------------------------------------------------ conviction
  function convictionSection(c) {
    const U = window.SMR.u;
    if (!c) return "";
    const ico = (ok) => (ok === true ? `<i class="ck ok" aria-label="passed">✓</i>` : ok === false ? `<i class="ck no" aria-label="not passed">✕</i>` : `<i class="ck na" aria-label="unknown">–</i>`);
    const rec = (window.SMR.data().fcAccuracy || {}).conviction_record;
    const record = rec && rec.n ? `So far ${Math.round((rec.beat_rate || 0) * 100)}% of graded Top conviction calls beat the S&amp;P 500 (${U.num(rec.n)} graded).` : "This rating has no graded track record yet: the first grades arrive about three months after the first call.";
    return `<section class="sec"><div class="sec-h"><h2>Rating</h2>${ratingChip(c)}</div><div class="panel">
      <p class="small">${c.rating === "top" ? `Every check the app can measure lines up. ${record}` : `"Top conviction" needs all ${U.esc(c.of)} checks; this stock passes ${U.esc(c.passed)}.`}${c.why && c.why.length ? ` ${U.esc(c.why.join("; "))}.` : ""}</p>
      <ul class="checklist">${(c.checks || []).map((k) => `<li>${ico(k.ok)}<div><b>${U.esc(k.label)}</b><div class="small muted">${U.esc(k.detail)}</div></div></li>`).join("")}</ul>
      <p class="small muted">Every Top conviction call is graded against the S&amp;P 500 after 3 months; its record is on the Records tab. Research, not personal advice.</p></div></section>`;
  }

  // ------------------------------------------------------------------ valuation
  function valuationSection(doc) {
    const U = window.SMR.u, v = doc.valuation;
    if (!v) return `<section class="sec"><h2>Valuation</h2><div class="panel"><p class="small muted">No SEC financial data for this ticker yet (funds, trusts, foreign filers reporting in other currencies, or not fetched yet).</p></div></section>`;
    if (v.note && !v.label && v.mcap == null) return `<section class="sec"><h2>Valuation</h2><div class="panel"><p class="small muted">Not valued: ${U.esc(v.note)}.</p></div></section>`;
    const p = v.rich_pctl;
    const steps = ["Very cheap", "Cheap", "Fairly valued", "Expensive", "Very expensive"];
    const scale = p != null ? `<div class="divscale" role="img" aria-label="Valuation ${U.esc(v.label)}: pricier than ${Math.round(p)}% of companies relative to fundamentals">
        ${steps.map((s) => `<span class="${VAL_CLASS[s]}${s === v.label ? " on" : ""}"></span>`).join("")}<i style="left:${Math.max(1, Math.min(99, p))}%"></i></div>
        <div class="divlabels small muted"><span>Cheap</span><span>Fair</span><span>Expensive</span></div>` : "";
    const m = (x) => (x == null ? null : U.num(x, x < 10 ? 1 : 0) + "×");
    const pc = (x, d = 0) => (x == null ? null : U.pct(x * 100, d));
    const dcf = v.dcf;
    const cf = doc.claude && doc.claude.valuation_view;
    return `<section class="sec"><div class="sec-h"><h2>Valuation</h2>${valChip(v)}</div><div class="panel">
      ${p != null ? `<p class="small">Pricier than <b>${Math.round(p)}%</b> of companies, judged by ${U.esc(v.basis || "its multiples")}.${v.fair_ev_s && v.ev_s ? ` It trades at ${m(v.ev_s)} sales; companies with similar growth and margins average about ${m(v.fair_ev_s)}.` : ""}</p>${scale}`
        : `<p class="small muted">The verdict appears once the app has financials for enough companies to compare against (it adds about 150 every half hour).</p>`}
      <table class="facts">
        ${row("Market value", U.money(v.mcap))}${row("Price / earnings", v.pe ? m(v.pe) : v.ni != null && v.ni <= 0 ? "Loss-making" : null)}
        ${row("Price / sales", m(v.ps))}${row("Enterprise value / sales", m(v.ev_s))}${row("EV / operating profit", m(v.ev_ebit))}
        ${row("Free-cash-flow yield", pc(v.fcf_yield, 1))}${row("Revenue growth (12 months)", pc(v.growth, 0))}
        ${row("Gross / operating margin", v.gross_margin != null || v.op_margin != null ? `${pc(v.gross_margin) || "—"} / ${pc(v.op_margin) || "—"}` : null)}
        ${row("Stock pay, share of revenue", pc(v.sbc_pct, 1))}${row("Share count change (12 months)", pc(v.dilution, 1))}
        ${row("Net cash (debt)", v.net_cash != null ? U.money(v.net_cash) : null)}${row("Price / book", m(v.pb))}
      </table>
      ${dcf ? `<p class="small">${dcf.growth != null ? `The price assumes free cash flow grows about <b>${U.pct(dcf.growth * 100, 0)} a year for 10 years</b>, then ${Math.round(dcf.g_term * 100)}% forever, at a ${Math.round(dcf.r * 100)}% required return (reverse DCF).` : dcf.above ? "The price assumes free cash flow grows more than 100% a year for a decade: priced for something extraordinary." : "The price implies shrinking cash flows: priced for decline."}</p>` : ""}
      ${cf && cf.reasoning ? `<p class="small"><b>Claude's view${cf.label ? `: ${U.esc(cf.label.toLowerCase())}` : ""}.</b> ${U.esc(cf.reasoning)}</p>` : ""}
      <p class="small muted">Trailing-12-month figures from the company's SEC filings (period to ${U.dateFmt(v.period_end)}, ${U.esc(v.method || "")})${v.price_date ? `, at the ${U.dateFmt(v.price_date)} close` : ""}. Valuation is a poor timing tool: expensive stocks can stay expensive for years.</p>
    </div></section>`;
  }
  const row = (k, v) => (v == null || v === "—" ? "" : `<tr><td>${window.SMR.u.esc(k)}</td><td>${v}</td></tr>`);

  // ------------------------------------------------------------------ bubble
  function bubbleSection(doc) {
    const U = window.SMR.u, b = doc.bubble, ai = doc.ai, th = (window.SMR.data().market || {}).ai_bubble;
    if (!b) return "";
    const lv = ["Low", "Moderate", "Elevated", "High"];
    const i = lv.indexOf(b.level);
    const cv = doc.claude && doc.claude.bubble_view;
    return `<section class="sec"><div class="sec-h"><h2>Bubble risk</h2>${bubChip(b)}</div><div class="panel">
      <div class="meter4" role="img" aria-label="Bubble risk ${U.esc(b.level)}">${lv.map((l, k) => `<span class="${k <= i ? "on l" + k : ""}"></span>`).join("")}</div>
      <div class="divlabels small muted"><span>Low</span><span>High</span></div>
      ${b.signs && b.signs.length ? `<ul class="signs">${b.signs.map((s) => `<li class="sg-${U.esc(s.kind)}"><b>${U.esc(s.title)}</b><span class="small muted">${U.esc(s.detail)}</span></li>`).join("")}</ul>` : `<p class="small">None of the warning signs are present.</p>`}
      ${b.gsy_crash_2y != null ? `<p class="small">Historical base rate: US industries with a run-up this large fell 40%+ within two years about <b>${Math.round(b.gsy_crash_2y * 100)}%</b> of the time (Greenwood, Shleifer &amp; You, 2019). Industry data, a different unit than one stock; and run-ups alone didn't predict lower average returns.</p>` : ""}
      ${ai ? `<h3>AI exposure: ${U.esc(ai.label)}</h3>${ai.why && ai.why.length ? `<ul class="list small">${ai.why.map((w) => `<li>${U.esc(w)}</li>`).join("")}</ul>` : ""}` : ""}
      ${th && ai && ai.level !== "none" ? `<p class="small">AI basket right now: <b>${U.esc(th.level)}</b>, ${U.esc(th.phase.toLowerCase())}; ${th.stats && th.stats.net_2y != null ? `${U.pct(th.stats.net_2y * 100, 0)} vs the S&amp;P 500 over two years` : ""}. <a href="#/ideas">See the gauge</a></p>` : ""}
      ${cv && cv.reasoning ? `<p class="small"><b>Claude's view${cv.level ? `: ${U.esc(cv.level.toLowerCase())}` : ""}.</b> ${U.esc(cv.reasoning)}</p>` : ""}
      <p class="small muted">A checklist of published warning signs, not a fitted model; the app checks it against what actually happened (Records → Bubble check).</p></div></section>`;
  }

  // ------------------------------------------------------------------ news
  function newsSection(doc) {
    const U = window.SMR.u, n = doc.news || {};
    const fl = n.filings || [], hl = n.headlines || [], cl = n.claude || [];
    if (!fl.length && !hl.length && !cl.length) return `<section class="sec"><h2>News</h2><div class="panel"><p class="small muted">No recent filings or headlines on file yet.</p></div></section>`;
    const link = (u, s) => (U.safeUrl(u) ? `<a href="${U.esc(U.safeUrl(u))}" target="_blank" rel="noopener noreferrer">${s}</a>` : s);
    return `<section class="sec"><h2>News</h2>
      ${cl.length ? `<div class="panel ai"><h3>Claude's read of the news</h3><ul class="news">${cl.map((x) => `<li><span class="nd">${U.esc(/^\d{4}-\d{2}-\d{2}$/.test(x.date || "") ? U.dateFmt(x.date) : x.date)}</span><div>${link(x.url, `<b>${U.esc(x.headline)}</b>`)}<div class="small muted">${U.esc(x.why_it_matters)}</div></div></li>`).join("")}</ul></div>` : ""}
      ${fl.length ? `<div class="panel"><h3>Company filings (8-K)</h3><ul class="news">${fl.slice(0, 8).map((f) => { const its = f.items || []; return `<li class="${U.esc(f.sev || "")}"><span class="nd">${U.dateFmt(f.date)}</span><div>${link(f.url, `<b>${f.sev === "red" ? "⚑ " : ""}${U.esc(f.title)}</b>`)}${its.length > 1 ? `<div class="small muted">${its.slice(1).map((i) => U.esc(i.label)).join("; ")}</div>` : ""}</div></li>`; }).join("")}</ul>
        <p class="small muted">Official announcements filed with the SEC, item by item.</p></div>` : ""}
      ${hl.length ? `<div class="panel"><h3>Headlines</h3><ul class="news">${hl.map((h) => `<li><span class="nd">${U.esc(U.ago(h.seen))}</span><div>${link(h.url, U.esc(h.title))}<div class="small muted">${U.esc(h.domain || "")}</div></div></li>`).join("")}</ul>
        <p class="small muted">Headlines via the <a href="https://www.gdeltproject.org/" target="_blank" rel="noopener noreferrer">GDELT Project</a>; reputable outlets first. Read the source before acting.</p></div>` : ""}
    </section>`;
  }

  function signals(evs) {
    const U = window.SMR.u;
    return evs.length ? `<section class="sec"><h2>Smart-money moves</h2><div class="rows">${evs.slice(0, 12).map(U.card).join("")}</div></section>` : "";
  }

  // ------------------------------------------------------------------ the list of every stock with a forecast
  const list = { q: "", sort: "ticker", data: null, version: null, shown: null, limit: 150, html: new Map() };
  async function loadList() {
    const S = window.SMR.data(), v = S.version || Date.now();
    if (list.data && list.version === v) return list.data;
    const r = await fetch(`data/forecasts.json?v=${v}`, { cache: "no-store" });
    if (r.status === 404) return { stocks: [], n: 0 };
    if (!r.ok) {
      let off = false;
      if (r.status === 503) { try { off = !!(await r.json()).offline; } catch { /* a server error page */ } }
      const e = new Error(off ? "offline, not saved" : `HTTP ${r.status}`); e.offline = off; throw e;
    }
    const d = await r.json();
    if (r.headers.get("X-SMR-From-Cache")) { d._saved = true; return d; }     // a saved copy: fetched again next time
    list.data = d; list.version = v;
    return d;
  }
  const chg = (row, k) => (row[k] && row[k].mid && row.close ? row[k].mid / row.close - 1 : null);
  const SORTS = { ticker: ["Ticker", (a, b) => a.t.localeCompare(b.t)],
    up3: ["3-month outlook", (a, b) => (chg(b, "m3") ?? -9) - (chg(a, "m3") ?? -9)],
    pup: ["Chance up (1 month)", (a, b) => ((b.m1 || {}).up ?? -1) - ((a.m1 || {}).up ?? -1)],
    risk: ["Widest range", (a, b) => (((b.m1 || {}).hi / (b.m1 || {}).lo) || 0) - (((a.m1 || {}).hi / (a.m1 || {}).lo) || 0)] };
  function renderList(view) {
    const my = ++seq;
    onScreen = null;
    const S = window.SMR.data(), v = S.version || Date.now();
    // coming Back to the list from a stock: the same place in it (only then; a new visit starts at the top)
    const st = history.state || {}, back = typeof st.fcScroll === "number" ? st.fcScroll : null;
    if (back != null) { try { history.replaceState({ ...st, fcScroll: null }, ""); } catch { /* not allowed here */ } }
    view.innerHTML = `<div class="pagebar"><button class="back" data-act="back">‹ Back</button></div>
      <h1 class="page-title">All forecasts</h1><div id="fc-list"></div>`;
    const place = () => { if (back != null) requestAnimationFrame(() => window.scrollTo(0, back)); };
    if (list.data) drawList(list.data);           // what's here already, at once (a newer version replaces it below)
    else document.getElementById("fc-list").innerHTML = `<p class="muted small loading">Loading the list…</p>`;
    if (list.data && list.version === v) { place(); return; }
    loadList().then((d) => {
      if (my !== seq) return;
      const y = window.scrollY;
      drawList(d);
      if (back != null) window.scrollTo(0, back); else window.scrollTo(0, y);
    }).catch((err) => {
      const box = document.getElementById("fc-list");
      if (my === seq && box && !list.data) box.innerHTML = `<div class="notice">${err.offline || !navigator.onLine ? "You're offline and the list wasn't saved for offline use yet." : `Couldn't load the list (${window.SMR.u.esc(err.message)}).`}</div>`;
    });
  }
  // The search box and sort menu are drawn once; typing only redraws the rows (so phone keyboards that compose
  // words aren't interrupted), a moment after the last key.
  function drawList(d) {
    const U = window.SMR.u, box = document.getElementById("fc-list");
    if (!box) return;
    if (list.shown !== d) { list.shown = d; list.html = new Map(); }      // the rows drawn for this copy of the list
    const all = (d && d.stocks) || [];
    if (!all.length) {
      box.innerHTML = `<div class="panel"><p class="small">${d && d.enabled === false
        ? "Price forecasts are switched off in this app's settings (forecast.enabled in config.yaml), so there's no list."
        : "No forecasts yet. The app builds one for every stock in the feed once it has about six months of that stock's prices; the first ones appear within a day of setup (More → Data shows the price history filling in)."}</p></div>`;
      return;
    }
    if (!box.querySelector("#fc-q")) {
      box.innerHTML = `<p class="page-sub" id="fc-sub"></p>
        <div class="toolbar"><input class="search" id="fc-q" type="search" placeholder="Ticker or company" aria-label="Search the forecasts by ticker or company" value="${U.esc(list.q)}" autocomplete="off">
          <select class="search" id="fc-sort" aria-label="Sort by">${Object.entries(SORTS).map(([k, [l]]) => `<option value="${k}"${list.sort === k ? " selected" : ""}>${l}</option>`).join("")}</select></div>
        <div class="fc-list" id="fc-rows"></div>
        <p class="disc">The middle outcome is as likely to be beaten as missed; the 80% range should hold the price about 80% of the time (graded on the Records tab). Research, not personal financial advice.</p>`;
      const qi = box.querySelector("#fc-q");
      let timer = null;
      const later = () => { clearTimeout(timer); timer = setTimeout(() => { if (list.q !== qi.value) { list.q = qi.value; list.limit = 150; drawRows(); } }, 150); };
      qi.addEventListener("input", (e) => { if (!e.isComposing) later(); });
      qi.addEventListener("compositionend", later);
      box.querySelector("#fc-sort").addEventListener("change", (e) => { list.sort = e.target.value in SORTS ? e.target.value : "ticker"; drawRows(); });
      box.querySelector("#fc-rows").addEventListener("click", (e) => {
        if (e.target.closest("a.fc-row")) { try { history.replaceState({ ...(history.state || {}), fcScroll: window.scrollY }, ""); } catch { /* not allowed here */ } }
        if (e.target.closest("[data-fc-more]")) { list.limit += 150; drawRows(); }
      });
    }
    drawRows();
  }
  function rowHtml(r, latest) {
    if (list.html.has(r.t)) return list.html.get(r.t);
    const U = window.SMR.u, m1 = r.m1 || {}, c3 = chg(r, "m3");
    const pc = (x) => (x == null ? "—" : U.pct(x * 100, 0, false));
    const h = `<a class="fc-row" href="#/ticker/${encodeURIComponent(r.t)}"><div class="grow"><b>${U.esc(r.t)}</b> <span class="small muted">${U.esc(U.niceName(r.name || ""))}</span>
        <div class="small muted">1 month: ${U.pxc(m1.lo)}–${U.pxc(m1.hi)} (80% range) · ${pc(m1.up)} chance it rises${r.claude ? " · Claude too" : ""}${r.made && r.made !== latest ? ` · from the ${U.dateFmt(r.made)} close` : ""}</div>
        <div class="chips-row">${ratingChip(r.rating ? { rating: r.rating } : null)}${valChip(r.val ? { label: r.val } : null)}${bubChip(r.bubble ? { level: r.bubble } : null)}</div></div>
        <div class="quote"><div class="small muted">3-month middle</div><div class="${c3 == null ? "" : c3 >= 0 ? "up" : "down"}"><b>${c3 == null ? "—" : U.pct(c3 * 100, 1)}</b></div><div class="small muted">from ${U.px(r.close)}</div></div></a>`;
    list.html.set(r.t, h);
    return h;
  }
  function drawRows() {
    const U = window.SMR.u, d = list.shown, el = document.getElementById("fc-rows"), sub = document.getElementById("fc-sub");
    if (!el || !d) return;
    const all = d.stocks || [];
    const latest = all.reduce((m, r) => (String(r.made || "") > m ? String(r.made) : m), "");
    const older = all.filter((r) => r.made && r.made !== latest).length;
    if (sub) {
      sub.innerHTML = `Every stock the app forecasts: ${U.num(all.length)}, from the ${U.dateFmt(latest)} close${older ? ` (${U.num(older)} from an earlier one: their latest prices were late)` : ""}. Tap one for its chart, peaks and dips, valuation and news.${d._saved ? ` <span class="muted">The network was slow or offline, so this is the copy saved on this phone.</span>` : ""}`;
    }
    const q = list.q.trim().toUpperCase();
    const rows = all.filter((r) => !q || r.t.includes(q) || String(r.name || "").toUpperCase().includes(q)).sort(SORTS[list.sort][1]);
    const more = rows.length - list.limit;
    el.innerHTML = rows.slice(0, list.limit).map((r) => rowHtml(r, latest)).join("")
      + (more > 0 ? `<button class="btn" data-fc-more="1">Show ${U.num(Math.min(150, more))} more (${U.num(more)} not shown)</button>` : "")
      || `<p class="small muted">No stock matches “${U.esc(list.q)}”.</p>`;
  }

  window.SMRStock = { render, renderList, load, devFor, devAll };
})();
