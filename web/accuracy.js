/* Smart Money Radar — Records → Forecasts: how accurate the price forecasts have been, whether that is improving,
   and what the learning loop has changed. Data: data/forecast_accuracy.json (+ on-device Claude forecasts). */
(() => {
  "use strict";
  const HZ = [["1d", "1 day"], ["1w", "1 week"], ["1m", "1 month"], ["3m", "3 months"], ["1y", "1 year"]];
  const SRC = [["quant", "Model"], ["claude", "Claude"], ["blend", "Combined"], ["quant_bt", "Backtest"]];
  const st = { h: "1m", src: "quant" };
  const NAME = { quant: "Model", claude: "Claude", blend: "Combined", quant_bt: "Backtest" };
  const pc = (v, d = 0) => (v == null || isNaN(v) ? "—" : `${(v * 100).toFixed(d)}%`);
  const signed = (v, d = 1) => (v == null || isNaN(v) ? "—" : `${v > 0 ? "+" : ""}${(v * 100).toFixed(d)}%`);
  // "Dec 2025", never "Dec 25" (which reads as a day)
  const mon = (m) => { const [y, mm] = m.split("-"); return new Date(Date.UTC(+y, +mm - 1, 15)).toLocaleDateString(undefined, { month: "short", year: "numeric", timeZone: "UTC" }); };

  let picked = false;
  function render(box) {
    const U = window.SMR.u, A = window.SMR.data().fcAccuracy || {};
    const M = A.metrics || {}, by = M.by || {};
    if (!picked) {                     // open on live grades when there are any, else on the backtest
      picked = true;
      const live = ((by.quant || {})[st.h] || {}).all || {}, bt = ((by.quant_bt || {})[st.h] || {}).all || {};
      if (!live.n && bt.n) st.src = "quant_bt";
    }
    const cur = ((by[st.src] || {})[st.h]) || null;
    const all = (cur && cur.all) || {}, rec = (cur && cur.recent) || {};
    const L = A.learning || {}, cal = (L.calibration || {})[st.h] || {};
    const tgt = (A.targets || {}).cov80_band || [0.75, 0.85];
    const shown = all.cov80 != null ? Math.round(all.cov80 * 100) : null;      // judge the number the reader sees
    const inBand = shown != null && shown >= Math.round(tgt[0] * 100) && shown <= Math.round(tgt[1] * 100);
    const liveSrc = st.src !== "quant_bt";
    const waiting = M.pending_by_src ? (M.pending_by_src[st.src] || 0) : (M.pending || 0);   // this source's own, not everyone's
    box.innerHTML = `
      <div class="notice small">The goal the app holds itself to: <b>80% ranges should contain the real price about 80% of the time</b>, and every forecast should beat a naive guess. Direction is shown honestly: for single stocks, a few points above 50% is the realistic best (Gu, Kelly &amp; Xiu 2020), so the app never claims 80–90% there.</div>
      <div class="seg" role="group" aria-label="Who made the forecast">${SRC.map(([k, l]) => `<button data-src="${k}" aria-pressed="${st.src === k}">${l}</button>`).join("")}</div>
      <div class="seg compact" role="group" aria-label="Horizon">${HZ.map(([k, l]) => `<button data-hz="${k}" aria-pressed="${st.h === k}">${k.toUpperCase()}<span class="sr-only"> (${l})</span></button>`).join("")}</div>
      ${st.src === "quant_bt" ? `<p class="small muted" style="margin:0 2px 10px">Backtest: the same model re-run on past dates for a sample of the stocks that were liquid on each date (including ones that later delisted), using only the prices and event dates known then. Live grades replace it as they come in.</p>` : ""}
      ${!all.n ? `<div class="panel"><p class="small">Nothing graded yet for ${U.esc((SRC.find((s) => s[0] === st.src) || [])[1])} at ${U.esc((HZ.find((h) => h[0] === st.h) || [])[1])}. ${liveSrc && waiting ? `${U.num(waiting)} of its forecasts are written down and waiting for their dates.` : liveSrc ? "Forecasts are recorded daily once prices are stored." : "The backtest runs within a day of setup."} The first 1-day grades arrive the next trading day; 1-year grades take a year.</p></div>` : `
      <div class="tiles">
        <div class="tile"><span>80% ranges held</span><b class="${inBand ? "" : "warn"}">${pc(all.cov80)}</b><small>target 80%${all.cov80_raw != null && st.src === "quant" ? `; ${pc(all.cov80_raw)} before learning` : ""}</small></div>
        <div class="tile"><span>Direction right</span><b>${pc(all.hit)}</b><small>"always up" scored ${pc(all.up)}</small></div>
        <div class="tile"><span>Better than naive</span><b class="${(all.skill || 0) >= 0 ? "" : "warn"}">${signed(all.skill)}</b><small>quantile score skill</small></div>
        <div class="tile"><span>Graded</span><b>${U.num(all.n)}</b><small>${liveSrc ? `${U.num(waiting)} waiting` : "past dates"}</small></div>
      </div>`}
      ${all.n ? `<section class="sec"><h2>Are the ranges honest?</h2><div class="panel"><div id="ch-cov" class="chart"></div>
        <p class="small muted">Share of outcomes inside the 80% range, by the month the forecasts were made. The band is the 75–85% target zone.${st.src === "quant" ? " “Before learning” is the raw model; the gap shows what calibration fixed." : ""}</p></div></section>
      <section class="sec"><h2>Direction</h2><div class="panel"><div id="ch-hit" class="chart"></div>
        <p class="small muted">How often “more likely up” or “more likely down” was right, next to simply always saying up.</p></div></section>
      <section class="sec"><h2>Skill versus a naive forecast</h2><div class="panel"><div id="ch-skill" class="chart"></div>
        <p class="small muted">Above 0% means the forecast's whole distribution scored better than a no-drift random walk with last year's volatility (lower quantile loss, a proper scoring rule).</p></div></section>
      ${rec.pit ? `<section class="sec"><h2>Where outcomes landed</h2><div class="panel"><div id="ch-pit" class="chart"></div>
        <p class="small muted">Each outcome's position in its own forecast (0% = far below the range, 100% = far above), last 120 days. Flat bars at 10% mean well calibrated; tall ends mean ranges too narrow; a tall middle means too wide.${rec.bias != null ? ` Median error ${signed(Math.expm1(rec.bias), 1)} (a consistent tilt the calibration step corrects).` : ""}</p></div></section>` : ""}` : ""}
      ${horizonTable(by)}
      ${learningCard(L)}
      ${lessonsCard(A)}
      ${bubbleCard(M.bubble_check)}
      ${convictionCard(A)}
      <section class="sec"><h2>Claude forecasts made on this phone</h2><div class="panel" id="devfc-acc"><p class="small muted">Checking…</p></div></section>`;
    const again = (sel) => { const nb = box.querySelector(sel); if (nb) nb.focus({ preventScroll: true }); };   // keep the keyboard's place
    box.querySelectorAll("[data-src]").forEach((b) => b.addEventListener("click", () => { st.src = b.dataset.src; render(box); again(`[data-src="${st.src}"]`); }));
    box.querySelectorAll("[data-hz]").forEach((b) => b.addEventListener("click", () => { st.h = b.dataset.hz; render(box); again(`[data-hz="${st.h}"]`); }));
    drawCharts(box, by);
    drawBubble(box, M.bubble_check);
    deviceRecord(box.querySelector("#devfc-acc"));
  }

  function series(by, src, key) {
    const t = ((by[src] || {})[st.h] || {}).trend || [];
    return { months: t.map((r) => r.month), vals: t.map((r) => r[key]), n: t.map((r) => r.n) };
  }
  function mergeMonths(list) { return [...new Set(list.flatMap((s) => s.months))].sort(); }
  function align(months, s) { const m = new Map(s.months.map((k, i) => [k, s.vals[i]])); const n = new Map(s.months.map((k, i) => [k, s.n[i]])); return { values: months.map((k) => (m.has(k) ? m.get(k) : null)), n: months.map((k) => n.get(k)) }; }

  function drawCharts(box, by) {
    const C = window.SMRCharts;
    if (!C) return;
    const q = series(by, "quant", "cov80"), qr = series(by, "quant", "cov80_raw"), cl = series(by, "claude", "cov80"), bl = series(by, "blend", "cov80");
    const bt = series(by, "quant_bt", "cov80");
    const covEl = box.querySelector("#ch-cov");
    const covSeries = st.src === "quant" ? [["Model", q, "var(--series-1)"], ["Before learning", qr, "var(--muted)"]]
      : st.src === "claude" ? [["Claude", cl, "var(--series-2)"], ["Model", q, "var(--series-1)"]]
      : st.src === "quant_bt" ? [["Backtest", bt, "var(--series-1)"]]
      : [["Combined", bl, "var(--series-1)"], ["Claude", cl, "var(--series-2)"]];
    const months = mergeMonths(covSeries.map((s) => s[1]));
    if (covEl) {
      if (!months.length) covEl.innerHTML = `<p class="small muted">The first point appears when the first forecasts are graded.</p>`;
      else C.keep(covEl, () => C.lines(covEl, { x: months.map(mon), yDomain: [0.4, 1], band: { lo: 0.75, hi: 0.85, label: "Target 75–85%" },
        series: covSeries.map(([name, s, color]) => ({ name, color, ...align(months, s) })), aria: "80% range coverage by month" }));
    }
    const src = st.src;
    const hit = series(by, src, "hit"), up = series(by, src, "up");
    const hm = mergeMonths([hit, up]);
    const hitEl = box.querySelector("#ch-hit");
    if (hitEl) {
      if (!hm.length) hitEl.innerHTML = `<p class="small muted">Appears with the first graded forecasts.</p>`;
      else C.keep(hitEl, () => C.lines(hitEl, { x: hm.map(mon), yDomain: [0.3, 0.8], ref: { y: 0.5, label: "coin flip" },
        series: [{ name: NAME[src] || src, color: src === "claude" ? "var(--series-2)" : "var(--series-1)", ...align(hm, hit) },
          { name: "Always say up", color: "var(--muted)", ...align(hm, up) }], aria: "Direction hit rate by month" }));
    }
    const sk = series(by, src, "skill");
    const skEl = box.querySelector("#ch-skill");
    if (skEl) {
      if (!sk.months.length) skEl.innerHTML = `<p class="small muted">Appears with the first graded forecasts.</p>`;
      else C.keep(skEl, () => C.lines(skEl, { x: sk.months.map(mon), ref: { y: 0, label: "naive" }, fmt: (v, d = 0) => `${v > 0 ? "+" : ""}${(v * 100).toFixed(d)}%`,
        series: [{ name: "Skill", color: src === "claude" ? "var(--series-2)" : "var(--series-1)", ...align(sk.months, sk) }], aria: "Skill versus naive by month" }));
    }
    const rec = (((by[src] || {})[st.h]) || {}).recent || {};
    const pitEl = box.querySelector("#ch-pit");
    if (pitEl && rec.pit) {
      const tot = rec.pit.reduce((a, b) => a + b, 0) || 1;
      C.keep(pitEl, () => C.bars(pitEl, { labels: ["0", "10", "20", "30", "40", "50", "60", "70", "80", "90"].map((x) => x + "%"), values: rec.pit.map((v) => v / tot),
        ref: { y: 0.1, label: "calibrated" }, color: src === "claude" ? "var(--series-2)" : "var(--series-1)", valueLabel: "of outcomes", aria: "Distribution of outcomes within forecasts" }));
    }
  }

  function horizonTable(by) {
    const U = window.SMR.u, rows = [];
    for (const [k, l] of HZ) {
      const v = ((by[st.src] || {})[k] || {}), a = v.all || {}, r = v.recent || {};
      if (!a.n) continue;
      const ci = r.in80 && r.in80.p != null ? `<div class="muted small">last 120 days ${pc(r.in80.p)} (${pc(r.in80.lo)}–${pc(r.in80.hi)})</div>` : "";
      rows.push(`<tr><td>${l}</td><td>${U.num(a.n)}</td><td>${pc(a.cov50)}</td><td>${pc(a.cov80)}${ci}</td><td>${pc(a.cov90)}</td><td>${pc(a.hit)} <span class="muted small">vs ${pc(a.up)}</span></td><td>${signed(a.skill)}</td></tr>`);
    }
    return rows.length ? `<section class="sec"><h2>Every horizon</h2><div class="panel scroll-x"><table class="cmp"><tr><th>Horizon</th><th>Graded</th><th>50% held</th><th>80% held</th><th>90% held</th><th>Direction</th><th>Skill</th></tr>${rows.join("")}</table>
      <p class="small muted">Targets: 50%, 80%, 90%. All figures are since the start; under the 80% figure, the last 120 days with its 95% interval (wide, because forecasts made the same week share the market's moves).</p></div></section>` : "";
  }

  const ALGO = { ridge: "Ridge regression", trees: "Gradient-boosted trees", neural_net: "Neural network", average: "Average of the three" };
  function rangeRows(rl) {
    const U = window.SMR.u;
    if (!rl || !Object.keys(rl).length) return "";
    const rows = HZ.filter(([k]) => rl[k]).map(([k, l]) => { const x = rl[k];
      const what = x.status === "applied" ? `On: the stretch varies by stock (×${U.num((x.scale_spread || [])[0], 2)} to ×${U.num((x.scale_spread || [])[1], 2)}), ${signed(x.holdout_gain_vs_single, 1)} better than one stretch for all on recent forecasts`
        : x.status === "held" ? `Off: not better than one stretch for all on recent forecasts (${signed(x.holdout_gain_vs_single || 0, 1)})`
        : `<span class="muted">Learning: ${U.num(x.n || 0)}/${U.num((x.need || {}).n || 1000)} graded, ${U.num(x.periods || 0)}/${U.num((x.need || {}).periods || 6)} months</span>`;
      return `<tr><td>${l}</td><td>${what}</td></tr>`; }).join("");
    return `<h3>Range learner (machine learning)</h3><div class="scroll-x"><table class="cmp left2"><tr><th>Horizon</th><th>Stock-by-stock range widths</th></tr>${rows}</table></div>
      <p class="small muted">Gradient-boosted trees learn which stocks' ranges need more or less room (recent vs usual volatility, a big recent jump, an earnings report inside the window…). Used only where they beat one stretch for all on the newest forecasts they weren't fitted on.</p>`;
  }
  function learningCard(L) {
    const U = window.SMR.u, c = L.calibration || {}, t = L.tilt || {}, b = L.blend || {};
    const src = (x) => `from ${x.from === "backtest" ? "the backtest" : "live grades"}, ${U.num(x.n)} forecasts over ${U.num(x.periods ?? x.dates)} ${x.periods != null ? "independent months" : "days"}`;
    const rows = HZ.map(([k, l]) => { const x = c[k] || {}; return `<tr><td>${l}</td><td>${x.status === "applied" ? `×${U.num(x.scale, 2)}${x.shift ? `, ${signed(Math.expm1(x.shift), 1)} median` : ""}<div class="small muted">${src(x)}</div>`
      : x.status === "held" ? `No change<div class="small muted">Held back: a correction fitted on older forecasts would have made the newest ones worse (${src(x)})</div>`
      : `<span class="muted">Learning: ${U.num(x.n || 0)}/200 graded, ${U.num(x.dates || 0)}/20 days</span>`}</td><td class="nw">${x.cov80_holdout != null ? `${pc(x.cov80_holdout_raw)} → ${pc(x.status === "held" ? x.cov80_holdout_raw : x.cov80_holdout)}<div class="small muted">on ${U.num(x.holdout_days)} later days</div>` : x.cov80_raw != null ? pc(x.cov80_raw) : "—"}</td></tr>`; }).join("");
    return `<section class="sec"><h2>What it has learned</h2><div class="panel">
      <h3>Range calibration</h3><div class="scroll-x"><table class="cmp left2"><tr><th>Horizon</th><th>Adjustment</th><th>80% held: before → after</th></tr>${rows}</table></div>
      <p class="small muted">Learned only from graded forecasts. Forecasts made in the same month share most of their outcome, so evidence is counted in independent months: the correction grows as more months agree, is capped, and is used only if it would have improved the newest forecasts it wasn't fitted on. "Before → after" is that test.</p>
      ${rangeRows(L.range)}
      <h3>Signal learner (machine learning)</h3><p class="small">${U.esc(t.status || "Waiting for graded forecasts.")}${t.test_gain != null ? ` Walk-forward record: ${signed(t.test_gain, 2)} average monthly error reduction over ${U.num(t.test_months)} test months (t = ${U.num(t.t, 1)}; ${U.num(t.t_needed ?? 2.5, 1)} needed, stricter while test months are few).` : ""}</p>
      ${t.candidates && Object.keys(t.candidates).length ? `<div class="scroll-x"><table class="cmp left2"><tr><th>Algorithm</th><th>Monthly gain on unseen months</th><th>t</th></tr>
        ${Object.entries(t.candidates).map(([k, v]) => `<tr><td>${U.esc(ALGO[k] || k)}${t.on && t.model === k ? ` <span class="tag">in use</span>` : !t.on && t.best === k ? ` <span class="small muted">(best)</span>` : ""}</td><td>${signed(v.gain, 2)}</td><td>${U.num(v.t, 1)}</td></tr>`).join("")}</table></div>
        <p class="small muted">Each algorithm is refit month by month on earlier months only and scored on the next, on how each stock did against the others that day. The one used each month is the one with the better record before it, so even the choice of algorithm is tested on data it hadn't seen.</p>` : ""}
      ${t.n ? `<p class="small muted">Learning from ${U.num(t.n)} graded 1-month forecasts (${U.num(t.n_live || 0)} live, the rest from the backtest) over ${U.num(t.months)} months.</p>` : ""}
      ${t.on && t.weights ? `<p class="small muted">Weights: ${Object.entries(t.weights).filter(([, w]) => Math.abs(w) > 1e-5).map(([k, w]) => `${U.esc(k)} ${w > 0 ? "+" : ""}${(w * 10000).toFixed(0)}`).join(", ")} (basis points of 1-month return per standard deviation of the signal).</p>` : ""}
      <h3>Model + Claude blend</h3><p class="small">${HZ.every(([k]) => ((b[k] || {}).w_claude ?? 0.5) === 0.5) ? "Equal weights at every horizon until both have graded forecasts." : HZ.map(([k, l]) => `${l}: Claude ${Math.round(((b[k] || {}).w_claude ?? 0.5) * 100)}%`).join("; ") + "."}</p>
      <p class="small muted">${L.updated ? `Updated ${U.esc(U.ago(L.updated))}.` : ""} Signal tilts must improve forecasts made after the data they were fitted on, month after month, before they're used; most market signals are weak, so "rejected" is the expected result for a long time.</p></div></section>`;
  }

  function lessonsCard(A) {
    const U = window.SMR.u, ls = A.lessons || [], pm = A.postmortems || [];
    if (!ls.length && !pm.length) return `<section class="sec"><h2>Claude's lessons</h2><div class="panel"><p class="small muted">When one of Claude's forecasts misses its 90% range, Claude investigates what happened and names the cause. Causes that at least two reviews agree on go into its later forecasts (its free-text notes are shown here but never fed back, so a misleading web page can't steer it). None yet.</p></div></section>`;
    const cause = { volatility_underestimated: "Volatility underestimated", event_surprise: "Event surprise", drift_bias: "Direction bias", regime_shift: "Regime shift", company_news: "Company news", sector_or_macro_move: "Sector or macro move", data_error: "Data error", unforeseeable: "Unforeseeable" };
    return `<section class="sec"><h2>Claude's lessons</h2><div class="panel">
      <ul class="lessons">${ls.slice(0, 10).map((x) => `<li><b>${U.esc(x.lesson)}</b><div class="small muted">${U.esc(x.t || "")}: ${U.esc(cause[x.cause] || x.cause || "")}, ${U.esc(x.foreseeable === "yes" ? "foreseeable" : x.foreseeable === "no" ? "not foreseeable" : "partly foreseeable")}${x.range_advice && x.range_advice !== "same" ? `; ranges ${U.esc(x.range_advice)}` : ""} · ${U.esc(U.ago(x.made))}</div></li>`).join("")}</ul>
      <p class="small muted">Its later forecasts see the causes that at least two reviews agree on; these notes are for you and aren't fed back to it.</p>
      ${pm.length ? `<details class="more"><summary>Post-mortems (${pm.length})</summary>${pm.map((p) => `<div class="pm"><b>${U.esc(p.t)}</b> <span class="muted small">forecast ${U.dateFmt(p.forecast_made)}, ${U.num(p.h)} sessions</span><p class="small">${U.esc(p.what_happened)}</p></div>`).join("")}</details>` : ""}
    </div></section>`;
  }

  function bubbleCard(bc) {
    const U = window.SMR.u, lv = ["Low", "Moderate", "Elevated", "High"];
    if (!bc || !Object.keys(bc).length) return `<section class="sec"><h2>Bubble check</h2><div class="panel"><p class="small muted">Does a "High" bubble rating mean more crashes? After each 3-month window the app counts how often stocks fell 20%+ from a peak, by the rating they had. First results about three months after setup.</p></div></section>`;
    return `<section class="sec"><h2>Bubble check</h2><div class="panel"><div id="ch-bub" class="chart"></div>
      <p class="small muted">Share of 3-month windows with a 20%+ drop from a peak, by the bubble rating when the forecast was made (bars: 95% intervals). Volatile stocks fall 20% more often anyway, so the test is the gap between each bar and what the model expected from volatility alone (in the tooltip): the rating adds something only if High beats its own expectation.</p></div></section>`;
  }
  function drawBubble(box, bc) {
    const el = box.querySelector("#ch-bub");
    if (!el || !bc) return;
    const lv = ["Low", "Moderate", "Elevated", "High"].filter((k) => bc[k]);
    window.SMRCharts.keep(el, () => window.SMRCharts.bars(el, { labels: lv, values: lv.map((k) => bc[k].p_dd20), lo: lv.map((k) => bc[k].lo), hi: lv.map((k) => bc[k].hi),
      color: "var(--serious)", valueLabel: "fell 20%+", extra: (i) => [{ key: "none", value: bc[lv[i]].expected != null ? pc(bc[lv[i]].expected) : "—", label: "expected from volatility" },
        { key: "none", value: window.SMR.u.num(bc[lv[i]].n), label: "windows" }], aria: "Drawdown frequency by bubble rating" }));
  }

  function convictionCard(A) {
    const U = window.SMR.u, c = A.conviction_record, cl = A.claude_beat_index_record;
    const line = (s, label) => (s && s.n ? `<div class="kv"><span>${label}</span><b>${pc(s.beat_rate)} beat the S&amp;P 500 <span class="small muted">n=${U.num(s.n)}${s.beat_lo != null ? `, 95% ${pc(s.beat_lo)}–${pc(s.beat_hi)}` : ""}${s.mean_excess != null ? `, avg ${signed(s.mean_excess)}` : ""}</span></b></div>` : "");
    return `<section class="sec"><h2>Top conviction record</h2><div class="panel">
      ${line(c, "Top conviction calls") || `<p class="small muted">Each new Top conviction call is written down and graded against the S&amp;P 500 after 63 trading days. The first grades arrive about three months after the first call.</p>`}
      ${line(cl, "All of Claude's calls")}</div></section>`;
  }

  // on-device forecasts, graded here from the stocks' price files
  async function deviceRecord(el) {
    if (!el) return;
    const U = window.SMR.u, all = (window.SMRStock && window.SMRStock.devAll()) || {};
    const tickers = Object.keys(all);
    if (!tickers.length) { el.innerHTML = `<p class="small muted">None yet. On any stock page, “Ask Claude for a fresh forecast” uses your own API key; those forecasts are stored and graded on this phone.</p>`; return; }
    let n = 0, in80 = 0, hit = 0, nh = 0, waiting = 0;
    const rows = [];
    for (const t of tickers.slice(0, 25)) {
      let doc = null;
      try { doc = await window.SMRStock.load(t); } catch { doc = null; }
      const px = doc && doc.px;
      const close = (d) => { if (!px) return null; let k = -1; for (let i = 0; i < px.d.length; i++) if (px.d[i] <= d) k = i; return k >= 0 && (Date.parse(d) - Date.parse(px.d[k])) <= 4 * 864e5 ? px.c[k] : null; };
      for (const f of all[t]) {
        // the price history is split-adjusted: re-base the stored prices for any split since the forecast
        const c0 = f.last_close_date ? close(f.last_close_date) : null, r = c0 && f.last_close ? c0 / f.last_close : 1;
        const k = isFinite(r) && r > 0.005 && r < 200 && Math.abs(Math.log(r)) > 0.02 ? r : 1;
        const base = f.last_close ? f.last_close * k : null;           // "chance it ends higher" was relative to this
        for (const [h, row] of Object.entries(f.horizons || {})) {
          const td = (f.target_dates || {})[h];
          if (!px || !td || td > px.d[px.d.length - 1]) { waiting++; continue; }
          const p0 = close(f.ref_date), p1 = close(td);
          if (p0 == null || p1 == null) continue;
          const lo = row.low_10 * k, hi = row.high_90 * k;
          n++; const ok = p1 >= lo && p1 <= hi; in80 += ok ? 1 : 0;
          if (row.p_up != null && row.p_up !== 0.5) { nh++; hit += (row.p_up > 0.5) === (p1 > (base || p0)) ? 1 : 0; }
          if (rows.length < 12) rows.push(`<tr><td>${U.esc(t)} ${U.esc(h)}</td><td>${U.pxc(lo)}–${U.pxc(hi)}</td><td>${U.pxc(p1)}</td><td>${ok ? "✓" : "✕"}</td></tr>`);
        }
      }
    }
    el.innerHTML = n ? `<div class="kv"><span>Inside the 80% range</span><b>${pc(in80 / n)} <span class="small muted">of ${n}</span></b></div>
        <div class="kv"><span>Direction right</span><b>${nh ? pc(hit / nh) : "—"} <span class="small muted">of ${nh}</span></b></div>
        <table class="cmp"><tr><th>Forecast</th><th>80% range</th><th>Actual</th><th></th></tr>${rows.join("")}</table>
        <p class="small muted">${waiting} horizons still waiting. Small samples: luck dominates until there are dozens.</p>`
      : `<p class="small muted">${waiting} forecast horizons made on this phone are waiting for their dates.</p>`;
  }

  window.SMRAccuracy = { render };
})();
