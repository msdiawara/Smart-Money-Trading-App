/* Smart Money Radar — Results: every forecast graded when its date arrives. What it said, what happened, the win
   rate against what the forecasts expected, and why each miss missed, with what the learning loop is doing about
   that kind of miss. Data: data/results.json (pipeline/review.py), loaded when the tab is opened. */
(() => {
  "use strict";
  const HZ = [["1d", "1 day"], ["1w", "1 week"], ["1m", "1 month"], ["3m", "3 months"], ["1y", "1 year"]];
  const ADJ = { "1d": "1-day", "1w": "1-week", "1m": "1-month", "3m": "3-month", "1y": "1-year" };
  const SRC = [["quant", "Model"], ["claude", "Claude"], ["blend", "Combined"], ["quant_bt", "Backtest"]];
  const CAUSE_ORDER = ["market", "earnings", "news", "volatility", "trend", "unknown"];
  const st = { src: null, h: "all", cause: "all", list: "all", shownMiss: 20, shownAll: 40, causeWin: "all" };
  const E = (v) => String(v ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const pc = (v, d = 0) => (v == null || isNaN(v) ? "—" : `${(v * 100).toFixed(d)}%`);
  const signed = (v, d = 1) => (v == null || isNaN(v) ? "—" : `${v > 0 ? "+" : v < 0 ? "−" : ""}${Math.abs(v * 100).toFixed(d)}%`);
  const lr = (x) => (x == null || isNaN(x) ? null : Math.expm1(x));        // log return → simple return
  const mon = (m) => { const [y, mm] = String(m || "").split("-"); const d = new Date(Date.UTC(+y, +mm - 1, 15));
    return isNaN(d) ? E(m) : d.toLocaleDateString(undefined, { month: "short", year: "numeric", timeZone: "UTC" }); };
  const hName = (k) => (HZ.find(([a]) => a === k) || [k, k])[1];
  // "Sep 4", with the year only when it isn't this year's (lists stay on one line)
  const sd = (s) => { if (!s) return "—"; const d = new Date(s + "T12:00:00"); if (isNaN(d)) return E(s);
    return d.toLocaleDateString(undefined, d.getFullYear() === new Date().getFullYear() ? { month: "short", day: "numeric" } : { month: "short", day: "numeric", year: "numeric" }); };
  // prices on a miss card keep their cents: the gap past the range can be small next to the price
  const px2 = (v) => (v == null || isNaN(v) ? "—" : Math.abs(v) < 1 ? `$${Number(v).toFixed(4)}`
    : `$${Number(v).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`);

  function data() { return (window.SMR.data() || {}).results || null; }

  function render(box) {
    const U = window.SMR.u, esc = U.esc, D = data();
    const srcs = (D && D.sources) || {};
    const have = SRC.filter(([k]) => srcs[k] && (srcs[k].headline || {}).n);
    if (!st.src || !srcs[st.src]) st.src = (have.find(([k]) => k === "quant") || have[0] || ["quant"])[0];
    const head = `<h1 class="page-title">Results</h1>
      <p class="page-sub">Every forecast, graded when its date arrives: what it said, what happened, and why the misses missed. Nothing is edited after the fact.${D && D.generated_at ? ` <span class="faint">Updated ${esc(U.ago(D.generated_at))}.</span>` : ""}</p>`;
    if (!have.length) {
      box.innerHTML = `${head}<div class="empty res-empty">Results appear as forecasts are graded: 1-day ones after the next trading day, 1-month ones after a month. The Backtest view fills in first, from the same model re-run on past dates.</div>`;
      return;
    }
    const S = srcs[st.src];
    const hzHave = HZ.filter(([k]) => (S.by_h || {})[k]);
    if (st.h !== "all" && !(S.by_h || {})[st.h]) st.h = "all";
    const H = st.h === "all" ? null : S.by_h[st.h];
    box.innerHTML = `${head}
      <div class="seg" role="group" aria-label="Whose forecasts">${have.map(([k, l]) => `<button data-rsrc="${k}" aria-pressed="${st.src === k}">${l}</button>`).join("")}</div>
      <div class="seg compact" role="group" aria-label="Horizon"><button data-rh="all" aria-pressed="${st.h === "all"}">All</button>${hzHave.map(([k, l]) => `<button data-rh="${k}" aria-pressed="${st.h === k}">${l.replace(" months", "m").replace(" month", "m").replace(" week", "w").replace(" day", "d").replace(" year", "y")}</button>`).join("")}</div>
      ${st.src === "quant_bt" ? `<p class="notice">Backtest: the same model re-run on past dates with only what was known then. It's how the app could be graded from day one; it isn't a live record.</p>` : ""}
      ${heroPanel(S, H, U)}
      ${expectedPanel(S, U)}
      <section class="panel"><h3>Month by month${H ? ` <span class="faint small">(${ADJ[st.h]} forecasts)</span>` : ""}</h3><div id="r-trend"></div>
        <p class="small muted">Share of forecasts made each month that ended inside their 80% range (the band is the 75–85% target zone), and share of direction calls that were right.${H ? "" : " All horizons together: the latest months hold only the short ones, whose longer siblings aren't graded yet."}</p>
        <details class="more"><summary>The numbers</summary><div class="scroll-x">${trendTable(S, H, U)}</div></details></section>
      ${causesPanel(S, H, D, U)}
      ${groupsPanel(S, U)}
      ${missesPanel(S, H, U)}
      ${gradedPanel(S, U)}
      ${changesPanel(D, U)}
      <p class="small muted">${esc(((D && D.definitions) || {}).counts || "")}</p>`;
    drawTrend(box.querySelector("#r-trend"), S, H);
    box.onclick = (ev) => {
      if (!box.querySelector(".res-hero")) return;          // another page is showing
      const b = ev.target.closest("button");
      if (!b || !box.contains(b)) return;
      const d = b.dataset;
      if (d.rsrc) { st.src = d.rsrc; st.cause = "all"; st.shownMiss = 20; st.shownAll = 40; }
      else if (d.rh) { st.h = d.rh; st.cause = "all"; st.shownMiss = 20; st.shownAll = 40; }
      else if (d.rcause) { st.cause = d.rcause; st.shownMiss = 20; }
      else if (d.rlist) { st.list = d.rlist; st.shownAll = 40; }
      else if (d.rcw) { st.causeWin = d.rcw; }
      else if (d.rmore === "miss") { st.shownMiss += 20; }
      else if (d.rmore === "all") { st.shownAll += 40; }
      else return;
      const y = window.scrollY;
      render(box);
      window.scrollTo(0, y);
    };
  }

  // ---------------------------------------------------------------- the headline
  function heroPanel(S, H, U) {
    const h0 = S.headline || {}, X = H || h0;
    const n = X.n, win = X.win;
    const rec = H ? H.recent : h0.recent;
    const recWin = rec && rec.n ? (rec.win && typeof rec.win === "object" ? rec.win.p : rec.win) : null;
    // the interval comes from the forecasts kept in detail (about 15 months), so it's stated with their own rate
    const ci = H && H.win_ci ? ` Over the ${U.num(H.win_ci.n)} kept in detail from the last 15 months or so: <b>${pc(H.win_ci.p)}</b> (likely ${pc(H.win_ci.lo)}–${pc(H.win_ci.hi)}, allowing for overlapping forecasts).` : "";
    const cmp = X.dir_cmp != null ? ` (${pc(X.dir_cmp)} right on those ${U.num(X.dir_cmp_n)} calls)` : "";
    const tgt = win == null ? "" : win >= 0.75 && win <= 0.85 ? "inside the 75–85% target zone" : win < 0.75 ? "below the target zone: the ranges have been too narrow" : "above the target zone: the ranges have been wider than needed";
    const shown = (S.misses || []).filter((m) => !H || m.h === st.h).length;
    const first = H ? H.first : h0.first;
    return `<section class="panel res-hero">
      <div class="res-big"><b>${pc(win)}</b><span>of ${U.num(n)} graded ${H ? ADJ[st.h] + " " : ""}forecasts ended inside their 80% range</span></div>
      <p class="small">An 80% range should hold about 80% of outcomes; this is ${tgt}.${ci}${recWin != null ? ` Last ${(data() || {}).recent_days || 90} days: <b>${pc(recWin)}</b> of ${U.num(rec.n)} kept in detail.` : ""}</p>
      <div class="kpis">
        <div class="kpi"><span>Direction right</span><b>${pc(X.dir)}</b><span class="kpi-sub">expected ${pc(X.dir_expected)}${cmp} · “always up” ${pc(X.always_up)}</span></div>
        <div class="kpi"><span>Better than a random walk</span><b>${X.skill != null ? signed(X.skill) : "—"}</b><span class="kpi-sub">quantile score (above 0 is better)</span></div>
        <div class="kpi"><span>Missed their range</span><b>${X.n_miss != null ? U.num(X.n_miss) : "—"}</b><span class="kpi-sub">${shown ? `the newest ${U.num(shown)} listed below, with why` : "taken apart as they're graded"}</span></div>
        <div class="kpi"><span>Graded so far</span><b>${U.num(n)}</b><span class="kpi-sub">${first ? `made since ${mon(first)}` : ""}</span></div>
      </div>
      <p class="small muted" style="margin-bottom:0">Direction calls are near a coin flip for everyone: 52–56% right is good, and nobody sustains 80%. The range is the promise this tool can keep, so it's the win rate shown first.</p></section>`;
  }

  function expectedPanel(S, U) {
    const part = HZ.some(([k]) => ((S.by_h || {})[k] || {}).dir_cmp != null);
    const rows = HZ.filter(([k]) => (S.by_h || {})[k]).map(([k, l]) => {
      const x = S.by_h[k];
      return `<tr><td class="nw" style="text-align:left">${l}</td><td>${U.num(x.n)}</td><td><b>${pc(x.win)}</b></td><td class="faint">80%</td>
        <td>${pc(x.dir)}</td><td class="faint">${pc(x.dir_expected)}${x.dir_cmp != null ? ` <span class="nw">(${pc(x.dir_cmp)} right)</span>` : ""}</td><td class="faint">${pc(x.always_up)}</td><td>${x.skill != null ? signed(x.skill) : "—"}</td></tr>`;
    }).join("");
    const kept = part;
    return `<section class="panel"><h3>What it expected, and what happened</h3><div class="scroll-x"><table class="cmp">
      <tr><th style="text-align:left">Horizon</th><th>Graded</th><th>Inside range</th><th>Target</th><th>Direction right</th><th>Expected</th><th>Always up</th><th>Vs random walk</th></tr>${rows}</table></div>
      <p class="small muted">“Expected” is the average of the forecasts' own odds for the side they picked: a “55% up” forecast expects to be right 55% of the time, so this shows whether they were over- or under-confident.${kept ? " Where the odds weren't recorded for every call (forecasts graded before this version), “expected” and the “right” beside it come from the same calls: those that recorded it, or else the ones kept in detail." : ""}</p></section>`;
  }

  // ---------------------------------------------------------------- month by month
  function drawTrend(el, S, H) {
    const T = ((H ? H.trend : S.trend) || []).filter((x) => x.n >= 5);
    if (!el) return;
    if (T.length < 2) { el.innerHTML = `<p class="small muted">The chart appears once two months have graded forecasts.</p>`; return; }
    const C = window.SMRCharts;
    C.keep(el, () => C.lines(el, { x: T.map((x) => mon(x.m)), series: [
      { name: "Inside the 80% range", values: T.map((x) => x.win), color: "var(--series-1)", n: T.map((x) => x.n) },
      { name: "Direction right", values: T.map((x) => x.dir), color: "var(--series-2)" }],
      band: { lo: 0.75, hi: 0.85, label: "Target zone for the range" }, yDomain: [0.4, 1], aria: "Win rate by the month forecasts were made" }));
  }
  function trendTable(S, H, U) {
    const T = (H ? H.trend : S.trend) || [];
    return `<table class="cmp"><tr><th style="text-align:left">Made in</th><th>Graded</th><th>Inside range</th><th>Direction right</th><th>Missed</th></tr>
      ${T.slice().reverse().map((x) => `<tr><td class="nw" style="text-align:left">${mon(x.m)}</td><td>${U.num(x.n)}</td><td>${pc(x.win)}</td><td>${pc(x.dir)}</td><td>${U.num(x.miss)}</td></tr>`).join("")}</table>`;
  }

  // ---------------------------------------------------------------- why the misses missed
  function causesPanel(S, H, D, U) {
    const names = (D && D.cause_names) || {};
    const all = H ? H.causes || {} : (S.causes || {})[st.causeWin === "recent" ? "recent" : "all"] || {};
    const pending = all.pending || 0;
    const tot = CAUSE_ORDER.reduce((a, c) => a + (all[c] || 0), 0);
    const lesson = Object.fromEntries((S.lessons || []).map((l) => [l.cause, l]));
    const fixFor = (cz) => {
      if (H) return ((H.fix || {})[cz]) || "";
      const l = lesson[cz];
      if (!l || !l.fix) return "";
      const k = l.top_h, many = Object.keys(l.fix).length > 1;
      return `${many ? `For ${ADJ[k] || k} forecasts, which have the most of these: ` : ""}${l.fix[k] || ""}`;
    };
    const win = H ? "" : `<div class="seg compact" role="group" aria-label="Which misses"><button data-rcw="all" aria-pressed="${st.causeWin !== "recent"}">All</button><button data-rcw="recent" aria-pressed="${st.causeWin === "recent"}">Last ${(D && D.recent_days) || 90} days</button></div>`;
    const wait = pending ? ` ${U.num(pending)} more ${pending > 1 ? "aren't" : "isn't"} taken apart yet (${st.src === "quant_bt" ? "backtest misses are taken apart as the backtest runs" : "live ones are, within the hour after they're graded"}).` : "";
    if (!tot) return `<section class="panel"><h3>Why the misses missed</h3>${win}<p class="small muted">No misses have been taken apart yet.${wait}</p></section>`;
    const rows = CAUSE_ORDER.filter((c) => all[c]).map((c) => {
      const share = all[c] / tot;
      return `<li class="res-cause"><div class="res-cause-top"><b>${U.esc(names[c] || c)}</b><span>${pc(share)} <span class="faint">(${U.num(all[c])})</span></span></div>
        <div class="meter" aria-hidden="true"><div style="width:${Math.max(2, share * 100).toFixed(1)}%;background:var(--series-1)"></div></div>
        <p class="small muted">${U.esc(fixFor(c))}</p></li>`;
    }).join("");
    return `<section class="panel"><h3>Why the misses missed</h3>${win}
      <p class="small">${U.num(tot)} misses kept in detail, each taken apart with what happened in its window: the market's surprise times the stock's beta, its earnings-day move, its biggest one-day shock, and how volatile it was. The biggest part is the cause. Under each: what the learning loop does about it.${wait}</p>
      <ul class="res-causes">${rows}</ul></section>`;
  }

  function groupsPanel(S, U) {
    const k = st.h !== "all" ? st.h : (S.by_h || {})["1m"] ? "1m" : (HZ.find(([x]) => (S.by_h || {})[x]) || ["1m"])[0];
    const G = ((S.by_h || {})[k] || {}).groups || [];
    if (!G.length) return "";
    const rows = G.map((g) => `<tr><td style="text-align:left">${U.esc(g.label)}</td><td>${U.num(g.n)}</td><td><b>${pc(g.win && g.win.p)}</b></td><td>${pc(g.dir && g.dir.p)}</td><td>${g.skill != null ? signed(g.skill) : "—"}</td></tr>`).join("");
    return `<section class="panel"><h3>Where it does better or worse <span class="faint small">(${ADJ[k] || k} forecasts)</span></h3><div class="scroll-x"><table class="cmp">
      <tr><th style="text-align:left">Forecasts</th><th>Graded</th><th>Inside range</th><th>Direction right</th><th>Vs random walk</th></tr>${rows}</table></div>
      <p class="small muted">From the forecasts kept in detail; slices with fewer than 20 aren't shown. A slice that keeps missing is what the range learner tests for wider (or narrower) ranges.</p></section>`;
  }

  // ---------------------------------------------------------------- the misses, explained
  function whyText(w, U, h) {
    const p = w.parts || {}, oneDay = h === "1d";
    const beta = `${w.beta != null ? w.beta.toFixed(1) : "1"}${w.beta_est ? " (estimated from its prices)" : ""}`;
    const by = {
      market: () => `The S&P 500 moved ${signed(lr(w.mkt))} ${oneDay ? "that day" : "over the window"}${w.mkt_err != null ? ` (${signed(lr(w.mkt_err))} against what its own forecast expected)` : ""}; at a beta of ${beta}, that's ${signed(lr(p.market))} of this stock's ${signed(lr(w.err))} move away from the middle of the range.`,
      earnings: () => `Its earnings reaction on ${sd(w.ed)}${w.est ? " (an earnings day estimated from a volume spike)" : ""}: ${signed(lr(w.em))} beyond the market's move that day.`,
      news: () => `${sd(w.jd)}: ${signed(lr(w.jm))} in one day beyond the market (${w.jz != null ? w.jz.toFixed(0) : "?"}× a normal day for it)${w.f8 && w.f8.label ? `, the day of an 8-K: ${U.esc(w.f8.label.toLowerCase())}` : ""}.`,
      volatility: () => oneDay
        ? `The day's move was ${w.vol != null ? w.vol.toFixed(1) : "?"}× the size of a typical day the forecast allowed for.`
        : `It swung ${w.vol != null ? w.vol.toFixed(1) : "?"}× as much from day to day as the forecast allowed, across the whole window.`,
      trend: () => `A steady ${signed(lr(w.err))} drift past the middle of the range, with no single big day.`,
      unknown: () => "The prices for this window weren't available, so it couldn't be taken apart.",
    };
    let s = (by[w.c] || by.unknown)();
    const sg = (w.err || 0) >= 0 ? 1 : -1, also = [];
    if (w.c !== "market" && p.market * sg > 0 && Math.abs(p.market) >= 0.25 * Math.abs(w.err || 0) && w.mkt != null) also.push(`the market ${signed(lr(w.mkt))}`);
    if (w.c !== "earnings" && w.em != null && w.em * sg > 0 && Math.abs(p.earnings || 0) >= 0.25 * Math.abs(w.err || 0)) also.push(`earnings ${signed(lr(w.em))} on ${sd(w.ed)}`);
    if (w.c !== "news" && w.jm != null && w.jm * sg > 0 && Math.abs(w.jm) >= 0.25 * Math.abs(w.err || 0)) also.push(`a ${signed(lr(w.jm))} day on ${sd(w.jd)}`);
    if (also.length) s += ` Also: ${also.join("; ")}.`;
    return s;
  }
  function missCard(m, U, names, S) {
    const w = m.why || null, esc = U.esc;
    const fix = m.fix || (((S.by_h || {})[m.h] || {}).fix || {})[(w && w.c) || "pending"] || "";
    const above = (w && w.side === "above") || (m.p1 != null && m.hi != null && m.p1 > m.hi);
    const gap = m.p1 != null && m.hi != null && m.lo != null ? (above ? m.p1 - m.hi : m.lo - m.p1) : null;
    const label = !w ? "Not taken apart yet" : w.c === "unknown" ? "Couldn't be taken apart" : names[w.c] || w.c;
    return `<li class="res-miss"><div class="res-miss-top"><a href="#/ticker/${encodeURIComponent(m.t)}"><b>${esc(m.t)}</b></a> <span class="faint">· ${esc(hName(m.h))}</span>
        <span class="res-tag miss">${above ? "▲ Ended above its range" : "▼ Ended below its range"}</span></div>
      <p class="small">${m.lo != null ? `Said <b>${px2(m.lo)}–${px2(m.hi)}</b> (middle ${px2(m.mid)}) from ${px2(m.p0)} on ${sd(m.made)}.` : `Made ${sd(m.made)}.`}
        Ended <b>${px2(m.p1)}</b> on ${sd(m.end)} (${signed(m.ret)})${gap != null && m.p0 ? `, ${px2(gap)} ${above ? "above the top" : "below the bottom"} of the range` : ""}${m.delisted ? " (its last trade before delisting)" : ""}.</p>
      <p class="small"><b>Why:</b> ${w ? `${esc(label)}.${w.c !== "unknown" ? ` ${whyText(w, U, m.h)}` : ""}` : `not taken apart yet (${st.src === "quant_bt" ? "the backtest does it as it runs" : "it will be within the hour"}).`}</p>
      ${fix && w ? `<p class="small muted"><b>What changes:</b> ${esc(fix)}</p>` : ""}</li>`;
  }
  function missesPanel(S, H, U) {
    const D = data(), names = (D && D.cause_names) || {};
    let M = (S.misses || []).filter((m) => st.h === "all" || m.h === st.h);
    const full = H ? H.causes || {} : (S.causes || {}).all || {};
    const total = Object.values(full).reduce((a, b) => a + b, 0);
    if (st.cause !== "all") M = M.filter((m) => ((m.why || {}).c || "pending") === st.cause);
    const keys = CAUSE_ORDER.concat(["pending"]).filter((c) => full[c]);
    const chips = [["all", `All (${U.num(total)})`]].concat(keys.map((c) => [c, `${names[c] || c} (${U.num(full[c])})`]));
    const listed = (S.misses || []).filter((m) => st.h === "all" || m.h === st.h).length;
    return `<section class="panel"><h3>Misses, explained <span class="faint small">(newest first)</span></h3>
      <div class="chips" role="group" aria-label="Filter by cause">${chips.map(([k, l]) => `<button class="chip" data-rcause="${k}" aria-pressed="${st.cause === k}">${U.esc(l)}</button>`).join("")}</div>
      ${listed < total ? `<p class="small muted">The newest ${U.num(listed)} of ${U.num(total)} are listed (the newest of each horizon); the counts on the buttons are all of them.</p>` : ""}
      ${M.length ? `<ul class="res-misses">${M.slice(0, st.shownMiss).map((m) => missCard(m, U, names, S)).join("")}</ul>
        ${M.length > st.shownMiss ? `<button class="btn" data-rmore="miss">Show more (${U.num(M.length - st.shownMiss)})</button>` : ""}`
        : `<p class="small muted">${st.cause !== "all" && full[st.cause] ? "None of the newest listed have that cause." : "No misses here."}</p>`}</section>`;
  }

  // ---------------------------------------------------------------- every graded forecast
  function gradedPanel(S, U) {
    let G = (S.graded || []).filter((g) => st.h === "all" || g.h === st.h);
    const n0 = G.length;
    if (st.list === "won") G = G.filter((g) => g.win); else if (st.list === "missed") G = G.filter((g) => !g.win);
    const mark = (ok, yes, no) => `<span class="res-tag ${ok ? "won" : "miss"}">${ok ? "✓ " + yes : "✗ " + no}</span>`;
    const rows = G.slice(0, st.shownAll).map((g) => `<li class="res-row"><div><a href="#/ticker/${encodeURIComponent(g.t)}"><b>${U.esc(g.t)}</b></a> <span class="faint small">${U.esc(hName(g.h))} · ${sd(g.made)} → ${sd(g.end)}</span></div>
      <div class="small">${g.lo != null ? `Said ${U.pxc(g.lo)}–${U.pxc(g.hi)} · ` : ""}ended ${g.p1 != null ? U.pxc(g.p1) : ""} (${signed(g.ret)})</div>
      <div class="res-marks">${mark(g.win, "In range", "Missed")}${g.dir != null ? mark(g.dir, "Right way", "Wrong way") : ""}</div></li>`).join("");
    return `<section class="panel"><h3>Every graded forecast <span class="faint small">(newest first)</span></h3>
      <div class="seg compact" role="group" aria-label="Which forecasts"><button data-rlist="all" aria-pressed="${st.list === "all"}">All</button><button data-rlist="won" aria-pressed="${st.list === "won"}">In range</button><button data-rlist="missed" aria-pressed="${st.list === "missed"}">Missed</button></div>
      ${G.length ? `<ul class="res-rows">${rows}</ul>${G.length > st.shownAll ? `<button class="btn" data-rmore="all">Show more (${U.num(G.length - st.shownAll)})</button>` : ""}` : `<p class="small muted">None yet.</p>`}
      <p class="small muted">The newest of each horizon are listed (${U.num(n0)} here); the rates above count every graded forecast.</p></section>`;
  }

  // ---------------------------------------------------------------- what the misses have changed
  function changesPanel(D, U) {
    const C = (D && D.corrections) || {};
    const status = (x, on, off, learning) => (x.status === "applied" ? on : x.status === "held" ? off : x.status === "n/a" ? `<span class="faint">doesn't run</span>` : learning);
    const rows = HZ.filter(([k]) => C[k]).map(([k, l]) => {
      const c = C[k], cal = c.calibration || {}, rg = c.range || {}, kp = c.keeper || {};
      const sc = cal.scale || 1;
      return `<tr><td class="nw" style="text-align:left">${l}</td>
        <td>${status(cal, `×${sc.toFixed(2)} ${sc < 1 ? "narrower" : "wider"}${cal.cov_before != null ? ` <span class="faint">(${pc(cal.cov_before)}→${pc(cal.cov_after)})</span>` : ""}`, "held back", "learning")}</td>
        <td>${status(rg, "on", "tested, off", "learning")}</td>
        <td>${status(kp, `×${(kp.m || 1).toFixed(2)}`, "watching", "learning")}</td></tr>`;
    }).join("");
    const sig = C.signal || {};
    const ALGO = { ridge: "ridge regression", huber: "robust regression", spline: "spline regression", trees: "boosted trees",
      random_forest: "random forest", catboost: "CatBoost", neural_net: "neural network", average: "average of all", fusion: "fused model" };
    const fus = (sig.fusion || []).map((b) => ALGO[b] || b).join(", ");
    const sigLine = sig.on ? `On: ${sig.model === "fusion" ? `the fused model (${U.esc(fus)})` : U.esc(ALGO[sig.model] || sig.model)} nudges forecasts by what its test found.`
      : `Off: ${sig.t != null && sig.t_needed ? `no algorithm has passed its test yet (t ${sig.t.toFixed(1)}; ${sig.t_needed.toFixed(1)} needed${sig.test_months ? `, over ${sig.test_months} test months` : ""}).` : "it switches on only when an algorithm passes its test."}`;
    const fusLine = fus ? `The fused model averages the three algorithms with the best record so far: now ${U.esc(fus)}. It competes with the others every week, tested only on months it didn't help choose, and can be put in use after 9 months as a fused model${sig.fusion_months != null ? ` (${U.num(sig.fusion_months)} so far)` : ""}.` : "";
    return `<section class="panel"><h3>What the misses have changed</h3><div class="scroll-x"><table class="cmp">
      <tr><th style="text-align:left">Horizon</th><th>Calibration</th><th>Range learner</th><th>Keeper</th></tr>${rows}</table></div>
      <p class="small"><b>Signal learner:</b> ${sigLine}</p>${fusLine ? `<p class="small">${fusLine}</p>` : ""}
      <p class="small muted">These apply to the model's own forecasts (Claude's ranges are its own). Every correction is learned from graded forecasts and has to beat “no change” on forecasts it wasn't fitted on before it's used. Each stock's own record of misses, and of misses caused by earnings, are inputs to the range learner; news coverage, 8-K filings and the latest financial statements are among the signals the signal learner tests. They count once enough live months have them. Details on Records → Forecasts.</p></section>`;
  }

  window.SMRResults = { render };
})();
