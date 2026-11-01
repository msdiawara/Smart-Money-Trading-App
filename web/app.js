/* Smart Money Radar — installable web app. No build step, no trackers; data comes from ./data/*.json,
   which the GitHub Actions pipeline refreshes every ~5 minutes. */
(() => {
  "use strict";
  const $ = (s, el = document) => el.querySelector(s);
  const view = $("#view");
  const LS = {
    get(k, d) { try { const v = localStorage.getItem("smr." + k); return v ? JSON.parse(v) : d; } catch { return d; } },
    set(k, v) { try { localStorage.setItem("smr." + k, JSON.stringify(v)); return true; } catch { return false; } },   // false: private mode or full
  };
  const S = { meta: null, feed: [], byId: {}, quotes: {}, leaders: null, health: null, market: null, ideas: null, accuracy: null,
    fcAccuracy: null, audit: null, device: null, loadedAt: 0, version: null, shown: 60 };
  const ENTRY_DEPTH = history.length;      // history entries that belong to whatever came before this app
  const STALE_MIN = 120;                   // data older than this (minutes) is stale everywhere: header dot and self-check

  // ------------------------------------------------------------------ formatting
  const esc = (v) => String(v ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const money = (v) => {
    if (v == null || isNaN(v)) return "—";
    const a = Math.abs(v), s = v < 0 ? "-" : "";
    if (a >= 1e12) return `${s}$${(a / 1e12).toFixed(2)}T`;
    if (a >= 1e9) return `${s}$${(a / 1e9).toFixed(2)}B`;
    if (a >= 1e6) return `${s}$${(a / 1e6).toFixed(2)}M`;
    if (a >= 1e3) return `${s}$${(a / 1e3).toFixed(a >= 1e5 ? 0 : 1)}K`;
    if (a > 0 && a < 1) return `${s}$${a.toPrecision(2)}`;
    return `${s}$${a.toFixed(a < 10 ? 2 : 0)}`;
  };
  const num = (v, d = 0) => (v == null || isNaN(v) ? "—" : Number(v).toLocaleString(undefined, { maximumFractionDigits: d }));
  const pct = (v, d = 1, sign = true) => {
    if (v == null || isNaN(v)) return "—";
    if (v !== 0 && Math.abs(v) < 0.5 * Math.pow(10, -d)) return `${v < 0 ? "-" : sign ? "+" : ""}<${Math.pow(10, -d)}%`;
    return `${sign && v > 0 ? "+" : ""}${Number(v).toFixed(d)}%`;
  };
  const px = (v) => (v == null || isNaN(v) ? "—" : `$${Number(v).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: Math.abs(v) < 0.01 ? 6 : Math.abs(v) < 1 ? 4 : 2 })}`);
  const dateFmt = (s) => {
    if (!s) return "—";
    const d = new Date(s.length <= 10 ? s + "T12:00:00" : s);
    if (isNaN(d)) return esc(s);
    const opts = s.length <= 10 ? { month: "short", day: "numeric", year: "numeric" } : { month: "short", day: "numeric", hour: "numeric", minute: "2-digit" };
    return d.toLocaleString(undefined, opts);
  };
  const ago = (s) => {
    if (!s) return "";
    const d = new Date(s.length <= 10 ? s + "T16:00:00" : s), m = (Date.now() - d) / 60000;
    if (isNaN(m)) return "";
    if (m < 1) return "just now"; if (m < 60) return `${Math.round(m)}m ago`;
    if (m < 1440) return `${Math.round(m / 60)}h ago`; if (m < 43200) return `${Math.round(m / 1440)}d ago`;
    return dateFmt(s.slice(0, 10));
  };
  const safeUrl = (u) => (typeof u === "string" && /^https:\/\/[^\s"'<>]+$/i.test(u) ? u : null);
  // forecast prices: whole dollars above $100 (cents there are false precision), cents below
  const pxc = (v) => (v == null || isNaN(v) ? "—" : Math.abs(v) < 1 ? px(v)
    : `$${Number(v).toLocaleString(undefined, { minimumFractionDigits: v >= 100 ? 0 : 2, maximumFractionDigits: v >= 100 ? 0 : 2 })}`);
  // SEC names are often ALL CAPS with suffixes: "NVIDIA CORP" → "Nvidia Corp"; short acronyms stay (IBM, AMD)
  const niceName = (n) => {
    const s = String(n || "").replace(/\s+\/[A-Z]{2}\/?$/, "").trim();
    if (!s || s !== s.toUpperCase() || s.length < 5) return s;
    return s.split(" ").map((w) => (w.length <= 3 && !/^(INC|CO|THE|AND|OF)$/.test(w) ? w : w.charAt(0) + w.slice(1).toLowerCase())).join(" ");
  };
  const KIND = { insider: ["", "Insider"], fund: ["", "Fund 13F"], activist: ["", "Activist / 5%+"], congress: ["", "Congress"], etoro: ["", "eToro trader"], etf: ["", "Active ETF (daily)"] };
  const ACTION = { buy: ["Buy", "buy"], sell: ["Sell", "sell"], new: ["New position", "buy"], add: ["Added", "buy"], trim: ["Trimmed", "sell"],
    exit: ["Sold out", "sell"], stake: ["Stake", "neutral"], open: ["Opened", "buy"], close: ["Closed", "neutral"], exchange: ["Exchange", "neutral"],
    acquire: ["Acquired", "neutral"], dispose: ["Disposed", "neutral"], other: ["Other", "neutral"], sale_notice: ["Plans to sell", "sell"] };
  const SIG = { strong: "Strong", notable: "Notable", mild: "Mild", weak: "Weak", noise: "Noise" };

  function badge(e) {
    let [label, cls] = ACTION[e.action] || [e.action_label || e.action, "neutral"];
    if (e.kind === "etoro" && e.details && e.details.is_buy === false && e.action === "open") { label = "Opened short"; cls = "sell"; }
    if (e.kind === "insider" && !["buy", "sell", "sale_notice"].includes(e.action)) label = e.action_label || label;
    return `<span class="badge ${cls}">${esc(label)}</span>`;
  }
  function scoreBox(e) {
    const bear = e.direction === "bearish" ? " bear" : "";
    const arrow = e.direction === "bullish" ? "▲" : e.direction === "bearish" ? "▼" : "•";
    return `<div class="score sig-${esc(e.signal)}${bear}" title="Signal strength"><b>${esc(e.score ?? "–")}</b><i>${arrow} ${esc(SIG[e.signal] || "")}</i></div>`;
  }
  function riskBar(level, big = false) {
    level = Math.round(Number(level));
    if (!level || isNaN(level)) return big ? `<span class="faint small">Not enough price history yet</span>` : "";
    level = Math.min(5, Math.max(1, level));
    let h = "";
    for (let i = 1; i <= 5; i++) h += `<span class="${i <= level ? "on" + level : ""}"></span>`;
    return `<span class="risk${big ? " big" : ""}" aria-label="Risk ${level} of 5">${h}</span>`;
  }
  const amount = (e) => {
    if (e.kind === "congress" && e.value_range) {
      const [lo, hi] = e.value_range; return lo ? (hi && hi !== lo ? `${money(lo)}–${money(hi)}` : `over ${money(lo)}`) : "";
    }
    if (e.kind === "etoro") return e.details && e.details.investment_pct != null ? `${num(e.details.investment_pct, 1)}% of portfolio` : "";
    if (e.kind === "activist") return e.details && e.details.percent != null ? `${num(e.details.percent, 2)}% of company` : "";
    if (e.kind === "etf") return e.value != null ? `${money(e.value)}${e.details && e.details.weight != null ? ` · ${num(e.details.weight, 1)}% of fund` : ""}` : "";
    return e.value ? money(e.value) : e.shares ? `${num(e.shares)} sh` : "";
  };

  // ------------------------------------------------------------------ data
  const fromCache = new Set();     // files the offline helper answered from its saved copy during this load
  async function getJSON(name, bust) {
    const r = await fetch(`data/${name}?v=${bust || Date.now()}`, { cache: "no-store" });
    if (r.headers.get("X-SMR-From-Cache")) fromCache.add(name);
    if (!r.ok) {
      let off = false;            // the offline helper's own "never saved" answer, as opposed to a server error
      if (r.status === 503) { try { off = !!(await r.json()).offline; } catch { /* a server error page */ } }
      const e = new Error(off ? `${name}: offline, not saved yet` : `${name}: HTTP ${r.status}`); e.offline = off; throw e;
    }
    return r.json();
  }
  const rest = (v) => Promise.all([getJSON("feed.json", v), getJSON("quotes.json", v),
    getJSON("leaders.json", v).catch(() => null), getJSON("health.json", v).catch(() => null),
    getJSON("market.json", v).catch(() => null), getJSON("ideas.json", v).catch(() => null), getJSON("accuracy.json", v).catch(() => null),
    getJSON("forecast_accuracy.json", v).catch(() => null), getJSON("audit.json", v).catch(() => null)]);
  // Returns true when what's on screen should change. A failed load sets S.loadErr (shown in the header until a load
  // succeeds) and returns false.
  async function loadAll(silent) {
    try {
      fromCache.clear();
      let meta, files;
      if (!S.meta) {              // opening the app: everything at once, so a weak signal means one wait, not two
        [meta, files] = await Promise.all([getJSON("meta.json"), rest(Date.now())]);
      } else {                    // a refresh: the small meta file first, and the rest only if there is something new
        meta = await getJSON("meta.json");
        if (meta.version === S.version) { S.fromCache = fromCache.has("meta.json"); S.loadErr = null; S.loadedAt = Date.now(); setUpdated(); return false; }
        files = await rest(meta.version);
      }
      const [feed, quotes, leaders, health, market, ideas, accuracy, fcAccuracy, audit] = files;
      const prevIds = new Set(S.feed.map((e) => e.id));
      // The trades and prices must be from the same update as meta.json (a slow file answered from the saved copy
      // can be older, or an update can land mid-load). If not, they're shown but the version isn't recorded, so the
      // next refresh fetches them again. A saved copy from the same update is fine.
      const g = meta.generated_at;
      const mixed = [feed, quotes].some((d) => d && d.generated_at && g && d.generated_at !== g);
      const sig = [meta].concat(files).map((d) => (d && d.generated_at) || "-").join("|");
      const changed = sig !== S.sig;              // anything on screen different from last time
      S.sig = sig; S.fromCache = fromCache.has("meta.json"); S.mixed = mixed; S.loadErr = null;
      S.meta = meta; S.version = mixed ? null : meta.version; S.feed = feed.events || []; S.quotes = quotes.quotes || {};
      S.leaders = leaders; S.health = health; S.market = market; S.ideas = ideas; S.accuracy = accuracy; S.loadedAt = Date.now();
      S.fcAccuracy = fcAccuracy; S.audit = audit;
      S.device = selfCheck();
      S.byId = Object.fromEntries(S.feed.map((e) => [e.id, e]));
      const fresh = prevIds.size ? S.feed.filter((e) => !prevIds.has(e.id)).length : 0;
      setUpdated();
      if (fresh && silent) toast(`${fresh} new trade${fresh > 1 ? "s" : ""}`);
      const flag = $("#audit-flag"); if (flag) flag.hidden = !(S.device.counts.fail || (S.audit && S.audit.counts && S.audit.counts.fail));
      return changed;
    } catch (err) {
      S.loadErr = err;
      setUpdated();
      if (!S.feed.length) view.innerHTML = `<div class="empty"><p>Couldn't load data yet.</p><p class="small faint">${esc(err.message)}</p>
        <p class="small">If you just set this up, the first refresh takes a few minutes to run on GitHub.</p></div>`;
      return false;
    }
  }
  function setUpdated() {
    const el = $("#updated"), dot = $("#fresh-dot"), err = S.loadErr;
    if (err) { el.textContent = err.offline || !navigator.onLine ? "Offline" : "Update failed"; el.title = err.message || ""; dot.className = "dot bad"; return; }
    const g = S.meta && S.meta.generated_at;
    const mins = g ? (Date.now() - new Date(g)) / 60000 : 999;
    if (mins < -10) {             // stamped in the future: a clock is off, so no age can be trusted (see Status)
      el.textContent = "Clock mismatch"; dot.className = "dot stale"; el.title = "This phone's clock and the data's timestamp disagree"; return;
    }
    el.textContent = g ? (mins < 1 ? "Just now" : ago(g)) + (S.fromCache ? " (saved copy)" : "") : "—";
    el.title = S.fromCache ? "The network was too slow or offline, so the last saved copy is showing" : "";
    dot.className = "dot " + (mins < 20 && !S.fromCache ? "ok" : mins < STALE_MIN ? "stale" : "bad");   // a saved copy is never green
  }
  function toast(msg) {
    const t = $("#toast"); t.textContent = msg; t.hidden = false;
    clearTimeout(toast._t); toast._t = setTimeout(() => (t.hidden = true), 3200);
  }

  // ------------------------------------------------------------------ watchlist & sizing settings
  const watch = () => LS.get("watch", { tickers: [], people: [] });
  const isWatched = (e) => { const w = watch(); return (e.ticker && w.tickers.includes(e.ticker)) || w.people.some((p) => p.id === e.actor.id); };
  function toggleWatchTicker(t) { const w = watch(); w.tickers = w.tickers.includes(t) ? w.tickers.filter((x) => x !== t) : [...w.tickers, t]; LS.set("watch", w); }
  function toggleWatchPerson(a) { const w = watch(); w.people = w.people.some((p) => p.id === a.id) ? w.people.filter((p) => p.id !== a.id) : [...w.people, { id: a.id, name: a.name }]; LS.set("watch", w); }
  const sizingCfg = () => LS.get("sizing", { portfolio: 10000, satellite: 20, risk: 1 });

  // Position-size ceiling from your own risk rules (not advice):
  //  size = portfolio × risk-per-idea ÷ plausible bad month (2 × monthly volatility, at least 10%),
  //  capped by a per-stock maximum that shrinks as the risk level rises, and by your "stock ideas" budget.
  const CAP_BY_LEVEL = { 1: 10, 2: 7.5, 3: 5, 4: 3, 5: 1.5 };
  function sizing(q) {
    const c = sizingCfg();
    const vol = q && q.vol_1y ? q.vol_1y : 0.45;
    const drop = Math.max(0.10, 2 * vol / Math.sqrt(12));
    const level = (q && q.risk_level) || 4;
    const byRisk = c.portfolio * (c.risk / 100) / drop;
    const cap = c.portfolio * Math.min(CAP_BY_LEVEL[level] || 3, c.satellite) / 100;
    const size = Math.max(0, Math.min(byRisk, cap));
    return { size, pctOf: (size / c.portfolio) * 100, drop, level, cap, byRisk, cfg: c, assumedVol: !(q && q.vol_1y) };
  }

  // ------------------------------------------------------------------ views
  const filters = () => LS.get("filters", { kind: "all", dir: "all", strong: false, q: "", sort: "new" });
  function matchesFilters(e, f) {
    if (f.kind !== "all" && e.kind !== f.kind) return false;
    if (f.strong && !["strong", "notable"].includes(e.signal)) return false;
    if (f.dir !== "all" && e.direction !== f.dir) return false;
    if (f.hideNoise !== false && e.signal === "noise" && f.kind !== "insider") return false;
    if (f.q) {
      const q = f.q.toLowerCase();
      if (!`${e.ticker || ""} ${e.company || ""} ${e.actor.name} ${e.actor.role || ""}`.toLowerCase().includes(q)) return false;
    }
    return true;
  }
  function miniTags(q) {
    const C = window.SMRChips;
    if (!C || !q) return "";
    const parts = [q.conv === "top" ? C.ratingChip({ rating: "top" }) : "", q.flag ? `<span class="tag flag"><i aria-hidden="true">⚑</i>${esc(q.flag)}</span>` : "",
      q.val && /^Very /.test(q.val.label || "") ? C.valChip(q.val) : "", q.bubble === "High" || q.bubble === "Elevated" ? C.bubChip({ level: q.bubble }) : ""].filter(Boolean);
    return parts.length ? `<div class="mini-tags">${parts.join("")}</div>` : "";
  }
  function card(e) {
    const [, kl] = KIND[e.kind] || ["", e.kind];
    const q = S.quotes[e.ticker] || {};
    const since = q.price && e.price && e.kind !== "fund" ? ((q.price / e.price) - 1) * 100 : null;
    return `<a class="card${isWatched(e) ? " watch" : ""}" href="#/event/${encodeURIComponent(e.id)}">
      <div class="row start">
        <div class="grow">
          <div class="row" style="gap:8px"><span class="tick">${esc(e.ticker || "—")}</span>${badge(e)}<span class="amt">${esc(amount(e))}</span></div>
          <div class="who">${esc(e.actor.name)} <span class="muted">${esc(e.actor.role ? "– " + e.actor.role : "")}</span></div>
          <div class="kind">${esc(kl)}, ${esc(ago(e.filed_at || e.trade_date))}${e.company ? `, ${esc(niceName(e.company))}` : ""}</div>
        </div>${scoreBox(e)}
      </div>
      <div class="sum">${esc(e.read ? e.read.summary : "")}</div>
      ${e.analysis ? `<div style="margin-top:8px">${stanceChip(e.analysis, true)}</div>` : ""}
      ${miniTags(q)}
      <div class="meta">${q.risk_level ? `<span>Risk ${riskBar(q.risk_level)} ${esc(q.risk_label || "")}</span>` : ""}
        ${q.earnings && q.earnings.days != null && q.earnings.days >= 0 && q.earnings.days <= 14 ? `<span class="warn">Earnings in ${num(q.earnings.days)}d</span>` : ""}
        ${q.rel_volume && q.rel_volume.ratio >= 2.5 ? `<span class="warn">Volume ${num(q.rel_volume.ratio, 1)}× normal</span>` : ""}
        ${q.short_volume && q.short_volume.avg20_pct != null && q.short_volume.pct >= q.short_volume.avg20_pct + 15 ? `<span class="warn">Heavy short selling</span>` : ""}
        ${since != null ? `<span class="${since >= 0 ? "up" : "down"}">${pct(since)} since their price</span>` : ""}
        ${e.trade_date ? `<span>Traded ${esc(dateFmt(e.trade_date))}</span>` : ""}</div>
    </a>`;
  }
  function installHint() {
    const standalone = window.matchMedia("(display-mode: standalone)").matches || navigator.standalone;
    if (standalone || LS.get("hideInstall", false)) return "";
    const ios = /iphone|ipad|ipod/i.test(navigator.userAgent);
    return `<div class="install"><b>Add to your Home Screen</b><p class="small muted" style="margin:4px 0 8px">
      ${ios ? "Tap the Share button in Safari, then “Add to Home Screen”." : "Open your browser menu and tap “Install app” or “Add to Home screen”."}
      It then opens like a normal app and keeps the latest data offline.</p>
      <button class="btn" data-act="hide-install">Got it</button></div>`;
  }
  function renderFeed() {
    const f = filters();
    const kinds = [["all", "All"], ["insider", "Insiders"], ["etf", "ETFs"], ["fund", "Funds"], ["activist", "Activists"], ["congress", "Congress"], ["etoro", "eToro"]];
    let list = S.feed.filter((e) => matchesFilters(e, f));
    if (f.sort === "score") list = [...list].sort((a, b) => (b.score || 0) - (a.score || 0));
    const strongToday = S.feed.filter((e) => e.signal === "strong" && (Date.now() - new Date(e.first_seen || e.filed_at)) < 864e5).length;
    view.innerHTML = `${installHint()}
      <h1 class="page-title">Smart-money moves</h1><p class="page-sub">Insiders, funds, activists, Congress and active ETFs, from official filings, scored and explained.</p>
      <div class="toolbar"><input class="search" id="q" type="search" placeholder="Ticker, company or person" value="${esc(f.q)}" autocomplete="off">
        <select class="search" id="sort" aria-label="Sort"><option value="new"${f.sort === "new" ? " selected" : ""}>Newest</option><option value="score"${f.sort === "score" ? " selected" : ""}>Top score</option></select></div>
      <div class="chips">${kinds.map(([k, l]) => `<button class="chip" data-kind="${k}" aria-pressed="${f.kind === k}">${l}</button>`).join("")}</div>
      <div class="chips"><button class="chip" data-strong aria-pressed="${f.strong}">Strong & notable only</button>
        <button class="chip" data-dir="bullish" aria-pressed="${f.dir === "bullish"}">▲ Bullish</button>
        <button class="chip" data-dir="bearish" aria-pressed="${f.dir === "bearish"}">▼ Bearish</button></div>
      ${strongToday ? `<p class="small muted" style="margin:0 2px 10px">${strongToday} strong signal${strongToday > 1 ? "s" : ""} in the last 24 hours.</p>` : ""}
      <div id="open-t">${openStock(f.q)}</div>
      <div id="list" class="rows">${list.slice(0, S.shown).map(card).join("") || `<div class="empty">No trades match these filters yet.</div>`}</div>
      ${list.length > S.shown ? `<button class="btn more" data-act="more">Show more (${list.length - S.shown} left)</button>` : ""}
      <p class="disc">Signal scores rate how informative a disclosed trade has historically been. They aren't predictions or advice. Always check the original filing.</p>`;
    $("#q").addEventListener("input", debounce((ev) => { LS.set("filters", { ...filters(), q: ev.target.value.trim() }); S.shown = 60; renderList(); }, 200));
    $("#sort").addEventListener("change", (ev) => { LS.set("filters", { ...filters(), sort: ev.target.value }); renderFeed(); });
  }
  function renderList() {
    const f = filters();
    let list = S.feed.filter((e) => matchesFilters(e, f));
    if (f.sort === "score") list = [...list].sort((a, b) => (b.score || 0) - (a.score || 0));
    $("#list").innerHTML = list.slice(0, S.shown).map(card).join("") || `<div class="empty">No trades match.</div>`;
    const o = $("#open-t"); if (o) o.innerHTML = openStock(f.q);
  }
  function openStock(q) {
    const t = String(q || "").trim().toUpperCase().replace(/^\$/, "");
    if (!t || !/^[A-Z][A-Z0-9.]{0,5}$/.test(t) || !(S.quotes[t] || S.feed.some((e) => e.ticker === t))) return "";
    const n = (S.quotes[t] || {}).name || "";
    return `<a class="panel row" href="#/ticker/${encodeURIComponent(t)}" style="text-decoration:none"><div class="grow"><b>${esc(t)}</b> <span class="muted small">${esc(n)}</span><div class="small muted">Forecast, valuation, bubble risk and news</div></div><span class="btn">Open</span></a>`;
  }

  function fact(label, value) { return value == null || value === "" || value === "—" ? "" : `<tr><td>${esc(label)}</td><td>${value}</td></tr>`; }
  function factsFor(e) {
    const d = e.details || {}, r = [];
    const yes = (v) => (v === true ? "Yes" : v === false ? "No" : "Unknown");
    if (e.kind === "insider" && d.notice) {
      r.push(fact("Insider", esc(e.actor.name)), fact("Relationship", esc(e.actor.role)),
        fact("Notice", `Form ${esc(d.form)} — intent to sell${d.amendment ? " (amended)" : ""}`),
        fact("Shares to sell", num(e.shares)), fact("Stated market value", money(e.value)), fact("Implied price", px(e.price)),
        fact("Share of all shares outstanding", d.pct_outstanding != null ? `${num(d.pct_outstanding, 3)}%` : null),
        fact("Approximate sale date", esc(dateFmt(d.approx_sale_date))), fact("Filed with SEC", esc(dateFmt(e.filed_at))),
        fact("Broker", esc(d.broker)), fact("Security / exchange", `${esc(d.security || "")}${d.exchange ? ` · ${esc(d.exchange)}` : ""}`),
        fact("Pre-planned (10b5-1)", yes(d.plan_10b5_1) + (d.plan_adoption_dates && d.plan_adoption_dates.length ? ` <span class="faint">(plan adopted ${d.plan_adoption_dates.map(dateFmt).map(esc).join(", ")})</span>` : "")),
        d.acquisitions && d.acquisitions.length ? fact("How the shares were acquired", d.acquisitions.map((a) => `${esc(dateFmt(a.date))}: ${esc(a.nature || "")}${a.shares ? ` (${num(a.shares)} shares)` : ""}`).join("<br>")) : "",
        d.sold_past_3m && d.sold_past_3m.length ? fact("Sold in the past 3 months", `${d.sold_past_3m.map((s) => `${esc(dateFmt(s.date))}: ${num(s.shares)} shares${s.proceeds ? ` for ${money(s.proceeds)}` : ""}`).join("<br>")}`) : fact("Sold in the past 3 months", "Nothing reported"),
        d.remarks ? fact("Remarks", `<div class="small">${esc(d.remarks)}</div>`) : "");
    } else if (e.kind === "insider") {
      r.push(fact("Insider", esc(e.actor.name)), fact("Role", esc(e.actor.role)),
        fact("Transaction", `${esc(e.action_label)} <span class="faint">(code ${esc((d.codes || []).join(", "))})</span>`),
        fact("Trade date", esc(dateFmt(e.trade_date)) + (e.trade_date_end ? ` → ${esc(dateFmt(e.trade_date_end))}` : "")),
        fact("Filed with SEC", esc(dateFmt(e.filed_at))),
        fact("Filing delay", e.lag_days != null ? `${num(e.lag_days)} business day${e.lag_days === 1 ? "" : "s"}${d.late ? ' <span class="warn">(late)</span>' : ""}` : null),
        fact("Shares", num(e.shares)), fact("Average price", px(e.price)), fact("Total value", money(e.value)),
        fact("Shares owned after", num(e.holdings_after)), fact("Change in their holdings", e.holdings_change_pct != null ? pct(e.holdings_change_pct) : null),
        fact("Pre-planned (10b5-1)", yes(d.plan_10b5_1) + (d.plan_flag_source ? ` <span class="faint">(${esc(d.plan_flag_source)})</span>` : "")),
        fact("Ownership", d.indirect ? `Indirect${d.ownership_nature ? " — " + esc(d.ownership_nature) : ""}` : "Direct"),
        fact("Security", esc(d.security)), fact("Form", esc(d.form) + (d.amendment ? " (amendment)" : "")),
        d.owners && d.owners.length > 1 ? fact("Joint filers", d.owners.map((o) => `${esc(o.name)} <span class="faint">(${esc(o.role)})</span>`).join("<br>")) : "",
        d.footnotes && d.footnotes.length ? fact("Footnotes", d.footnotes.map((n) => `<div class="small" style="margin-bottom:6px">${esc(n)}</div>`).join("")) : "");
    } else if (e.kind === "fund") {
      r.push(fact("Fund", esc(e.actor.name)), fact("Manager", esc(e.actor.role)), fact("Change", esc(e.action_label)),
        fact("Quarter ended", esc(dateFmt(d.period))), fact("Filed", esc(dateFmt(e.filed_at))),
        fact("Filed after quarter end", e.lag_days != null ? `${num(e.lag_days)} days` : null),
        fact("Holdings age today", d.days_since_quarter_end != null ? `${num(d.days_since_quarter_end)} days since the quarter ended` : null),
        fact("Shares now", num(e.action === "exit" ? 0 : e.shares)), fact("Shares last quarter", num(d.prev_shares)),
        fact("Share change", d.pct_change != null ? pct(d.pct_change) : e.action === "new" ? "New" : null),
        fact("Value at quarter end", money(e.value)), fact("Share of fund now", d.weight != null ? `${num(d.weight, 2)}%` : null),
        fact("Share of fund before", d.prev_weight != null ? `${num(d.prev_weight, 2)}%` : null),
        fact("Rank in fund", d.rank ? `#${num(d.rank)} of ${num(d.fund_positions)}` : null),
        fact("Fund's 13F portfolio", money(d.fund_value)), fact("Implied price at quarter end", px(e.price)),
        fact("Class / CUSIP", `${esc(d.class || "")} · ${esc(d.cusip || "")}`), fact("Fund style", esc(d.style)));
    } else if (e.kind === "etf") {
      r.push(fact("Fund", esc(e.actor.name)), fact("Manager", esc(e.actor.role)), fact("Change", esc(e.action_label)),
        fact("Traded on", esc(dateFmt(e.trade_date))), fact("Holdings file dated", esc(dateFmt(d.file_date))),
        fact("Shares traded (net of fund flow)", num(Math.abs(d.delta_shares || 0))), fact("Shares held now", num(e.shares)),
        fact("Change beyond fund flow", d.pct_change != null ? pct(d.pct_change) : e.action === "new" ? "New" : null),
        fact("Estimated value", money(e.value)), fact("Price used", px(e.price)),
        fact("Share of fund now", d.weight != null ? `${num(d.weight, 2)}%` : null), fact("Share of fund before", d.prev_weight != null ? `${num(d.prev_weight, 2)}%` : null),
        fact("Fund flow that day", d.flow_pct != null ? `${pct(d.flow_pct, 2)} (all positions scaled by this)` : null),
        fact("CUSIP", esc(d.cusip)), fact("Method", `<div class="small">${esc(d.method)}</div>`));
    } else if (e.kind === "activist") {
      r.push(fact("Filer", esc(e.actor.name)), fact("Schedule", `${esc(d.schedule)} <span class="faint">(${esc(d.form)})</span>`),
        fact("Event date", esc(dateFmt(e.trade_date))), fact("Filed", esc(dateFmt(e.filed_at))),
        fact("% of the company", d.percent != null ? `${num(d.percent, 2)}%` : null), fact("Shares", num(e.shares)),
        fact("Class", esc(d.class_title)), fact("CUSIP", esc(d.cusip)), fact("Source of funds", esc(d.funds_source)),
        d.persons && d.persons.length ? fact("Reporting persons", d.persons.map((p) => `${esc(p.name)}${p.percent != null ? ` — ${num(p.percent, 2)}%` : ""}`).join("<br>")) : "",
        d.purpose ? fact("Purpose of transaction", `<div class="small">${esc(d.purpose)}</div>`) : "",
        d.parsed === false ? fact("Note", "Older text-format filing; open it for details") : "");
    } else if (e.kind === "congress") {
      r.push(fact("Member", esc(e.actor.name)), fact("Chamber / party", esc(e.actor.role)), fact("Owner", esc(d.owner)),
        fact("Asset", esc(e.company)), fact("Asset type", esc(d.asset_type)), fact("Transaction", esc(e.action_label)),
        fact("Trade date", esc(dateFmt(e.trade_date))), fact("Disclosed", esc(dateFmt(e.filed_at))),
        fact("Disclosure delay", e.lag_days != null ? `${num(e.lag_days)} days${d.late ? ' <span class="warn">(late — limit is 45)</span>' : ""}` : null),
        fact("Amount (range)", esc(d.amount_text || amount(e))), fact("Description", esc(d.description)),
        d.unparsed ? fact("Note", "Scanned/paper filing — the app can't read it; open the original") : "");
    } else if (e.kind === "etoro") {
      r.push(fact("Investor", esc(e.actor.name)), fact("Win rate (closed trades)", d.win_ratio != null ? `${num(d.win_ratio, 1)}%` : null),
        fact("Gain over ranking period", d.gain_pct != null ? pct(d.gain_pct) : null), fact("eToro risk score", d.risk_score != null ? `${num(d.risk_score)}/10` : null),
        fact("Copiers", num(d.copiers)), fact("Position", esc(e.action_label)), fact("Opened", esc(dateFmt(d.opened_at))),
        fact("Open price", px(e.price)), fact("Leverage", d.leverage ? `x${num(d.leverage, 1)}` : null),
        fact("Share of their portfolio", d.investment_pct != null ? `${num(d.investment_pct, 2)}%` : null),
        fact("Asset type", esc(d.asset_type)), fact("Profit when closed", e.action === "close" && d.net_profit_pct != null ? pct(d.net_profit_pct) : null));
    }
    if (e.source && e.source.accession) r.push(fact("Accession #", `<span class="small">${esc(e.source.accession)}</span>`));
    return r.join("");
  }
  function riskPanel(t) {
    const q = S.quotes[t];
    if (!q) return `<div class="panel"><h3>Risk</h3><p class="small muted">No price data yet for ${esc(t || "this asset")}.</p></div>`;
    return `<div class="panel"><div class="row"><h3 class="grow">Risk level</h3><b>${q.risk_level ? `${num(q.risk_level)}/5 · ${esc(q.risk_label)}` : "—"}</b></div>
      <div style="margin:10px 0 8px">${riskBar(q.risk_level, true)}</div>
      <div class="row small faint" style="justify-content:space-between"><span>Lower risk</span><span>Speculative</span></div>
      <ul class="list small" style="margin-top:8px">${(q.risk_factors || []).map((f) => `<li>${esc(f)}</li>`).join("")}</ul>
      <div class="grid2 wide" style="margin-top:8px">
        <div class="stat"><b>${q.vol_1y != null ? pct(q.vol_1y * 100, 0, false) : "—"}</b><span>Volatility (1y)</span></div>
        <div class="stat"><b>${q.mdd_1y != null ? "-" + pct(q.mdd_1y * 100, 0, false) : "—"}</b><span>Max drop (1y)</span></div>
        <div class="stat"><b>${q.beta != null ? num(q.beta, 2) : "—"}</b><span>Beta vs S&amp;P</span></div>
        <div class="stat"><b>${money(q.mcap)}</b><span>Market cap</span></div></div>
      ${q.earnings ? (q.earnings.days != null && q.earnings.days < 0
        ? `<div class="small faint" style="margin-top:10px">Last earnings date on file: ${esc(dateFmt(q.earnings.date))}; the next one isn't announced yet.</div>`
        : `<div class="small ${q.earnings.days != null && q.earnings.days <= 10 ? "warn" : ""}" style="margin-top:10px">Next earnings: <b>${esc(dateFmt(q.earnings.date))}</b>${q.earnings.days != null ? (q.earnings.days === 0 ? " (today)" : ` (in ${num(q.earnings.days)} days)`) : ""}${q.earnings.time ? ` · ${esc(q.earnings.time)}` : ""}. Results can move a stock past any stop.</div>`) : ""}
      ${q.rel_volume ? `<div class="small${q.rel_volume.ratio >= 2.5 ? " warn" : ""}" style="margin-top:8px">Volume on ${esc(dateFmt(q.rel_volume.date))}: <b>${num(q.rel_volume.ratio, 1)}×</b> its 20-day average${q.rel_volume.ratio >= 2.5 ? " — unusually active" : ""}.</div>` : ""}
      ${q.short_volume ? `<div class="small${q.short_volume.avg20_pct != null && q.short_volume.pct >= q.short_volume.avg20_pct + 15 ? " warn" : ""}" style="margin-top:6px">Short selling on ${esc(dateFmt(q.short_volume.asof))}: <b>${pct(q.short_volume.pct, 0, false)}</b> of off-exchange volume${q.short_volume.avg20_pct != null ? ` (20-day average ${pct(q.short_volume.avg20_pct, 0, false)})` : ""} <span class="faint">· FINRA daily file; off-exchange trades only, so compare it with its own average, not with 50%</span></div>` : ""}
      ${q.short ? `<div class="small" style="margin-top:6px">Short interest: <b>${q.short.pct_shares != null ? pct(q.short.pct_shares, 1, false) + " of shares" : num(q.short.shares) + " shares"}</b>${q.short.days_to_cover != null ? `, ${num(q.short.days_to_cover, 1)} days to cover` : ""}${q.short.change_pct != null ? ` (${pct(q.short.change_pct, 0)} vs prior report)` : ""} <span class="faint">· FINRA, positions as of ${esc(dateFmt(q.short.asof))}</span></div>` : ""}
      <p class="small faint" style="margin:8px 0 0">Risk describes how bumpy the stock has been, not whether it will go up. No stock is “safe”.</p></div>`;
  }
  function sizingPanel(e) {
    if (e.direction !== "bullish" || !e.ticker) return "";
    const q = S.quotes[e.ticker];
    const z = sizing(q), c = z.cfg;
    const weak = !["strong", "notable"].includes(e.signal);
    return `<div class="panel"><div class="row"><h3 class="grow">Position-size ceiling</h3><a class="small" href="#/size">Edit rules</a></div>
      <div class="row" style="margin-top:8px"><div class="grow"><div class="price">${money(z.size)}</div>
        <div class="small muted">${num(z.pctOf, 1)}% of your ${money(c.portfolio)} portfolio, the most your own rules allow</div></div></div>
      <ul class="list checks small" style="margin-top:8px">
        <li>Risking ${num(c.risk, 2)}% of your portfolio (${money(c.portfolio * c.risk / 100)}) if it has a bad month (~${num(z.drop * 100, 0)}% drop${z.assumedVol ? ", assumed" : ""})</li>
        <li>Capped at ${num(Math.min(CAP_BY_LEVEL[z.level] || 3, c.satellite), 1)}% for a risk-${num(z.level)} stock</li>
        ${weak ? `<li>This signal is ${esc(SIG[e.signal] || "weak").toLowerCase()}: many investors would size below the ceiling or skip it</li>` : ""}
      </ul><p class="small faint" style="margin:6px 0 0">A rule-of-thumb calculator, not advice. Keep most money diversified (index funds); treat ideas like this as a small slice.</p></div>`;
  }
  // ------------------------------------------------------------------ Claude's analysis & measured predictability
  const STANCE = { favorable: ["Favorable setup", "fav"], watch: ["Worth watching", "wat"], neutral: ["No clear edge", "neu"], unfavorable: ["Unfavorable", "unf"] };
  const MODEL = { "claude-opus-5-5": "Claude Opus 5.5", "claude-fable-5-1": "Claude Fable 5.1", "claude-sonnet-5-5": "Claude Sonnet 5.5", "claude-haiku-5-5": "Claude Haiku 5.5" };
  const modelName = (m) => MODEL[m] || m || "Claude";
  const pc = (v, d = 0) => (v == null || isNaN(v) ? "—" : pct(v * 100, d, false));
  function stanceChip(a, withName) {
    if (!a || !a.stance) return "";
    const [l, c] = STANCE[a.stance] || [a.stance, "neu"];
    return `<span class="stance ${c}">${withName ? "Claude: " : ""}${esc(l)}${a.probability != null ? ` · ${pc(a.probability)}` : ""}</span>`;
  }
  function probBar(p, band, base) {
    const x = (v) => Math.max(0, Math.min(100, v * 100));
    return `<div class="pbar" role="img" aria-label="Probability ${pc(p)}${base != null ? `, base rate ${pc(base)}` : ""}">
      ${band && band.lo != null ? `<div class="pband" style="left:${x(band.lo)}%;width:${Math.max(1, x(band.hi) - x(band.lo))}%"></div>` : ""}
      <div class="pmid"></div>${base != null ? `<div class="pbase" style="left:${x(base)}%"></div>` : ""}
      ${p != null ? `<div class="pdot" style="left:${x(p)}%"></div>` : ""}</div>
      <div class="row small faint" style="justify-content:space-between"><span>0%</span><span>${base != null ? `<span class="warn">▮</span> base rate · ` : ""}50% = coin flip</span><span>100%</span></div>`;
  }
  const bullets = (title, arr, cls) => (arr && arr.length ? `<h3 style="margin-top:12px">${esc(title)}</h3><ul class="list ${cls}">${arr.map((x) => `<li>${esc(x)}</li>`).join("")}</ul>` : "");
  function sourcesList(src) {
    const ok = (src || []).filter((s) => safeUrl(s.url));
    return ok.length ? `<details class="srcs"><summary>Sources Claude read (${ok.length})</summary><ul class="list small" style="margin-top:6px">${ok.map((s) =>
      `<li><a href="${esc(safeUrl(s.url))}" target="_blank" rel="noopener noreferrer">${esc(s.title || s.url)}</a>${s.age ? ` <span class="faint">· ${esc(s.age)}</span>` : ""}</li>`).join("")}</ul></details>` : "";
  }
  function analysisBody(a) {
    return `<p style="margin:8px 0 6px"><b>${esc(a.headline)}</b></p><p style="margin:0 0 6px">${esc(a.thesis)}</p>
      ${a.probability != null ? `<div class="small muted" style="margin-top:8px">Claude's chance it beats the S&amp;P 500 over ~3 months: <b>${pc(a.probability)}</b> · ${esc(a.confidence || "low")} confidence${a.base_p != null ? ` · measured base rate ${pc(a.base_p)}` : ""}</div>${probBar(a.probability, null, a.base_p)}` : ""}
      ${a.flag ? `<p class="small warn" style="margin:6px 0 0">${esc(a.flag)}</p>` : ""}
      ${bullets("Bull case", a.bull_case, "pros")}${bullets("Bear case", a.bear_case, "cons")}
      ${a.catalysts && a.catalysts.length ? `<h3 style="margin-top:12px">Catalysts</h3><ul class="list">${a.catalysts.map((c) => `<li><b>${esc(c.when)}</b> · ${esc(c.what)}</li>`).join("")}</ul>` : ""}
      ${bullets("What to watch for", a.watch_for, "checks")}
      ${a.invalidation ? `<h3 style="margin-top:12px">What would prove it wrong</h3><p class="small" style="margin:4px 0 0">${esc(a.invalidation)}</p>` : ""}
      ${a.position_note ? `<h3 style="margin-top:12px">Position note</h3><p class="small" style="margin:4px 0 0">${esc(a.position_note)}</p>` : ""}
      ${a.key_facts_checked && a.key_facts_checked.length ? `<p class="small faint" style="margin:10px 0 0">Checked: ${a.key_facts_checked.map(esc).join("; ")}</p>` : ""}
      ${a.data_gaps && a.data_gaps.length ? `<p class="small faint" style="margin:4px 0 0">Couldn't confirm: ${a.data_gaps.map(esc).join("; ")}</p>` : ""}
      ${sourcesList(a.sources)}
      <p class="small faint" style="margin:10px 0 0">${esc(modelName(a.model))} · effort ${esc(a.effort || "—")}${a.tier === "deep" ? " · with web research" : ""} · ${esc(ago(a.made))}. Graded after ~3 months on the Records tab. Research, not personal financial advice.</p>`;
  }
  function analystPanel(e) {
    const a = e.analysis;
    if (a) return `<div class="panel ai section"><div class="row"><h2 class="grow">Claude's analysis</h2>${stanceChip(a)}</div>${analysisBody(a)}</div>`;
    const h = S.health || {}, an = h.analyst || {};
    const why = !(h.integrations || {}).claude_api ? "Add an Anthropic API key and Claude will research and comment on the strongest trades (setup guide)."
      : e.signal === "noise" ? "Claude skips routine paperwork such as grants and tax withholding."
      : (e.score || 0) < 50 ? "To keep costs down Claude analyses notable and strong signals and your watchlist; this one scored lower."
      : an.today && an.budget_usd && an.today.usd >= an.budget_usd * 0.95 ? "Today's Claude budget is used up; new signals are analysed again tomorrow."
      : "Queued: Claude's analysis usually appears within about 30 minutes of a new signal.";
    return `<div class="panel ai section"><h2>Claude's analysis</h2><p class="small muted" style="margin:6px 0 0">${esc(why)}</p></div>`;
  }
  function projectionHtml(pj) {
    if (!pj || !pj.p05 || !pj.p95 || pj.p95 <= pj.p05) return "";
    const lo = Math.log(pj.p05), hi = Math.log(pj.p95), x = (v) => ((Math.log(v) - lo) / (hi - lo)) * 100;
    return `<h3 style="margin-top:14px">Typical 3-month range</h3>
      <div class="rbar"><div class="r95"></div><div class="r68" style="left:${x(pj.p16)}%;width:${x(pj.p84) - x(pj.p16)}%"></div><div class="rnow" style="left:${x(pj.p50)}%"></div></div>
      <div class="row small" style="justify-content:space-between"><span>${px(pj.p05)}</span><span class="faint">${px(pj.p16)} – ${px(pj.p84)}</span><span>${px(pj.p95)}</span></div>
      <p class="small faint" style="margin:6px 0 0">If it keeps swinging like it has: 2 in 3 outcomes land between ${px(pj.p16)} and ${px(pj.p84)}, 9 in 10 between ${px(pj.p05)} and ${px(pj.p95)}. A range, not a call on direction.</p>`;
  }
  function bandText(v) {
    const b = v.band;
    if (!b || !b.n || v.source !== "model") return "";
    const ord = ["", "lowest", "second", "middle", "fourth", "highest"][b.fifth] || "";
    const who = b.fifth ? `filings the model ranked in the same (${ord}) fifth` : "filings with similar scores";
    return `<div class="faint">In the held-out test, ${who}: ${pc(b.obs)} beat the index (95% range ${pc(b.lo)}–${pc(b.hi)}, ${num(b.n)} cases)</div>`;
  }
  function predictPanel(e) {
    const v = e.quant, pj = e.projection;
    if (!v && !pj) return "";
    let body = "";
    if (v) {
      const br = v.base_rate;
      const src = v.source === "model" ? "Model tested on filings it had never seen" : v.source === "base_rate" ? "Historical base rate for this kind of signal" : "Not enough history for this kind of signal yet";
      body += `<div class="small muted" style="margin-top:6px">Chance the stock beats the S&amp;P 500 over the next ~3 months</div>
        ${v.p != null ? `<div class="row" style="margin:8px 0 0;gap:12px"><div class="price">${pc(v.p)}</div><div class="grow small">${esc(src)}${bandText(v)}</div></div>${probBar(v.p, v.band, br && br.n ? br.p : null)}` : `<p class="small" style="margin:6px 0 0">${esc(src)}.</p>`}
        <div class="small" style="margin-top:8px">Predictability: <b>${esc(v.predictability || "—")}</b>${v.auc ? ` · ranking accuracy ${num(v.auc, 3)} (0.5 = coin flip)` : ""}</div>
        ${br && br.n ? `<div class="small faint" style="margin-top:4px">${pc(br.p)} of past signals like this beat the S&amp;P 500 over 3 months (${num(br.n)} cases${br.lo != null ? `; 95% range ${pc(br.lo)}–${pc(br.hi)}, wide because signals in the same months move together` : ""}${br.mean_excess != null ? `; average ${pct(br.mean_excess * 100)} vs the index` : ""}).</div>` : ""}`;
    }
    body += projectionHtml(pj);
    return `<div class="panel section"><h3>Measured predictability</h3>${body}
      <p class="small faint" style="margin:8px 0 0">Same test every time: from the close after the filing to 63 trading days later, versus SPY over the same days. In markets, small edges are the norm and nothing is certain.</p></div>`;
  }

  // ------------------------------------------------------------------ Ideas & market
  const REG = { "Risk-on": "on", "Mixed": "mixed", "Risk-off": "off" };
  function marketCard() {
    const M = S.market || {};
    const r = M.regime;
    if (!r) return `<div class="panel"><h3>Market backdrop</h3><p class="small muted">Appears after the first refresh with price history.</p></div>`;
    const spy = M.spy || {}, br = M.breadth || {};
    const sectors = (M.sectors || []).slice(0, 17);
    const fred = Object.entries(M.fred || {});
    return `<div class="panel"><div class="row"><h3 class="grow" style="margin:0">Market backdrop</h3><span class="regime ${REG[r.label] || "mixed"}">${esc(r.label)}</span></div>
      ${bullets("Supportive", r.positives, "pros")}${bullets("Headwinds", r.negatives, "cons")}
      <div class="grid2 wide" style="margin-top:10px">
        <div class="stat"><b>${spy.sma200 ? pct((spy.price / spy.sma200 - 1) * 100, 1) : "—"}</b><span>S&amp;P vs 200-day</span></div>
        <div class="stat"><b>${spy.from_high != null ? pct(spy.from_high * 100, 1) : "—"}</b><span>From 52-week high</span></div>
        <div class="stat"><b>${br.above_200d != null ? pc(br.above_200d) : "—"}</b><span>Stocks above 200-day</span></div>
        <div class="stat"><b>${br.new_highs != null ? `${num(br.new_highs)} / ${num(br.new_lows)}` : "—"}</b><span>New highs / lows</span></div></div>
      ${sectors.length ? `<h3 style="margin-top:14px">Sectors &amp; assets, 3 months</h3><table class="lb" style="margin-top:6px"><tr><th>Name</th><th>1m</th><th>3m</th><th>vs S&amp;P</th></tr>
        ${sectors.map((x) => `<tr><td>${esc(x.name)} <span class="faint small">${esc(x.ticker)}</span></td><td class="${x.r1m >= 0 ? "up" : "down"}">${pct((x.r1m || 0) * 100, 1)}</td><td class="${x.r3m >= 0 ? "up" : "down"}">${pct(x.r3m * 100, 1)}</td><td class="${x.vs_spy_3m >= 0 ? "up" : "down"}">${pct(x.vs_spy_3m * 100, 1)}</td></tr>`).join("")}</table>` : ""}
      ${fred.length ? `<h3 style="margin-top:14px">Rates &amp; volatility</h3>${fred.map(([k, f]) => `<div class="kv"><span>${esc(f.label)}</span><b>${num(f.value, 2)}${f.change_1m != null ? ` <span class="small faint">(${f.change_1m >= 0 ? "+" : ""}${num(f.change_1m, 2)} 1m)</span>` : ""}</b></div>`).join("")}
        <p class="small faint" style="margin:6px 0 0">${esc(M.fred_notice || "")}</p>` : ""}
      <p class="small faint" style="margin:8px 0 0">${esc(r.method)} As of ${esc(dateFmt(M.asof))}.</p></div>`;
  }
  function briefCard() {
    const B = S.ideas && S.ideas.brief;
    if (!B || !B.brief) {
      const has = ((S.health || {}).integrations || {}).claude_api;
      return `<div class="panel ai section"><h2>Claude's market brief</h2><p class="small muted" style="margin:6px 0 0">${has ? "Claude writes a brief each weekday morning (about 7:45 am ET) with setups to watch." : "Add an Anthropic API key to get Claude's daily market brief and setups to watch (setup guide)."}</p></div>`;
    }
    const b = B.brief;
    const TYPE = { insider_conviction: "Insider conviction", smart_money_convergence: "Smart-money convergence", momentum: "Momentum", quality_at_discount: "Quality at a discount", event_driven: "Event-driven", turnaround: "Turnaround", other: "Other" };
    return `<div class="panel ai section"><div class="row"><h2 class="grow">Claude's market brief</h2><span class="small faint">${esc(dateFmt(B.day))}</span></div>
      <p style="margin:8px 0 0">${esc(b.regime_read)}</p>
      ${bullets("This week", b.what_matters_this_week, "checks")}${bullets("Risks", b.risks, "cons")}
      ${b.sector_view ? `<h3 style="margin-top:12px">Sectors</h3><p class="small" style="margin:4px 0 0">${esc(b.sector_view)}</p>` : ""}
      ${b.setups && b.setups.length ? `<h3 style="margin-top:14px">Setups to watch</h3>${b.setups.map((x) => `<div class="setup">
        <div class="row"><a class="tick" href="#/ticker/${encodeURIComponent(x.ticker)}">${esc(x.ticker)}</a><span class="small faint" style="margin-left:8px">${esc(x.name)}</span><span class="grow"></span>${x.probability != null ? `<span class="stance wat">${pc(x.probability)}</span>` : ""}</div>
        <div class="small muted" style="margin-top:2px">${esc(TYPE[x.setup_type] || x.setup_type)} · ${num(x.horizon_days)} days · ${esc(x.confidence)} confidence</div>
        <p class="small" style="margin:6px 0 0">${esc(x.thesis)}</p>
        ${x.why_now ? `<p class="small" style="margin:4px 0 0"><b>Why now:</b> ${esc(x.why_now)}</p>` : ""}
        <p class="small" style="margin:4px 0 0"><b>Trigger:</b> ${esc(x.trigger)}</p>
        <p class="small" style="margin:4px 0 0"><b>Wrong if:</b> ${esc(x.invalidation)}</p>
        ${x.risk_note ? `<p class="small faint" style="margin:4px 0 0">${esc(x.risk_note)}</p>` : ""}</div>`).join("")}` : ""}
      ${b.avoid && b.avoid.length ? `<h3 style="margin-top:14px">Would avoid</h3><ul class="list cons">${b.avoid.map((x) => `<li><b>${esc(x.ticker)}</b>: ${esc(x.reason)}</li>`).join("")}</ul>` : ""}
      ${b.bottom_line ? `<div class="note" style="margin-top:12px">${esc(b.bottom_line)}</div>` : ""}
      ${sourcesList(B.sources)}
      <p class="small faint" style="margin:10px 0 0">${esc(modelName(B.model))} · effort ${esc(B.effort || "—")} · ${esc(ago(B.made))}. Setup probabilities are graded after ~3 months on the Records tab. Research, not personal financial advice.</p></div>`;
  }
  function screenTrack(key, t, rel) {
    if (!t || !t.n) {
      const r = rel && rel.n ? ` ${esc(rel.label || "Closest measured pattern")}: ${pc(rel.p)} beat the S&amp;P 500 over 3 months (${num(rel.n)} cases${rel.lo != null ? `, 95% range ${pc(rel.lo)}–${pc(rel.hi)}` : ""}). A related rule, not this one, so no probability is shown for it.` : "";
      return `<p class="small faint" style="margin:4px 0 0">Track record of this exact screen: not enough history yet; the app is grading it as it goes.${r}</p>`;
    }
    if (t.unit === "months") return `<p class="small faint" style="margin:4px 0 0">Backtest: in ${num(t.n)} past months, ${pc(t.p)} of picks beat the S&amp;P 500 over the next 3 months${t.lo != null ? ` (95% range ${pc(t.lo)}–${pc(t.hi)})` : ""}${t.all_stocks != null ? `; all liquid stocks: ${pc(t.all_stocks)}` : ""}${t.mean_excess != null ? `; average ${pct(t.mean_excess * 100, 1)} vs the index${t.ex_lo != null ? ` (95% range ${pct(t.ex_lo * 100, 1)} to ${pct(t.ex_hi * 100, 1)})` : ""}` : ""}. Months overlap, so this is about ${num(Math.max(1, Math.floor(t.n / 3)))} independent 3-month periods.</p>`;
    return `<p class="small faint" style="margin:4px 0 0">Track record of this rule: ${pc(t.p)} beat the S&amp;P 500 over 3 months (${num(t.n)} cases${t.lo != null ? `, 95% range ${pc(t.lo)}–${pc(t.hi)}` : ""}).</p>`;
  }
  function ideaItem(it) {
    const a = it.analysis;
    return `<details class="idea"><summary><div class="row"><span class="tick">${esc(it.ticker)}</span><span class="small faint" style="margin-left:8px">${esc(it.company || "")}</span><span class="grow"></span>${a ? stanceChip(a) : it.risk_level ? riskBar(it.risk_level) : ""}</div>
      <div class="small muted" style="margin-top:3px">${esc(it.title)}</div></summary>
      <ul class="list small" style="margin-top:8px">${(it.why || []).map((w) => `<li>${esc(w)}</li>`).join("")}</ul>
      ${projectionHtml(it.projection)}
      ${a ? `<div class="panel ai" style="margin-top:10px"><h3>Claude's take</h3>${analysisBody(a)}</div>` : ""}
      <div class="btns"><a class="btn" href="#/ticker/${encodeURIComponent(it.ticker)}">Open ${esc(it.ticker)}</a><button class="btn" data-act="watch-t" data-t="${esc(it.ticker)}">${watch().tickers.includes(it.ticker) ? "★ Watching" : "☆ Watch"}</button></div></details>`;
  }
  function aiGaugeCard() {
    const G = (S.market || {}).ai_bubble;
    if (!G) return `<section class="sec"><h2>AI bubble gauge</h2><div class="panel"><p class="small muted">Appears once the app has about 15 months of whole-market price history (free Massive key).</p></div></section>`;
    const st = G.stats || {}, lv = ["Low", "Moderate", "Elevated", "High"], i = lv.indexOf(G.level);
    return `<section class="sec"><div class="sec-h"><h2>AI bubble gauge</h2>${window.SMRChips ? window.SMRChips.bubChip({ level: G.level }) : ""}</div><div class="panel">
      <div class="meter4" role="img" aria-label="AI theme ${esc(G.level)}">${lv.map((l, k) => `<span class="${k <= i ? "on l" + k : ""}"></span>`).join("")}</div>
      <div class="divlabels small muted"><span>Low</span><span>High</span></div>
      <p class="small"><b>${esc(G.phase)}.</b> An equal-weight basket of ${num(G.n)} AI stocks is ${st.net_2y != null ? `${pct(st.net_2y * 100, 0)} versus the S&amp;P 500 over two years` : st.net_1y != null ? `${pct(st.net_1y * 100, 0)} versus the S&amp;P 500 over a year` : "being measured"}${st.from_high != null ? ` and ${pct(st.from_high * 100, 0)} from its 52-week high` : ""}.</p>
      <div class="grid2 wide" style="margin-top:8px">
        <div class="stat"><b>${st.accel_share != null && st.accel_share >= 0 ? pct(Math.min(st.accel_share, 1.5) * 100, 0, false) : "—"}</b><span>Of the gain in 6 months</span></div>
        <div class="stat"><b>${G.breadth_above_200d != null ? pct(G.breadth_above_200d * 100, 0, false) : "—"}</b><span>Above 200-day average</span></div>
        <div class="stat"><b>${G.median_ev_s != null ? num(G.median_ev_s, 1) + "×" : "—"}</b><span>Median EV/sales</span></div>
        <div class="stat"><b>${G.gsy_crash_2y != null ? pct(G.gsy_crash_2y * 100, 0, false) : "—"}</b><span>Past crash rate at this run-up</span></div></div>
      ${(G.big_down_days || []).length ? `<h3>Big down days this month</h3><ul class="list small">${G.big_down_days.slice(0, 5).map((d) => `<li><a href="#/ticker/${encodeURIComponent(d.t)}">${esc(d.t)}</a> ${pct(d.move * 100, 1)} on ${esc(dateFmt(d.day))}</li>`).join("")}</ul>` : ""}
      <p class="small muted">${esc(G.about || "")}</p></div></section>`;
  }
  function convictionCard() {
    const C = (S.ideas || {}).conviction, rows = (C && C.rows) || [];
    const top = rows.filter((r) => r.rating === "top"), pos = rows.filter((r) => r.rating === "positive").slice(0, 8), warn = rows.filter((r) => r.rating === "caution" || r.rating === "red").slice(0, 8);
    const line = (r) => `<a class="card" href="#/ticker/${encodeURIComponent(r.ticker)}"><div class="row"><div class="grow"><span class="tick">${esc(r.ticker)}</span> <span class="small muted">${esc(r.name || "")}</span>
      <div class="small muted">${esc(r.passed)} of ${esc(r.of)} checks${r.why && r.why.length ? `: ${esc(r.why[0])}` : ""}</div></div>${window.SMRChips ? window.SMRChips.ratingChip(r) : ""}</div></a>`;
    return `<section class="sec"><h2>Top conviction</h2>
      ${top.length ? `<div class="rows">${top.map(line).join("")}</div>${(((S.fcAccuracy || {}).conviction_record) || {}).n ? "" : `<p class="small muted" style="margin:6px 2px">No Top conviction call has been graded yet, so this rating has no track record so far (first grades about three months after the first call).</p>`}` : `<div class="panel"><p class="small">No stock passes every check right now. That's normal: Top conviction needs Claude's favorable view, measured evidence, a sane valuation, no high bubble risk, liquidity and clean filings all at once.</p></div>`}
      ${pos.length ? `<h3 style="margin:14px 2px 6px">Positive</h3><div class="rows">${pos.map(line).join("")}</div>` : ""}
      ${warn.length ? `<h3 style="margin:14px 2px 6px">Caution and red flags</h3><div class="rows">${warn.map(line).join("")}</div>` : ""}
      <p class="small muted" style="margin:6px 2px">Graded against the S&amp;P 500 after 3 months; the record is on Records → Forecasts. Research, not personal advice.</p></section>`;
  }
  function renderIdeas() {
    const lists = (S.ideas && S.ideas.lists) || {};
    view.innerHTML = `<h1 class="page-title">Ideas &amp; market</h1>
      <p class="page-sub">The app's strongest calls, the AI-bubble gauge, the market backdrop and Claude's brief. Each screen shows its measured record.</p>
      <div class="menu"><a href="#/forecasts"><b>All forecasts</b><span>every stock with a price forecast, searchable</span></a></div>
      ${convictionCard()}${aiGaugeCard()}
      <section class="sec"><h2>Market</h2>${marketCard()}${briefCard()}</section>
      ${Object.entries(lists).map(([k, l]) => `<div class="panel section"><h3>${esc(l.title)}</h3><p class="small muted" style="margin:6px 0 0">${esc(l.about)}</p>${screenTrack(k, l.track, l.related)}
        <div style="margin-top:8px">${(l.items || []).map(ideaItem).join("") || `<div class="small faint">Nothing qualifies right now.</div>`}</div></div>`).join("")}
      <p class="disc">Screens are transparent rules with measured track records; Claude's notes are research, not personal financial advice. Past results don't guarantee future ones.</p>`;
  }

  // ------------------------------------------------------------------ Accuracy (Records tab)
  const RATE_LABEL = { insider_buy: "Insider purchases (all)", insider_buy_senior: "…by a CEO, CFO, Chair or President", insider_buy_cluster: "…with 3+ insiders buying within 30 days",
    insider_buy_weakness: "…after a 20%+ fall over 3 months", insider_buy_cluster_weakness: "…2+ insiders within 30 days after a 15%+ fall (the Ideas screen)", insider_buy_small: "…in thinly traded (smaller) companies", insider_buy_large: "…in heavily traded (larger) companies",
    insider_sell: "Insider sales", congress_buy: "Congress purchases" };
  const SRC_LABEL = { model: "Statistical model", base_rate: "Base rates", claude: "Claude's calls", idea: "Idea screens", none: "Tracked without a probability" };
  function calibRows(cal) {
    return (cal || []).map((c) => `<div class="calib"><span>Said ${pc(c.from)}–${pc(Math.min(c.to, 1))}</span><div class="track"><div class="obs" style="width:${Math.min(100, c.obs * 100)}%"></div><div class="pred" style="left:${Math.min(100, c.pred * 100)}%"></div></div><span>${pc(c.obs)} <span class="faint">n=${num(c.n)}</span></span></div>`).join("");
  }
  function renderAccuracy() {
    const A = S.accuracy || {}, m = A.model || {}, L = A.ledger || {}, rates = A.rates || {}, sc = A.screen || {};
    const model = m.status === "ok" ? `<div class="kv"><span>Verdict</span><b>${esc(m.predictability)}</b></div>
        <div class="kv"><span>Ranking accuracy, filings from the same week and size group (0.5 = coin flip)</span><b>${num(m.auc, 3)} <span class="small faint">(95% ${num((m.auc_ci || [])[0], 3)}–${num((m.auc_ci || [])[1], 3)})</span></b></div>
        <div class="kv"><span>Better than quoting the base rate (Brier skill)</span><b>${m.skill != null ? pct(m.skill * 100, 1) : "—"}</b></div>
        <div class="kv"><span>Model's top fifth beat the S&amp;P</span><b>${pc(m.top_fifth && m.top_fifth.beat_rate)} <span class="small faint">avg ${pct(((m.top_fifth || {}).mean_excess || 0) * 100, 1)}</span></b></div>
        <div class="kv"><span>Model's bottom fifth beat the S&amp;P</span><b>${pc(m.bottom_fifth && m.bottom_fifth.beat_rate)} <span class="small faint">avg ${pct(((m.bottom_fifth || {}).mean_excess || 0) * 100, 1)}</span></b></div>
        <div class="kv"><span>Trained on / tested on</span><b class="small">${num(m.n_train)} filings to ${esc(dateFmt((m.train_period || [])[1]))} / ${num(m.n_test)} later filings</b></div>
        <p class="small faint" style="margin:8px 0 0">${m.has_skill ? "It passed: better than the base rate on filings it never saw, in both halves of the test period. Its probabilities are shown on insider purchases." : "It did not show reliable skill on filings it never saw, so the app shows base rates instead of model probabilities. That's an honest result, not a bug."}${m.gate && m.gate.decided ? ` Decided ${esc(dateFmt(m.gate.decided))} for ${esc(m.gate.quarter)}; re-decided once a quarter (re-testing daily would give luck too many chances).${m.passed_today != null && m.passed_today !== m.has_skill ? ` Today's re-check ${m.passed_today ? "would pass" : "would not pass"}.` : ""}` : ""}</p>`
      : `<p class="small muted" style="margin:6px 0 0">${esc(m.note || "Builds after the daily job has the SEC insider data sets and a year of price history (free Massive key).")}</p>`;
    const rateRows = Object.entries(rates).map(([k, r]) => `<tr${r.n < 30 ? ' class="faint"' : ""}><td>${esc(RATE_LABEL[k] || k)}${r.n < 30 ? `<div class="small">too few cases to trust yet</div>` : ""}</td><td>${pc(r.p)}<div class="small faint">${pc(r.lo)}–${pc(r.hi)}</div></td><td class="${(r.mean_excess || 0) >= 0 ? "up" : "down"}">${r.mean_excess != null ? pct(r.mean_excess * 100, 1) : "—"}</td><td>${num(r.n)}</td></tr>`).join("");
    const bySrc = Object.entries(L.by_source || {}).filter(([k]) => k !== "none");
    const stance = Object.entries(L.claude_by_stance || {});
    view.querySelector("#acc").innerHTML = `
      <div class="panel"><h3>How predictable are insider purchases?</h3>${model}</div>
      <div class="panel section"><h3>Base rates: % that beat the S&amp;P 500 over 3 months</h3>
        ${rateRows ? `<table class="lb" style="margin-top:6px"><tr><th>Signal</th><th>Beat</th><th>Avg vs S&amp;P</th><th>n</th></tr>${rateRows}</table>` : `<p class="small muted">Builds from the SEC insider data sets and price history.</p>`}
        <p class="small faint" style="margin:6px 0 0">Most single stocks lag the index over short windows, so a base rate below 50% is normal. Ranges are 95% intervals that allow for signals in the same months moving together, so they are wide. Stocks that were delisted count, graded to their last trade.</p></div>
      <div class="panel section"><h3>Momentum screen backtest</h3>${sc.status === "ok" ? `<div class="kv"><span>Months tested</span><b>${num(sc.cohorts)} <span class="small faint">${esc(dateFmt((sc.period || [])[0]))}–${esc(dateFmt((sc.period || [])[1]))}</span></b></div>
        <div class="kv"><span>Picks that beat the S&amp;P</span><b>${pc(sc.beat_rate)} <span class="small faint">all stocks: ${pc(sc.all_stocks_beat_rate)}</span></b></div>
        <div class="kv"><span>Average vs the S&amp;P per 3 months</span><b>${pct(sc.mean_excess * 100, 1)}${sc.mean_excess_ci ? ` <span class="small faint">(95% ${pct(sc.mean_excess_ci[0] * 100, 1)} to ${pct(sc.mean_excess_ci[1] * 100, 1)})</span>` : ""}</b></div>` : `<p class="small muted">Needs about a year of whole-market price history.</p>`}</div>
      <div class="panel section"><h3>Live scorecard</h3><p class="small muted" style="margin:6px 0 8px">Every probability the app shows is written down when shown and graded 63 trading days later. ${num(L.graded || 0)} graded, ${num(L.pending || 0)} waiting.</p>
        ${bySrc.length ? bySrc.map(([k, s]) => `<div class="setup"><div class="row"><b class="grow">${esc(SRC_LABEL[k] || k)}</b><span class="small faint">n=${num(s.n)}</span></div>
          <div class="small" style="margin-top:4px">Beat the S&amp;P: ${pc(s.beat_rate)}${s.beat_lo != null ? ` <span class="faint">(95% ${pc(s.beat_lo)}–${pc(s.beat_hi)})</span>` : ""}${s.hit_rate != null ? ` · direction right: ${pc(s.hit_rate)}` : ""}${s.skill != null ? ` · Brier skill ${pct(s.skill * 100, 1)}` : ""}</div>
          ${s.calibration ? `<div style="margin-top:6px">${calibRows(s.calibration)}</div><p class="small faint" style="margin:2px 0 0">Bar = what happened; yellow tick = what was predicted.</p>` : ""}</div>`).join("")
          : `<p class="small faint">The first grades arrive about 3 months after setup. Until then this shows what's waiting.</p>`}
        ${stance.length ? `<h3 style="margin-top:12px">Claude's calls by stance</h3><table class="lb" style="margin-top:6px"><tr><th>Stance</th><th>Beat</th><th>Avg vs S&amp;P</th><th>n</th></tr>${stance.map(([k, s]) => `<tr><td>${esc((STANCE[k] || [k])[0])}</td><td>${pc(s.beat_rate)}</td><td>${pct((s.mean_excess || 0) * 100, 1)}</td><td>${num(s.n)}</td></tr>`).join("")}</table>` : ""}</div>`;
  }

  function recordPanel(e) {
    const r = e.actor_record;
    if (!r || !r.n) return "";
    const label = e.kind === "fund" ? "Copying their 13F" : "Their past buys (3 months after the filing)";
    return `<div class="panel"><h3>Track record</h3><p class="small muted" style="margin:6px 0 8px">${esc(label)}</p>
      <div class="grid2 wide">
        <div class="stat"><b>${r.win_rate != null ? pct(r.win_rate * 100, 0, false) : "—"}</b><span>Went up</span></div>
        <div class="stat"><b>${r.beat_rate != null ? pct(r.beat_rate * 100, 0, false) : "—"}</b><span>Beat S&amp;P 500</span></div>
        <div class="stat"><b>${r.avg_excess != null ? pct(r.avg_excess * 100) : "—"}</b><span>Avg vs S&amp;P</span></div>
        <div class="stat"><b>${num(r.n)}</b><span>Trades measured</span></div></div>
      ${r.n < 5 ? `<p class="small warn" style="margin:8px 0 0">Small sample: luck can easily explain this.</p>` : ""}</div>`;
  }
  function renderEvent(id) {
    const e = S.byId[id];
    if (!e) { view.innerHTML = `<a class="back" href="#/feed">‹ Feed</a><div class="empty">That trade is no longer in the feed.</div>`; return; }
    const q = S.quotes[e.ticker] || {};
    const since = q.price && e.price && e.kind !== "fund" ? ((q.price / e.price) - 1) * 100 : null;
    const [ico, kl] = KIND[e.kind] || ["•", e.kind];
    const meterColor = e.direction === "bearish" ? "var(--down)" : e.signal === "strong" ? "var(--up)" : e.signal === "notable" ? "var(--s2)" : "var(--warn)";
    const rd = e.read || { pros: [], cons: [], check: [] };
    view.innerHTML = `<button class="back" data-act="back">‹ Back</button>
      <div class="panel hero"><div class="kind">${esc(kl)}</div>
        <div class="row" style="margin-top:4px"><div class="grow"><a class="tick" href="#/ticker/${encodeURIComponent(e.ticker || "")}">${esc(e.ticker || "—")}</a>
          <div class="co">${esc(e.company || "")}</div></div>${scoreBox(e)}</div>
        <div class="row" style="margin-top:10px;gap:10px">${badge(e)}<span class="amt">${esc(amount(e))}</span></div>
        <div class="row" style="margin-top:10px"><div class="grow">${q.price ? `<span class="price">${px(q.price)}</span> <span class="small faint">${esc(q.price_src === "alpaca-iex" ? "IEX real-time (indicative)" : q.price_src === "yahoo" ? "delayed (Yahoo)" : "last close")}</span>` : ""}</div>
          ${since != null ? `<span class="${since >= 0 ? "up" : "down"}"><b>${pct(since)}</b> <span class="small">since their ${px(e.price)}</span></span>` : ""}</div>
        <a class="who" href="#/person/${encodeURIComponent(e.actor.id)}">${esc(e.actor.name)} <span class="faint">· ${esc(e.actor.role || "")}</span></a></div>

      ${analystPanel(e)}
      ${predictPanel(e)}
      <div class="panel section"><h2>Signal score</h2>
        <p style="margin:8px 0 10px">${esc(rd.summary)}</p>
        <div class="meter" aria-label="Signal ${num(e.score)} of 100"><div style="width:${Math.max(0, Math.min(100, +e.score || 0))}%;background:${meterColor}"></div></div>
        <p class="small muted" style="margin:6px 0 12px">${esc(SIG[e.signal] || "")} ${esc(e.direction || "")} signal · ${esc(e.score)}/100</p>
        ${rd.pros.length ? `<h3>Why it could matter</h3><ul class="list pros">${rd.pros.map((x) => `<li>${esc(x)}</li>`).join("")}</ul>` : ""}
        ${rd.cons.length ? `<h3 style="margin-top:10px">Why to be careful</h3><ul class="list cons">${rd.cons.map((x) => `<li>${esc(x)}</li>`).join("")}</ul>` : ""}
        ${rd.check.length ? `<h3 style="margin-top:10px">Before acting, check</h3><ul class="list checks">${rd.check.map((x) => `<li>${esc(x)}</li>`).join("")}</ul>` : ""}
        <p class="small faint" style="margin:10px 0 0">Rule-based: how strong this kind of disclosure has historically been, using published research. Not a prediction, and not a recommendation to buy or sell.</p></div>

      <div class="section">${riskPanel(e.ticker)}</div>
      <div class="section">${sizingPanel(e)}</div>
      <div class="section">${recordPanel(e)}</div>
      <div class="panel section"><h3>Trade details</h3><table class="facts" style="margin-top:6px">${factsFor(e)}</table>
        <div class="btns">${e.source && safeUrl(e.source.url) ? `<a class="btn primary" href="${esc(safeUrl(e.source.url))}" target="_blank" rel="noopener noreferrer">Open original filing ↗</a>` : ""}
          ${e.ticker ? `<button class="btn" data-act="watch-t" data-t="${esc(e.ticker)}">${watch().tickers.includes(e.ticker) ? "★ Watching" : "☆ Watch"} ${esc(e.ticker)}</button>` : ""}
          <button class="btn" data-act="share" data-id="${esc(e.id)}">Share</button>
          <a class="btn" href="#/ticker/${encodeURIComponent(e.ticker || "")}">Research ${esc(e.ticker || "")}</a>
          <a class="btn" href="#/ask/${encodeURIComponent(e.id)}">Ask Claude about this</a></div>
        <p class="small faint" style="margin:10px 0 0">Source: ${esc(e.source ? e.source.name : "")}</p></div>
      ${related(e)}`;
  }
  function related(e) {
    const same = S.feed.filter((x) => x.id !== e.id && x.ticker && x.ticker === e.ticker).slice(0, 6);
    return same.length ? `<div class="section"><h3>Other moves in ${esc(e.ticker)}</h3><div class="rows" style="margin-top:8px">${same.map(card).join("")}</div></div>` : "";
  }
  function renderTicker(t) {
    if (window.SMRStock) { window.SMRStock.render(view, t); return; }
    const evs = S.feed.filter((e) => e.ticker === t);
    const q = S.quotes[t] || {};
    const buys = evs.filter((e) => e.direction === "bullish").length, sells = evs.filter((e) => e.direction === "bearish").length;
    view.innerHTML = `<button class="back" data-act="back">‹ Back</button>
      <div class="panel hero"><div class="row"><div class="grow"><div class="tick">${esc(t)}</div><div class="co">${esc(q.name || (evs[0] && evs[0].company) || "")}</div></div>
        <div style="text-align:right">${q.price ? `<div class="price">${px(q.price)}</div>` : ""}<div class="small ${q.ret_3m >= 0 ? "up" : "down"}">${q.ret_3m != null ? pct(q.ret_3m) + " 3m" : ""}</div></div></div>
        <div class="grid2" style="margin-top:12px"><div class="stat"><b class="up">${buys}</b><span>Bullish moves</span></div><div class="stat"><b class="down">${sells}</b><span>Bearish moves</span></div></div>
        <div class="btns"><button class="btn" data-act="watch-t" data-t="${esc(t)}">${watch().tickers.includes(t) ? "★ Watching" : "☆ Watch"}</button></div></div>
      <div class="section">${riskPanel(t)}</div>
      <div class="section"><h3>All tracked moves</h3><div class="rows" style="margin-top:8px">${evs.map(card).join("") || `<div class="empty">None yet.</div>`}</div></div>`;
  }
  function renderPerson(id) {
    const evs = S.feed.filter((e) => e.actor.id === id);
    const a = evs[0] ? evs[0].actor : { id, name: id };
    const rec = evs.find((e) => e.actor_record) || {};
    view.innerHTML = `<button class="back" data-act="back">‹ Back</button>
      <div class="panel hero"><div class="tick" style="font-size:20px">${esc(a.name)}</div><div class="co">${esc(a.role || "")}</div>
        <div class="btns"><button class="btn" data-act="watch-p">${watch().people.some((p) => p.id === id) ? "★ Following" : "☆ Follow"}</button></div></div>
      ${rec.actor_record ? `<div class="section">${recordPanel(rec)}</div>` : ""}
      <div class="section"><h3>Recent moves</h3><div class="rows" style="margin-top:8px">${evs.map(card).join("") || `<div class="empty">No recent trades.</div>`}</div></div>`;
    view.querySelector('[data-act="watch-p"]').onclick = () => { toggleWatchPerson(a); renderPerson(id); };
  }
  function renderLeaders() {
    const L = S.leaders || {};
    const tab = LS.get("ltab2", "forecasts");
    const tabs = [["forecasts", "Forecasts"], ["accuracy", "Signals"], ["insiders", "Insiders"], ["congress", "Congress"], ["funds", "Funds"], ["etoro", "eToro"]];
    if (tab === "forecasts") {
      view.innerHTML = `<h1 class="page-title">How accurate are the forecasts?</h1>
        <p class="page-sub">Every forecast is written down when made and graded when its date arrives. Nothing is edited after the fact.</p>
        <div class="seg">${tabs.map(([k, l]) => `<button data-ltab="${k}" aria-pressed="${tab === k}">${l}</button>`).join("")}</div><div id="facc"></div>`;
      if (window.SMRAccuracy) window.SMRAccuracy.render(view.querySelector("#facc"));
      return;
    }
    if (tab === "accuracy") {
      view.innerHTML = `<h1 class="page-title">How good are these signals?</h1>
        <p class="small muted" style="margin:0 2px 12px">Measured by this app on data it had not seen, and graded live. Past results don't guarantee future ones.</p>
        <div class="seg">${tabs.map(([k, l]) => `<button data-ltab="${k}" aria-pressed="${tab === k}">${l}</button>`).join("")}</div><div id="acc"></div>`;
      renderAccuracy();
      return;
    }
    const rows = L[tab] || [];
    let body = "";
    if (tab === "etoro") {
      body = rows.length ? `<table class="lb"><tr><th>Investor</th><th>Win rate</th><th>Gain</th><th>Risk</th><th>Copiers</th></tr>
        ${rows.map((r) => `<tr><td><a href="#/person/${encodeURIComponent("etoro:" + r.username)}">${esc(r.name)}</a></td><td>${num(r.win_ratio, 0)}%</td>
        <td class="${r.gain_pct >= 0 ? "up" : "down"}">${pct(r.gain_pct, 0)}</td><td>${esc(r.risk_score)}</td><td>${num(r.copiers)}</td></tr>`).join("")}</table>
        <p class="small faint">From eToro's official API: win rate = share of profitable closed trades over the ranking period. Followed investors' new trades appear in the feed.</p>`
        : `<div class="empty">Add your eToro API keys to follow top eToro investors (see the setup guide).</div>`;
    } else if (tab === "funds") {
      body = rows.length ? `<table class="lb"><tr><th>Fund</th><th>Beat S&amp;P</th><th>Copy return</th><th>S&amp;P</th></tr>
        ${rows.map((r) => `<tr><td><a href="#/person/${encodeURIComponent(r.id)}">${esc(r.name)}</a><div class="small faint">${esc(r.manager || "")}</div></td>
        <td>${r.beat_rate != null ? pct(r.beat_rate * 100, 0, false) : "—"} <span class="faint small">of ${esc(r.quarters)}q</span></td>
        <td class="${r.cum_ret >= 0 ? "up" : "down"}">${pct(r.cum_ret * 100, 0)}</td><td class="faint">${pct(r.cum_spy * 100, 0)}</td></tr>`).join("")}</table>
        <p class="small faint">“Copy return”: buying each fund's top 13F positions the day after it filed and holding to the next filing, versus SPY over the same dates.</p>`
        : `<div class="empty">Fund backtests appear after the weekly track-record job runs.</div>`;
    } else {
      body = rows.length ? `<table class="lb"><tr><th>${tab === "congress" ? "Member" : "Insider"}</th><th>Up</th><th>Beat S&amp;P</th><th>Avg vs S&amp;P</th><th>n</th></tr>
        ${rows.map((r) => `<tr><td><a href="#/person/${encodeURIComponent(r.id)}">${esc(r.name || r.id)}</a><div class="small faint">${esc(r.ticker || r.role || "")}</div></td>
        <td>${pct(r.win_rate * 100, 0, false)}</td><td>${pct(r.beat_rate * 100, 0, false)}</td><td class="${r.avg_excess >= 0 ? "up" : "down"}">${pct(r.avg_excess * 100, 1)}</td><td>${esc(r.n)}</td></tr>`).join("")}</table>
        <p class="small faint">Each purchase is measured from the close the day after it became public to ~3 months (63 trading days) later. Ranked by a confidence-adjusted beat rate, so a lucky 3-for-3 doesn't outrank 30-for-45.${tab === "insiders" ? " Pre-planned (10b5-1) buys are excluded." : ""}</p>`
        : `<div class="empty">${tab === "insiders" ? "Insider win rates need a free Massive key for whole-market price history (setup guide), then appear after the weekly job." : "Win rates appear once enough disclosed buys are 3 months old."}</div>`;
    }
    view.innerHTML = `<h1 class="page-title">Who has the best record?</h1>
      <p class="small muted" style="margin:0 2px 12px">Computed by this app from official filings and prices. Past results don't guarantee future ones.</p>
      <div class="seg">${tabs.map(([k, l]) => `<button data-ltab="${k}" aria-pressed="${tab === k}">${l}</button>`).join("")}</div>
      <div class="panel">${body}</div>`;
  }
  function renderWatch() {
    const w = watch();
    const evs = S.feed.filter(isWatched);
    view.innerHTML = `<h1 class="page-title">Watchlist</h1><p class="page-sub">Tap a stock for its forecast, valuation, bubble risk and news.</p>
      <div class="panel"><label class="f" for="addt" style="margin-top:0">Add a ticker</label>
        <div class="toolbar"><input class="search" id="addt" placeholder="e.g. NVDA" autocapitalize="characters" autocomplete="off"><button class="btn" data-act="addt">Add</button></div>
        ${w.tickers.length ? `<div class="rows" style="margin:8px 0">${w.tickers.map((t) => { const q = S.quotes[t] || {}; return `<a class="card" href="#/ticker/${encodeURIComponent(t)}"><div class="row"><div class="grow"><span class="tick">${esc(t)}</span> <span class="small muted">${esc(q.name || "")}</span>${miniTags(q)}</div><div style="text-align:right">${q.price ? `<b>${px(q.price)}</b>` : ""}<div class="small ${q.ret_1m >= 0 ? "up" : "down"}">${q.ret_1m != null ? pct(q.ret_1m) + " 1m" : ""}</div></div></div></a>`; }).join("")}</div>` : ""}
        <div class="chips">${w.tickers.map((t) => `<button class="chip" aria-pressed="true" data-rm-t="${esc(t)}">${esc(t)} ✕</button>`).join("") || `<span class="small faint">No tickers yet</span>`}</div>
        <div class="chips">${w.people.map((p) => `<button class="chip" aria-pressed="true" data-rm-p="${esc(p.id)}">${esc(p.name)} ✕</button>`).join("") || `<span class="small faint">Follow people from their page</span>`}</div>
        <p class="small faint" style="margin:6px 0 0">For push alerts on these, add them to <code>watch_tickers</code> / <code>watch_people</code> in config.yaml (setup guide, step 6).</p></div>
      <section class="sec"><h2>Recent moves</h2>${evs.length ? `<div class="rows">${evs.map(card).join("")}</div>` : `<div class="empty">Nothing from your watchlist in the feed yet.</div>`}</section>`;
    $("#addt").addEventListener("keydown", (ev) => { if (ev.key === "Enter") addTicker(); });
  }
  function addTicker() { const t = $("#addt").value.trim().toUpperCase().replace(/[^A-Z0-9.]/g, ""); if (t) { toggleWatchTicker(t); renderWatch(); } }
  function renderSize() {
    const c = sizingCfg();
    view.innerHTML = `<button class="back" data-act="back">‹ Back</button><h1 class="page-title">Sizing rules</h1>
      <p class="small muted" style="margin:0 2px 12px">Stored only on this phone. Used to show a position-size ceiling on each bullish trade.</p>
      <div class="panel"><label class="f" for="pf" style="margin-top:0">Portfolio value ($)</label><input class="f" id="pf" type="number" inputmode="decimal" min="0" value="${esc(c.portfolio)}">
        <label class="f" for="sat">Max % of portfolio per single stock idea</label><input class="f" id="sat" type="number" inputmode="decimal" min="0" max="100" step="0.5" value="${esc(c.satellite)}">
        <label class="f" for="rk">How much of your portfolio you'd accept losing on one idea in a bad month</label>
        <select class="f" id="rk">${[[0.5, "0.5% (conservative)"], [1, "1% (balanced)"], [2, "2% (aggressive)"]].map(([v, l]) => `<option value="${v}"${c.risk == v ? " selected" : ""}>${l}</option>`).join("")}</select>
        <button class="btn primary full" style="margin-top:14px" data-act="save-size">Save</button></div>
      <div class="panel section"><h3>How the ceiling is worked out</h3><ol class="small muted" style="padding-left:18px;margin:8px 0 0">
        <li>Estimate a bad month for the stock: twice its typical monthly move (from 1-year volatility), at least 10%.</li>
        <li>Size so that a bad month costs at most your chosen % of the portfolio.</li>
        <li>Cap by risk level: 10% (level 1), 7.5% (2), 5% (3), 3% (4), 1.5% (5), and by your per-idea max.</li></ol>
        <p class="small faint" style="margin:10px 0 0">A common rule of thumb, not personal advice. Most wealth is built with a diversified core (index funds) plus small satellite bets.</p></div>`;
  }
  function renderStatus() {
    const h = S.health || {};
    const ints = h.integrations || {};
    const intLabel = { sec_contact_email: "SEC contact e-mail (required)", massive: "Massive price history (recommended)", alpaca: "Alpaca live prices (optional)", openfigi: "OpenFIGI key (optional)", etoro: "eToro API (optional)",
      fmp: "FMP (Senate fallback)", quiver: "Quiver (Senate fallback)", claude_api: "Claude analyst (Anthropic API key)", fred: "FRED macro data (optional)",
      alphavantage: "Alpha Vantage earnings dates (optional)", ntfy: "ntfy push alerts", telegram: "Telegram alerts" };
    const an = h.analyst || {}, lr = an.last_run || {};
    const analystCard = `<div class="panel section"><h3>Claude analyst</h3>${ints.claude_api ? `
        <div class="kv"><span>Model</span><b>${esc(modelName(an.deep_model))} · effort ${esc(an.deep_effort || "—")}</b></div>
        <div class="kv"><span>Spent today</span><b>${money((an.today || {}).usd || 0)} of ${money(an.budget_usd || 0)}</b></div>
        <div class="kv"><span>This month</span><b>${money(an.month_usd || 0)}</b></div>
        <div class="kv"><span>Analyses on file</span><b>${num(an.analyses || 0)} trades · ${num(an.briefs || 0)} briefs</b></div>
        <div class="kv"><span>Last run</span><b class="small">${lr.started ? esc(ago(lr.started)) : "not yet"}${lr.done != null ? ` · ${num(lr.done)} done` : ""}</b></div>
        ${(lr.errors || []).length ? `<ul class="list cons small" style="margin-top:6px">${lr.errors.slice(0, 4).map((x) => `<li>${esc(x)}</li>`).join("")}</ul>` : ""}
        <p class="small faint" style="margin:6px 0 0">Costs are computed from the API's own usage reports. Change the model, effort or daily budget in config.yaml (analyst section).</p>`
      : `<p class="small muted" style="margin:6px 0 0">Not set up. Add ANTHROPIC_API_KEY (setup guide) for Claude's per-trade analysis, idea write-ups and the morning brief.</p>`}</div>`;
    const srcRows = (h.sources || []).map((s) => {
      const mins = s.last_ok ? (Date.now() - new Date(s.last_ok)) / 60000 : null;
      const cls = s.error ? "bad" : mins == null ? "" : "ok";
      return `<div class="src"><span class="dot ${cls}" style="margin-top:6px"></span><div class="grow"><b>${esc(s.name)}</b>
        <div class="small muted">${esc(s.origin)}${s.official ? " · official" : ""}</div>
        <div class="small">${s.last_ok ? `Last OK ${esc(ago(s.last_ok))}` : "Not run yet"}${s.note ? ` · ${esc(s.note)}` : ""}</div>
        ${s.error ? `<div class="small down">${esc(s.error)}</div>` : ""}<div class="small faint">${esc(s.freshness)}</div></div></div>`;
    }).join("");
    const badFunds = (h.funds || []).filter((f) => f.error);
    const fundNotes = (h.funds || []).filter((f) => !f.error && f.info);
    view.innerHTML = `<button class="back" data-act="back">‹ Back</button><h1 class="page-title">Data, sources &amp; self-check</h1>
      <div class="panel"><div class="row"><div class="grow"><b>Last refresh</b><div class="small muted">${esc(dateFmt(h.generated_at))}, ${esc(ago(h.generated_at))}</div></div>
        <button class="btn" data-act="reload">Refresh and check</button></div></div>
      ${auditCard()}
      ${analystCard}
      <div class="panel section"><h3>Sources</h3>${srcRows || `<p class="small muted">No health data yet.</p>`}</div>
      <div class="panel section"><h3>Connected services</h3><ul class="list small" style="margin-top:6px">${Object.entries(intLabel).map(([k, l]) =>
        `<li>${ints[k] ? "✅" : "⚪️"} ${esc(l)}</li>`).join("")}</ul></div>
      ${badFunds.length ? `<div class="panel section"><h3>Fund list issues</h3><ul class="list cons small">${badFunds.map((f) => `<li>${esc(f.error)}</li>`).join("")}</ul></div>` : ""}
      ${(h.etf_funds || []).length ? `<div class="panel section"><h3>Active ETFs (daily files)</h3><ul class="list small" style="margin-top:6px">${h.etf_funds.map((f) => `<li>${f.error ? "⚠️" : f.last ? "✅" : "⚪️"} <b>${esc(f.ticker)}</b> ${esc(f.name || "")}${f.last && f.last.date ? ` · file dated ${esc(dateFmt(f.last.date))}, ${num(f.last.n)} holdings` : ""}${f.error ? ` · <span class="down">${esc(f.error)}${/HTTP|not found|Error/i.test(f.error) ? " — set the fund's “Full Holdings CSV” link in config.yaml" : ""}</span>` : ""}</li>`).join("")}</ul></div>` : ""}
      ${fundNotes.length ? `<div class="panel section"><h3>Fund notes</h3><ul class="list small">${fundNotes.map((f) => `<li><b>${esc(f.sec_name || f.cik)}</b>: ${esc(f.info)}</li>`).join("")}</ul></div>` : ""}
      <div class="panel section"><h3>About the scores</h3><p class="small muted" style="margin:6px 0 0">Scores weigh what research says makes disclosures informative:
        open-market buys over sales; senior executives; several insiders buying together; smaller companies; not pre-planned (10b5-1); concentrated long-term funds;
        leadership roles in Congress; and each person's own measured track record. Every point is explained on the trade's page.</p></div>
      <p class="disc">Not financial advice. Filings can contain errors and are delayed by law (insiders up to 2 business days, Congress up to 45 days, funds up to ~4.5 months). Always check the original filing.</p>`;
  }

  // ------------------------------------------------------------------ More, appearance, self-check
  const themePref = () => LS.get("theme", "auto");
  function applyTheme(t) {
    LS.set("theme", t);
    const root = document.documentElement;
    if (t === "light" || t === "dark") root.setAttribute("data-theme", t); else root.removeAttribute("data-theme");
    const dark = t === "dark" || (t === "auto" && window.matchMedia("(prefers-color-scheme: dark)").matches);
    document.querySelectorAll('meta[name="theme-color"]').forEach((m) => { m.setAttribute("content", dark ? "#0f1213" : "#edf0ef"); m.removeAttribute("media"); });
    const b = $("#theme-btn");
    if (b) b.setAttribute("aria-label", `Theme: ${t === "auto" ? "following your phone" : t}. Tap to switch.`);
  }
  function cycleTheme() {
    const dark = document.documentElement.getAttribute("data-theme") === "dark" ||
      (!document.documentElement.getAttribute("data-theme") && window.matchMedia("(prefers-color-scheme: dark)").matches);
    applyTheme(dark ? "light" : "dark");
    toast(dark ? "Light mode" : "Dark mode");
    if (viewName() === "more") renderMore();
    else refreshView();
  }
  function renderMore() {
    const t = themePref(), a = S.audit || {}, d = S.device || { counts: {} };
    const issues = (a.counts ? (a.counts.fail || 0) + (a.counts.warn || 0) : 0) + (d.counts.fail || 0) + (d.counts.warn || 0);   // device counts exclude the server line
    view.innerHTML = `<h1 class="page-title">More</h1>
      <section class="sec"><h2>Appearance</h2><div class="panel"><div class="seg" role="radiogroup" aria-label="Theme">
        ${[["auto", "Match phone"], ["light", "Light"], ["dark", "Dark"]].map(([k, l]) => `<button role="radio" data-theme="${k}" aria-checked="${t === k}">${l}</button>`).join("")}</div>
        <p class="small muted" style="margin:0">Saved on this phone. The moon button at the top switches quickly.</p></div></section>
      <section class="sec"><h2>Tools &amp; settings</h2><div class="menu">
        <a href="#/status"><b>Data, sources &amp; self-check</b><span>${issues ? `${issues} to look at` : "all clear"}</span></a>
        <a href="#/size"><b>Sizing rules</b><span>position-size ceiling</span></a>
        <a href="#/ask"><b>Claude settings</b><span>API key, model, effort</span></a>
        <a href="#/forecasts"><b>All forecasts</b><span>every stock with a price forecast</span></a>
        <a href="#/leaders"><b>Records</b><span>forecast accuracy, win rates</span></a></div></section>
      <section class="sec"><h2>About</h2><div class="panel"><p class="small">Smart Money Radar follows disclosed trades by insiders, funds, activists, Congress and active ETFs, then adds forecasts, valuation, bubble risk and news for each stock. Every probability it shows is written down and graded in the open. Research, not personal financial advice.</p>
        <p class="small muted">Headlines: GDELT Project. Filings: SEC EDGAR. Font: IBM Plex Sans (SIL Open Font License).</p></div></section>`;
  }
  // Device-side checks: run on every load and every manual refresh. The pipeline runs its own audit on the server.
  const errLog = [];
  window.addEventListener("error", (e) => { errLog.push({ at: new Date().toISOString(), msg: String(e.message || "error").slice(0, 200), src: (e.filename || "").split("/").pop() + ":" + (e.lineno || "") }); if (errLog.length > 20) errLog.shift(); });
  window.addEventListener("unhandledrejection", (e) => { errLog.push({ at: new Date().toISOString(), msg: String((e.reason && e.reason.message) || e.reason || "rejected promise").slice(0, 200), src: "promise" }); if (errLog.length > 20) errLog.shift(); });
  function selfCheck() {
    const checks = [], add = (status, msg) => checks.push({ status, msg });
    const age = S.meta && S.meta.generated_at ? (Date.now() - new Date(S.meta.generated_at)) / 60000 : null;
    if (age == null || isNaN(age)) add("fail", "No data loaded from the server yet");
    else if (age < -10) add("warn", `The data is stamped ${Math.round(-age / 60) || 1} hours in the future: this phone's clock (or the server's) is off, so ages shown in the app may be wrong`);
    else if (age > STALE_MIN) add("warn", `Data is ${age >= 120 ? `${Math.round(age / 60)} hours` : `${Math.round(age)} minutes`} old: the refresh workflow may be paused or delayed (GitHub pauses schedules after 60 days without activity)`);
    else add("pass", `Data refreshed ${Math.max(0, Math.round(age))} minutes ago`);
    const ids = new Set(); let dup = 0, bad = 0;
    for (const e of S.feed) { if (ids.has(e.id)) dup++; ids.add(e.id); if (!e.actor || !e.actor.name || !e.kind) bad++; }
    if (dup || bad) add("fail", `Feed problems: ${dup} duplicate and ${bad} malformed trades`); else add("pass", `${S.feed.length} trades, all well-formed`);
    let badPx = 0; for (const q of Object.values(S.quotes)) if (q.price != null && !(q.price > 0)) badPx++;
    if (badPx) add("fail", `${badPx} invalid prices`); else add("pass", `${Object.keys(S.quotes).length} prices valid`);
    const missing = [["leaders", S.leaders], ["health", S.health], ["market", S.market], ["ideas", S.ideas], ["signal accuracy", S.accuracy], ["forecast accuracy", S.fcAccuracy], ["server audit", S.audit]].filter(([, v]) => !v).map(([k]) => k);
    if (missing.length) add("warn", `Couldn't load: ${missing.join(", ")}`); else add("pass", "All data files loaded");
    if (S.meta && S.health && S.health.generated_at && Math.abs(new Date(S.meta.generated_at) - new Date(S.health.generated_at)) > 15 * 60000) add("warn", "Data files come from different refreshes (a partial update)");
    try { localStorage.setItem("smr.__t", "1"); localStorage.removeItem("smr.__t"); add("pass", "This phone's storage works (watchlist, settings)"); }
    catch { add("warn", "This phone's storage is blocked (private mode?): watchlist and settings won't be kept"); }
    if (!("serviceWorker" in navigator)) add("warn", "Offline mode isn't available in this browser");
    else if (navigator.serviceWorker.controller) add("pass", "Offline copy is active");
    if (!navigator.onLine) add("warn", "You're offline: showing the last saved data");
    for (const [n, ok] of [["charts", window.SMRCharts], ["stock pages", window.SMRStock], ["forecast records", window.SMRAccuracy], ["chat", window.SMRChat]]) if (!ok) add("fail", `The ${n} module didn't load: reload the app`);
    if (errLog.length) add("warn", `${errLog.length} script error${errLog.length > 1 ? "s" : ""} since opening: ${errLog[errLog.length - 1].msg}`);
    const a = S.audit;
    if (a && a.generated_at && S.meta && S.meta.generated_at && Math.abs(new Date(S.meta.generated_at) - new Date(a.generated_at)) > 15 * 60000)
      add("warn", "The server's self-check is from an earlier refresh than the data: the last run may have stopped before it finished");
    const counts = { pass: 0, warn: 0, fail: 0 }; for (const c of checks) counts[c.status]++;
    // the server's own result is shown beside these, not counted again
    if (a && a.overall) checks.push({ status: "info", msg: `Server self-check: ${a.counts.pass} passed, ${a.counts.warn} warnings, ${a.counts.fail} failures` });
    const flag = $("#audit-flag"); if (flag) flag.hidden = !(counts.fail || (a && a.counts && a.counts.fail));
    return { at: new Date().toISOString(), checks, counts };
  }
  function auditCard() {
    const a = S.audit, d = S.device || selfCheck();
    const ico = { pass: "✓", warn: "!", fail: "✕", info: "i" };
    const rowsOf = (list) => list.map((c) => `<div class="check-row"><span class="st ${esc(c.status)}" aria-label="${esc(c.status)}">${ico[c.status] || "?"}</span><div>${esc(c.msg)}${c.detail && Array.isArray(c.detail) ? `<div class="small muted">${c.detail.slice(0, 5).map((x) => esc(typeof x === "string" ? x : JSON.stringify(x))).join("; ")}</div>` : ""}</div></div>`).join("");
    const srv = a && a.checks ? a.checks.slice().sort((x, y) => ({ fail: 0, warn: 1, pass: 2 }[x.status] - { fail: 0, warn: 1, pass: 2 }[y.status])) : [];
    return `<section class="sec"><h2>App self-check</h2>
      <div class="panel"><h3>On this phone</h3><div class="audit-sum"><span><b>${esc(d.counts.pass)}</b> passed</span><span><b>${esc(d.counts.warn)}</b> warnings</span><span><b>${esc(d.counts.fail)}</b> failures</span></div>${rowsOf(d.checks.filter((c) => c.status === "fail" || c.status === "warn").concat(d.checks.filter((c) => c.status === "pass")))}</div>
      <div class="panel"><h3>On the server</h3>${a ? `<div class="audit-sum"><span><b>${esc(a.counts.pass)}</b> passed</span><span><b>${esc(a.counts.warn)}</b> warnings</span><span><b>${esc(a.counts.fail)}</b> failures</span><span class="muted">${esc(ago(a.generated_at))}</span></div>${rowsOf(srv)}` : `<p class="small muted">The server's audit appears after the next refresh.</p>`}
        <p class="small muted">Runs at the end of every refresh: data files, feed structure, price sanity, every forecast's shape, valuations, both report-card ledgers, source freshness, Claude's runs and spend, cache sizes, config, and whether the automated tests last passed. New failures send a push alert.</p></div></section>`;
  }

  // helpers shared with stock.js / accuracy.js (read-only)
  function sourcesListPub(src) { return sourcesList(src); }

  // ------------------------------------------------------------------ router & events
  function renderAsk() {
    const id = safeDecode(((location.hash || "").match(/^#\/ask\/?(.*)$/) || [, ""])[1] || "");
    if (window.SMRChat) window.SMRChat.render(view, id || null);
    else view.innerHTML = `<div class="empty">The chat module didn't load. Reload the app.</div>`;
  }
  function safeDecode(x) { try { return decodeURIComponent(x); } catch { return x; } }
  function viewName() { return ((location.hash || "#/feed").match(/^#\/([^/]+)/) || [, "feed"])[1]; }
  function route(keepScroll) {
    const h = location.hash || "#/feed";
    const [, name, arg] = h.match(/^#\/([^/]+)\/?(.*)$/) || [, "feed", ""];
    document.querySelectorAll(".tabs a").forEach((a) => {
      const on = a.dataset.tab === name || (name === "event" || name === "ticker" || name === "person") && a.dataset.tab === "feed" ||
        (name === "size" || name === "status") && a.dataset.tab === "more" || name === "forecasts" && a.dataset.tab === "ideas";
      a.classList.toggle("on", on);
      if (on) a.setAttribute("aria-current", "page"); else a.removeAttribute("aria-current");
    });
    const id = safeDecode(arg || "");
    if (!S.feed.length && !S.meta) return;
    S.renderedVersion = S.version;
    ({ feed: renderFeed, event: () => renderEvent(id), ticker: () => renderTicker(id), person: () => renderPerson(id), leaders: renderLeaders, ideas: renderIdeas,
      forecasts: () => (window.SMRStock && window.SMRStock.renderList ? window.SMRStock.renderList(view) : renderIdeas()),
      watch: renderWatch, size: renderSize, status: renderStatus, ask: renderAsk, more: renderMore }[name] || renderFeed)();
    if (name !== "feed" && !keepScroll) window.scrollTo(0, 0);
  }
  // after a redraw, keyboard focus goes back to the control that was pressed (the old one was replaced)
  const sameButton = (b) => { const a = [...b.attributes].filter((x) => x.name.startsWith("data-"));
    return a.length ? "button" + a.map((x) => `[${x.name}="${CSS.escape(x.value)}"]`).join("") : null; };
  document.addEventListener("click", (ev) => {
    const b = ev.target.closest("button");
    if (!b) return;
    const sel = sameButton(b), nth = sel ? [...document.querySelectorAll(sel)].indexOf(b) : -1;
    const back = () => {
      const lost = !document.activeElement || document.activeElement === document.body;
      if (sel && lost && !b.isConnected) { const all = document.querySelectorAll(sel), nb = all[nth] || all[0]; if (nb) nb.focus({ preventScroll: true }); }
    };
    let p;
    try { p = onButton(b); } finally { if (p && p.then) p.finally(back); else back(); }
  });
  function onButton(b) {
    const f = filters();
    if (b.dataset.kind) { LS.set("filters", { ...f, kind: b.dataset.kind }); S.shown = 60; renderFeed(); }
    else if (b.hasAttribute("data-strong")) { LS.set("filters", { ...f, strong: !f.strong }); renderFeed(); }
    else if (b.dataset.dir) { LS.set("filters", { ...f, dir: f.dir === b.dataset.dir ? "all" : b.dataset.dir }); renderFeed(); }
    else if (b.dataset.ltab) { LS.set("ltab2", b.dataset.ltab); renderLeaders(); }
    else if (b.dataset.rmT) { toggleWatchTicker(b.dataset.rmT); renderWatch(); }
    else if (b.dataset.theme) { applyTheme(b.dataset.theme); renderMore(); }
    else if (b.dataset.rmP) { const w = watch(); w.people = w.people.filter((p) => p.id !== b.dataset.rmP); LS.set("watch", w); renderWatch(); }
    else switch (b.dataset.act) {
      case "more": S.shown += 60; renderFeed(); break;
      case "back": if (history.length > ENTRY_DEPTH) history.back(); else location.hash = "#/feed"; break;   // never leave the app
      case "hide-install": LS.set("hideInstall", true); renderFeed(); break;
      case "watch-t": toggleWatchTicker(b.dataset.t); route(true); break;     // keep the reader's place
      case "addt": addTicker(); break;
      case "save-size": LS.set("sizing", { portfolio: Math.max(0, +$("#pf").value || 0), satellite: Math.min(100, Math.max(0, +$("#sat").value || 0)), risk: +$("#rk").value }); toast("Saved"); break;
      case "reload": return manualRefresh();
      case "share": { const e = S.byId[b.dataset.id]; const text = e ? `${e.read.summary} (signal ${e.score}/100)` : "";
        if (navigator.share) navigator.share({ title: "Smart Money Radar", text, url: location.href }).catch(() => {});
        else navigator.clipboard && navigator.clipboard.writeText(`${text} ${location.href}`).then(() => toast("Copied")); break; }
    }
  }
  async function manualRefresh() {
    toast("Checking for new data…");
    const changed = await loadAll(true);
    S.device = selfCheck();
    const c = S.device.counts, srv = (S.audit && S.audit.counts) || {};
    const fails = c.fail + (srv.fail || 0), warns = c.warn + (srv.warn || 0);
    if (changed || S.renderedVersion !== S.version || viewName() === "status") refreshView();
    toast(`${S.loadErr ? `Couldn't update (${S.loadErr.message})` : S.fromCache ? "Couldn't reach the server, so the saved copy is showing" : changed ? "Updated" : "You're up to date"}. Self-check: ${fails ? `${fails} problem${fails > 1 ? "s" : ""}` : warns ? `${warns} warning${warns > 1 ? "s" : ""}` : "all clear"}`);
  }
  // Re-draw the current screen with new data without yanking the page: keep the scroll position, and never
  // re-draw while someone is typing (search box, sizing form) — the next navigation shows the new data anyway.
  function refreshView() {
    const a = document.activeElement;
    if (a && /^(INPUT|SELECT|TEXTAREA)$/.test(a.tagName)) return;
    if (viewName() === "size" || viewName() === "ask") return;       // never re-draw a chat in progress
    const y = window.scrollY;
    route(true);
    window.scrollTo(0, y);
  }
  $("#refresh").addEventListener("click", manualRefresh);
  $("#theme-btn").addEventListener("click", cycleTheme);
  applyTheme(themePref());
  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => { if (themePref() === "auto") applyTheme("auto"); });
  window.addEventListener("hashchange", () => route());
  function debounce(fn, ms) { let t; return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); }; }

  // pull-to-refresh (feed only)
  let y0 = null;
  window.addEventListener("touchstart", (e) => { y0 = window.scrollY === 0 ? e.touches[0].clientY : null; }, { passive: true });
  window.addEventListener("touchend", (e) => { if (y0 != null && e.changedTouches[0].clientY - y0 > 90) manualRefresh(); y0 = null; }, { passive: true });

  // auto-refresh: check for a new version every 60 s while visible, and when the app comes back to the foreground
  async function poll() {
    if (document.visibilityState !== "visible" || !(await loadAll(true))) return;
    if (["ticker", "leaders", "forecasts"].includes(viewName())) toast("New data: tap “Updated” to refresh this page");   // don't redraw mid-read
    else refreshView();
  }
  setInterval(poll, 60000);
  document.addEventListener("visibilitychange", () => { if (document.visibilityState === "visible" && Date.now() - S.loadedAt > 30000) poll(); });
  setInterval(setUpdated, 30000);

  window.SMR = { data: () => S, watch,           // read-only access for the chat, stock and records modules
    u: { esc, money, num, pct, px, pxc, niceName, dateFmt, ago, safeUrl, card, riskBar, riskPanel, stanceChip, modelName, sourcesList: sourcesListPub,
      watch, toggleWatchTicker, toast, LS, sizing } };
  if ("serviceWorker" in navigator) navigator.serviceWorker.register("sw.js").catch(() => {});
  loadAll(false).then(() => route());
})();
