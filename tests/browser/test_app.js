// Browser test of the redesigned app: themes, the stock research page and its forecast chart, Records → Forecasts,
// the self-check on refresh, an on-device Claude forecast against the mock API, and no horizontal overflow.
const { chromium, devices } = require("playwright");
const assert = require("assert");
(async () => {
  const browser = await chromium.launch();
  const base = process.env.SITE || "http://127.0.0.1:8767/";
  const ctx = await browser.newContext({ ...devices["iPhone 13"], colorScheme: "dark" });
  const page = await ctx.newPage();
  await page.clock.setFixedTime(new Date("2026-11-02T13:40:00Z"));   // ten minutes after the demo data's refresh
  const errors = [];
  page.on("pageerror", (e) => errors.push("pageerror: " + e.message));
  page.on("console", (m) => { if (m.type() === "error") errors.push("console: " + m.text()); });
  const noOverflow = async (where) => {
    const [sw, iw] = await page.evaluate(() => [document.documentElement.scrollWidth, window.innerWidth]);
    assert(sw <= iw + 1, `${where}: page is wider than the screen (${sw} > ${iw})`);
  };
  const bg = () => page.evaluate(() => getComputedStyle(document.body).backgroundColor);

  // 1. theme follows the phone (dark), the toggle overrides it and survives a reload
  await page.goto(base + "#/feed"); await page.waitForTimeout(900);
  assert.strictEqual(await bg(), "rgb(15, 18, 19)", "dark by default on a dark phone");
  await page.click("#theme-btn"); await page.waitForTimeout(200);
  assert.strictEqual(await page.evaluate(() => document.documentElement.getAttribute("data-theme")), "light");
  assert.strictEqual(await bg(), "rgb(237, 240, 239)", "light after the toggle");
  await page.reload(); await page.waitForTimeout(700);
  assert.strictEqual(await bg(), "rgb(237, 240, 239)", "choice kept after reload (no flash: theme.js runs before paint)");
  assert.strictEqual(await page.$eval('meta[name="theme-color"]', (m) => m.content), "#edf0ef", "browser bar follows the chosen theme");
  await page.goto(base + "#/more"); await page.waitForTimeout(400);
  await page.click('[data-theme="auto"]'); await page.waitForTimeout(200);
  assert.strictEqual(await bg(), "rgb(15, 18, 19)", "back to following the phone");
  await noOverflow("feed/more");

  // 2. stock page: chart, horizon switch, tooltip, sections
  await page.goto(base + "#/ticker/NVDA"); await page.waitForTimeout(1200);
  assert(await page.$("#fan svg .viz-band.b80"), "forecast fan drawn");
  assert(await page.$("#fan svg .viz-price"), "price history drawn");
  const txt = await page.innerText("#view");
  for (const s of ["Forecast", "Peaks and dips", "Valuation", "Bubble risk", "News", "Rating", "When the 3-month low"]) assert(txt.includes(s), "stock page shows " + s);
  await page.click('[data-range="1y"]'); await page.waitForTimeout(500);
  assert((await page.innerText(".hz-title")).startsWith("In 1 year"), "horizon switch updates the summary");
  await page.focus('[data-range="1m"]'); await page.keyboard.press("Enter"); await page.waitForTimeout(300);
  assert.strictEqual(await page.evaluate(() => document.activeElement && document.activeElement.dataset.range), "1m", "keyboard focus stays on the horizon button after the redraw");
  assert(/^(So far|Backtest): held \d+% of \d+/.test(await page.innerText(".kpi .kpi-sub")), "measured coverage shown beside the 80% range");
  assert.strictEqual(await page.$eval('.tabs a[aria-current="page"]', (a) => a.dataset.tab), "feed", "the current tab is announced");
  await page.click('[data-range="1y"]'); await page.waitForTimeout(300);
  const box = await (await page.$("#fan svg")).boundingBox();
  await page.mouse.move(box.x + box.width * 0.85, box.y + box.height * 0.4); await page.waitForTimeout(200);
  const tip = await page.$eval("#fan .viz-tip", (e) => ({ hidden: e.hidden, text: e.textContent }));
  assert(!tip.hidden && /Median forecast/.test(tip.text), "crosshair tooltip shows the forecast: " + JSON.stringify(tip));
  await noOverflow("stock page");

  // 2b. a slow page for one stock never paints over the next one (the reviewer's race)
  await page.route("**/data/stock/NVDA.json*", async (route) => { await new Promise((r) => setTimeout(r, 1500)); await route.continue(); });
  await page.evaluate(() => { location.hash = "#/ticker/NVDA"; });
  await page.waitForTimeout(150);
  await page.evaluate(() => { location.hash = "#/ticker/MSFT"; });
  await page.waitForTimeout(2600);
  assert.strictEqual(await page.innerText("h1.ticker"), "MSFT");
  assert((await page.innerText("#stock-body")).includes("376.86"), "MSFT's own forecast shown, not NVDA's");
  await page.unroute("**/data/stock/NVDA.json*");
  // keyboard: arrow keys read the chart out loud
  await page.focus("#fan"); await page.keyboard.press("ArrowLeft"); await page.waitForTimeout(150);
  assert(/Close|Median/.test(await page.$eval("#fan .viz-live", (e) => e.textContent)), "chart values announced");
  await page.click("details.more summary"); await page.waitForTimeout(200);
  await noOverflow("method table open");
  await page.goto(base + "#/ticker/NVDA"); await page.waitForTimeout(1200);

  // 3. on-device Claude forecast through the mock API
  await page.evaluate(() => { localStorage.setItem("smr.chat.apiBase", JSON.stringify("http://127.0.0.1:8899/v1/messages")); localStorage.setItem("smr.chat.settings", JSON.stringify({ key: "sk-ant-test-123", model: "claude-opus-5-5", effort: "max", search: true, maxSearches: 3 })); });
  await page.reload(); await page.waitForTimeout(1200);
  // a slow answer: leave the page and come back while it runs; it carries on, can't be started twice, and lands
  let apiCalls = 0;
  await page.route("http://127.0.0.1:8899/**", async (route) => { apiCalls++; await new Promise((r) => setTimeout(r, 1500)); await route.continue(); });
  await page.click('[data-act="device-forecast"]'); await page.waitForTimeout(150);
  await page.evaluate(() => { location.hash = "#/ticker/MSFT"; }); await page.waitForTimeout(500);
  await page.evaluate(() => { location.hash = "#/ticker/NVDA"; }); await page.waitForTimeout(500);
  assert(await page.$eval('[data-act="device-forecast"]', (b) => b.disabled), "still running after coming back: the button stays off");
  assert(/Claude is/.test(await page.innerText("#devfc-status")), "progress shown after coming back");
  await page.$eval('[data-act="device-forecast"]', (b) => b.click());     // a second tap on a disabled button does nothing
  await page.waitForTimeout(3500);
  await page.unroute("http://127.0.0.1:8899/**");
  assert.strictEqual(apiCalls, 1, "one paid request, not two");
  const dev = await page.evaluate(() => JSON.parse(localStorage.getItem("smr.devfc") || "{}"));
  assert(dev.NVDA && dev.NVDA[0].horizons["3m"].median > 0 && dev.NVDA[0].ref_date, "device forecast stored: " + JSON.stringify(dev).slice(0, 200));
  assert((await page.innerText("#view")).includes("made on this phone"), "device forecast shown");
  assert(await page.$("#fan svg .viz-claude"), "Claude markers on the chart");
  assert(/^Done \(about \$/.test(await page.innerText("#devfc-status")), "cost and status survive the re-render");
  assert((await page.innerText("table.cmp")).includes("Combined"), "combined row computed for the phone's forecast");
  const spent = await page.evaluate(() => JSON.parse(localStorage.getItem("smr.chat.spend_total") || "0"));
  assert(spent > 0, "phone forecast cost added to the device total");

  // 4. Records → Forecasts (backtest data in the demo), charts drawn
  await page.goto(base + "#/leaders"); await page.waitForTimeout(1200);
  const rec = await page.innerText("#view");
  assert(rec.includes("80% ranges held") && rec.includes("Backtest"), "accuracy tiles render");
  assert(await page.$("#ch-cov svg"), "coverage trend drawn");
  assert(rec.includes("Claude forecasts made on this phone"), "device record section");
  await noOverflow("records");

  // 5. a failed update says so (toast and header) instead of "up to date"; then a refresh runs the self-check
  await ctx.route("**/data/meta.json*", (r) => r.fulfill({ status: 500, body: "server error" }));
  await page.click("#refresh"); await page.waitForTimeout(1200);
  assert(/Couldn't update/.test(await page.innerText("#toast")), "failed refresh reported: " + await page.innerText("#toast"));
  assert.strictEqual(await page.innerText("#updated"), "Update failed", "header keeps the failure");
  await ctx.unroute("**/data/meta.json*");
  await page.click("#refresh"); await page.waitForTimeout(1200);
  assert(/Self-check:/.test(await page.innerText("#toast")), "refresh reports the self-check");
  await page.goto(base + "#/status"); await page.waitForTimeout(600);
  const st = await page.innerText("#view");
  assert(st.includes("On this phone") && st.includes("On the server"), "self-check sections");
  for (const h of ["#/ideas", "#/watch", "#/ask", "#/size"]) { await page.goto(base + h); await page.waitForTimeout(500); await noOverflow(h); }

  // 6. offline: a stock never opened before says so, instead of "no data"
  await page.goto(base + "#/feed"); await page.waitForTimeout(1500);
  await page.reload(); await page.waitForTimeout(1500);              // now controlled by the service worker
  await ctx.setOffline(true);
  await page.click("#refresh"); await page.waitForTimeout(1500);
  assert(/saved copy/.test(await page.innerText("#toast")) && /saved copy/.test(await page.innerText("#updated")), "offline refresh says it's showing the saved copy");
  await page.evaluate(() => { location.hash = "#/ticker/BTC1"; }); await page.waitForTimeout(5000);
  assert((await page.innerText("#stock-body")).includes("offline"), "offline message for an unsaved stock");
  await ctx.setOffline(false);
  const errs = errors.filter((e) => !/Failed to load resource|net::ERR_INTERNET_DISCONNECTED|503|500/.test(e));   // expected while offline / in the failure test
  errors.length = 0; errors.push(...errs);

  console.log("ERRORS", JSON.stringify(errors));
  assert(errors.length === 0, "no console/page errors");
  console.log("APP TEST PASSED");
  await browser.close();
})().catch((e) => { console.error("APP TEST FAILED:", e.message); process.exit(1); });
