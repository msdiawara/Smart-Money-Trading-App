// Browser test of the chat box against the mock API: setup, a question with a search pause, cost, sources, errors.
const { chromium, devices } = require("playwright");
const fs = require("fs");
const assert = require("assert");
(async () => {
  const browser = await chromium.launch();
  const ctx = await browser.newContext({ ...devices["iPhone 13"] });
  const page = await ctx.newPage();
  const errors = [];
  page.on("pageerror", (e) => errors.push("pageerror: " + e.message));
  page.on("console", (m) => { if (m.type() === "error" && !/401 \(Unauthorized\)/.test(m.text())) errors.push("console: " + m.text()); });   // the deliberate bad-key call
  const base = process.env.SITE || "http://127.0.0.1:8767/";
  const readReqs = () => { try { return JSON.parse(fs.readFileSync(require("path").join(__dirname, "mock_claude_requests.json"), "utf8")); } catch { return []; } };
  const offset = readReqs().length;            // the mock keeps a running log across test runs
  await page.goto(base + "#/feed"); await page.waitForTimeout(800);
  // point the chat at the mock API (only localhost overrides are honoured)
  await page.evaluate(() => { localStorage.setItem("smr.chat.apiBase", JSON.stringify("http://127.0.0.1:8899/v1/messages")); localStorage.setItem("smr.chat.settings", JSON.stringify({ key: "sk-ant-wrong", model: "claude-opus-5-5", effort: "max", search: true, maxSearches: 5 })); });
  await page.goto(base + "#/ask"); await page.waitForTimeout(500);
  assert((await page.innerText("#view")).includes("Ask Claude"), "chat screen renders");
  // a wrong key is reported plainly
  await page.fill("#chat-in", "hello"); await page.click("#chat-send"); await page.waitForTimeout(800);
  let t = await page.innerText("#chat-log");
  assert(t.includes("rejected (401)"), "401 explained: " + t);
  // fix the key in settings
  await page.click('[data-chat="settings"]'); await page.waitForTimeout(300);
  await page.fill("#ck-key", "sk-ant-test-123"); await page.selectOption("#ck-effort", "max"); await page.click("#ck-save"); await page.waitForTimeout(300);
  // ask from a trade page
  await page.goto(base + "#/event/" + encodeURIComponent("f4:0001999001-26-000011:P")); await page.waitForTimeout(500);
  // the Back button works under the content-security policy (it used to be a javascript: link, which CSP blocks)
  await page.click("button.back"); await page.waitForTimeout(400);
  assert(!/#\/event\//.test(await page.evaluate(() => location.hash)), "back leaves the trade page: " + await page.evaluate(() => location.hash));
  await page.goto(base + "#/event/" + encodeURIComponent("f4:0001999001-26-000011:P")); await page.waitForTimeout(500);
  await page.click("text=Ask Claude about this"); await page.waitForTimeout(500);
  assert((await page.innerText(".ctxline")).includes("ACME"), "focus shown in context line");
  await page.fill("#chat-in", "Is this insider buy worth following?"); await page.keyboard.press("Enter");
  await page.waitForTimeout(2500);
  t = await page.innerText("#chat-log");
  assert(t.includes("Favorable") && t.includes("three insiders bought"), "answer rendered: " + t.slice(0, 300));
  assert(!(await page.content()).includes("<script>alert(1)"), "model output is escaped");
  const logHtml = await page.innerHTML("#chat-log");
  assert(logHtml.includes("<table") && logHtml.includes("<pre><code>size &lt;= 5%"), "tables and fenced code render");
  assert(!/ACME won a contract this week\. <\/p>/.test(logHtml) || logHtml.includes("<h4>View</h4>"), "continued turn starts a new paragraph");
  assert(t.includes("Sources (2)"), "sources listed: " + t);
  const html = await page.innerHTML("#chat-log");
  assert(html.includes('href="https://news.example.com/acme"') && html.includes("read, not cited"), "cited vs read sources");
  assert(/1 search/.test(t), "search count shown");
  // cost: turn 1: 12000*4 + 9000*5 + 60*20 (+ $0.01 search) ; turn 2: 3000*4 + 9000*0.2 + 500*20 → per million
  const expected = (12000 * 4 + 9000 * 5 + 60 * 20 + 3000 * 4 + 9000 * 0.2 + 500 * 20) / 1e6 + 0.01;
  const m = t.match(/\$(\d+\.\d\d)/);
  assert(m && Math.abs(parseFloat(m[1]) - expected) < 0.006, `cost shown ${m && m[1]} vs expected ${expected.toFixed(4)}`);
  await page.screenshot({ path: process.env.SHOT_DIR ? process.env.SHOT_DIR + "/chat-answer.png" : "/tmp/chat-answer.png", fullPage: true });
  // what was sent
  const reqs = readReqs().slice(offset);
  const good = reqs.filter((r) => r.headers["x-api-key"] === "sk-ant-test-123");
  assert(good.length === 2, "two requests for the paused turn, got " + good.length);
  const first = good[0].body, second = good[1].body;
  assert(first.stream === true && first.output_config.effort === "max" && first.model === "claude-opus-5-5", "request shape");
  assert(first.max_tokens === 64000, "room for thinking at max effort");
  assert(/Asked at \d{4}-/.test(first.messages[first.messages.length - 1].content[0].text), "the current time lives in the user turn (the cached context stays stable)");
  assert(first.tools[0].type === "web_search_20250305" && first.tools[0].max_uses === 5, "web search tool");
  assert(first.system[1].cache_control && first.system[1].text.includes("FOCUS") && first.system[1].text.includes("ACME"), "app context with focus in system prompt");
  assert(first.system[0].text.includes("research, not personal financial advice"), "playbook");
  const cont = second.messages[second.messages.length - 1];
  assert(cont.role === "assistant" && cont.content[2].type === "web_search_tool_result" && cont.content[2].content[0].encrypted_content === "ENC1", "continuation returns search results unchanged");
  assert(cont.content[0].type === "thinking" && cont.content[0].signature === "SIG123" && cont.content[1].input.query === "ACME robotics news", "thinking + tool input rebuilt");
  assert(cont.content[3].citations[0].encrypted_index === "IDX1", "citations kept");
  assert(good[0].headers["anthropic-dangerous-direct-browser-access"] === "true", "browser header");
  // a second question carries the history as text only
  await page.fill("#chat-in", "and the bear case?"); await page.keyboard.press("Enter"); await page.waitForTimeout(2000);
  const reqs2 = readReqs().slice(offset).filter((r) => r.headers["x-api-key"] === "sk-ant-test-123");
  const h = reqs2[2].body.messages;
  assert(h.length === 3 && h[0].role === "user" && h[1].role === "assistant" && h[1].content[0].type === "text" && /and the bear case\?$/.test(h[2].content[0].text), "history as text turns");
  assert(reqs2[2].body.system[1].text === reqs2[0].body.system[1].text, "the app-context block is byte-identical between questions (cacheable)");
  // a stream that dies before the API says why is marked as possibly incomplete
  await page.fill("#chat-in", "cut this"); await page.keyboard.press("Enter"); await page.waitForTimeout(1500);
  t = await page.innerText("#chat-log");
  assert(t.includes("Half an answer") && t.includes("may be incomplete"), "dropped stream flagged: " + t.slice(-300));
  // Stop keeps what arrived so far
  await page.fill("#chat-in", "answer slowly"); await page.keyboard.press("Enter"); await page.waitForTimeout(1200);
  assert((await page.innerText("#chat-send")) === "Stop", "button turns into Stop while streaming");
  await page.click("#chat-send"); await page.waitForTimeout(600);
  t = await page.innerText("#chat-log");
  assert(/part \d+/.test(t) && t.includes("· stopped") && /at least <?\$/.test(t) && (await page.innerText("#chat-send")) === "Ask", "stopped answer kept and priced: " + t.slice(-200));
  const stored = await page.evaluate(() => localStorage.getItem("smr.chat.conv"));
  assert(!stored.includes("(Asked at"), "the time prefix never reaches the saved conversation");
  // New chat while streaming leaves no ghost error behind
  await page.fill("#chat-in", "answer slowly"); await page.keyboard.press("Enter"); await page.waitForTimeout(800);
  await page.click('[data-chat="new"]'); await page.waitForTimeout(800);
  const after = await page.evaluate(() => JSON.parse(localStorage.getItem("smr.chat.conv") || "{}"));
  assert((after.messages || []).length === 0, "new chat is empty after aborting: " + JSON.stringify(after).slice(0, 200));
  await page.fill("#chat-in", "Is this insider buy worth following?"); await page.keyboard.press("Enter"); await page.waitForTimeout(2500);
  await page.fill("#chat-in", "and the bear case?"); await page.keyboard.press("Enter"); await page.waitForTimeout(2000);
  // overload error mid-stream is shown, not swallowed
  await page.fill("#chat-in", "please overload"); await page.keyboard.press("Enter"); await page.waitForTimeout(1200);
  t = await page.innerText("#chat-log");
  assert(t.includes("overloaded"), "overload explained");
  // conversation survives a reload; New chat clears it
  await page.reload(); await page.waitForTimeout(800);
  assert((await page.innerText("#chat-log")).includes("Favorable"), "conversation persisted");
  await page.click('[data-chat="new"]'); await page.waitForTimeout(300);
  assert(!(await page.innerText("#chat-log")).includes("Favorable"), "new chat clears");
  assert.strictEqual(await page.evaluate(() => document.activeElement && document.activeElement.id), "chat-in", "New chat leaves the cursor in the text box");
  
  // other tabs still fine under the CSP
  for (const h2 of ["#/feed", "#/ideas", "#/leaders", "#/status"]) { await page.goto(base + h2); await page.waitForTimeout(400); }
  console.log("ERRORS", JSON.stringify(errors));
  assert(errors.length === 0, "no console/page errors");
  console.log("CHAT TEST PASSED");
  await browser.close();
})().catch((e) => { console.error("CHAT TEST FAILED:", e.message); process.exit(1); });
