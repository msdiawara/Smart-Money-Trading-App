// Mock of the Claude Messages API (streaming) for browser tests of the chat box. Records every request body.
const http = require("http");
const fs = require("fs");
const LOG = process.env.MOCK_LOG || require("path").join(__dirname, "mock_claude_requests.json");
const requests = [];
function sse(res, events) {
  res.writeHead(200, { "Content-Type": "text/event-stream", "Cache-Control": "no-cache", "Access-Control-Allow-Origin": "*" });
  for (const [ev, data] of events) res.write(`event: ${ev}\ndata: ${JSON.stringify(data)}\n\n`);
  res.end();
}
const CORS = { "Access-Control-Allow-Origin": "*", "Access-Control-Allow-Headers": "content-type, x-api-key, anthropic-version, anthropic-dangerous-direct-browser-access",
  "Access-Control-Allow-Methods": "POST, OPTIONS" };
http.createServer((req, res) => {
  if (req.method === "OPTIONS") { res.writeHead(204, CORS); res.end(); return; }
  let body = "";
  req.on("data", (c) => { body += c; });
  req.on("end", () => {
    let data = {};
    try { data = JSON.parse(body); } catch { /* ignore */ }
    requests.push({ headers: req.headers, body: data });
    fs.writeFileSync(LOG, JSON.stringify(requests, null, 1));
    if (req.headers["x-api-key"] !== "sk-ant-test-123") {
      res.writeHead(401, { ...CORS, "Content-Type": "application/json" });
      res.end(JSON.stringify({ type: "error", error: { type: "authentication_error", message: "invalid x-api-key" } }));
      return;
    }
    if (!req.headers["anthropic-dangerous-direct-browser-access"]) {
      res.writeHead(400, { ...CORS, "Content-Type": "application/json" });
      res.end(JSON.stringify({ type: "error", error: { type: "invalid_request_error", message: "missing browser access header" } }));
      return;
    }
    const textOf = (m) => !m ? "" : typeof m.content === "string" ? m.content : (m.content || []).filter((x) => x.type === "text").map((x) => x.text).join("\n");
    const lastUser = data.messages[data.messages.length - 1];
    const text = lastUser && lastUser.role === "user" ? textOf(lastUser) : "";
    if (/overload/i.test(text)) {
      sse(res, [["message_start", { type: "message_start", message: { id: "m0", type: "message", role: "assistant", model: data.model, content: [], usage: { input_tokens: 10, output_tokens: 1 } } }],
        ["error", { type: "error", error: { type: "overloaded_error", message: "Overloaded" } }]]);
      return;
    }
    if (/cut this/i.test(text)) {        // stream that dies after some text (no message_delta / message_stop)
      sse(res, [["message_start", { type: "message_start", message: { id: "m2", type: "message", role: "assistant", model: data.model, content: [], usage: { input_tokens: 10, output_tokens: 1 } } }],
        ["content_block_start", { type: "content_block_start", index: 0, content_block: { type: "text", text: "" } }],
        ["content_block_delta", { type: "content_block_delta", index: 0, delta: { type: "text_delta", text: "Half an answer" } }]]);
      return;
    }
    if (/slowly/i.test(text)) {           // trickles text for a few seconds so the reader can press Stop
      res.writeHead(200, { "Content-Type": "text/event-stream", "Cache-Control": "no-cache", "Access-Control-Allow-Origin": "*" });
      const w = (ev, d) => res.write(`event: ${ev}\ndata: ${JSON.stringify(d)}\n\n`);
      w("message_start", { type: "message_start", message: { id: "m3", type: "message", role: "assistant", model: data.model, content: [], usage: { input_tokens: 10, output_tokens: 1 } } });
      w("content_block_start", { type: "content_block_start", index: 0, content_block: { type: "text", text: "" } });
      let i = 0;
      const iv = setInterval(() => { i++; w("content_block_delta", { type: "content_block_delta", index: 0, delta: { type: "text_delta", text: `part ${i} ` } }); if (i >= 40) { clearInterval(iv); res.end(); } }, 100);
      res.on("close", () => clearInterval(iv));   // the client went away
      return;
    }
    const fcTool = (data.tools || []).find((t) => t.name === "submit_forecast");
    if (fcTool) {                         // on-device price forecast: answer with the tool call, streamed as JSON deltas
      const first = textOf(data.messages.find((m) => m.role === "user" && textOf(m).includes("{")));
      const payload = JSON.parse(first.slice(first.indexOf("{")));
      const p = payload.last_close;
      const row = (h, lo, mid, hi, pu) => ({ h, low_10: +(p * lo).toFixed(2), median: +(p * mid).toFixed(2), high_90: +(p * hi).toFixed(2), p_up: pu });
      const input = { horizons: [row("1w", 0.95, 1.004, 1.06, 0.52), row("1m", 0.88, 1.01, 1.15, 0.53), row("3m", 0.8, 1.03, 1.3, 0.55), row("1y", 0.6, 1.08, 1.75, 0.58)],
        thesis: "Mock device forecast thesis.", path_notes: "Choppy.", dip_risk: { window: "Into earnings", why: "Guidance", p_dip10_3m: 0.3 },
        peak_view: { window: "After earnings", why: "Beatable" }, key_dates: [{ date: "2026-11-19", event: "Earnings", impact: "Big" }],
        scenarios: [{ name: "bull", probability: 0.3, price_3m: p * 1.3, drivers: "x" }, { name: "base", probability: 0.4, price_3m: p, drivers: "y" }, { name: "bear", probability: 0.3, price_3m: p * 0.8, drivers: "z" }],
        news_digest: [{ date: "2026-10-30", headline: "Mock headline", why_it_matters: "Because", url: "https://news.example.com/x" }],
        stance: "watch", probability: 0.52, confidence: "low", risks: ["r1"], what_would_change_my_mind: "m" };
      const js = JSON.stringify(input);
      sse(res, [
        ["message_start", { type: "message_start", message: { id: "mf", type: "message", role: "assistant", model: data.model, content: [], stop_reason: null, usage: { input_tokens: 5000, output_tokens: 1 } } }],
        ["content_block_start", { type: "content_block_start", index: 0, content_block: { type: "tool_use", id: "toolu_1", name: "submit_forecast", input: {} } }],
        ["content_block_delta", { type: "content_block_delta", index: 0, delta: { type: "input_json_delta", partial_json: js.slice(0, 40) } }],
        ["content_block_delta", { type: "content_block_delta", index: 0, delta: { type: "input_json_delta", partial_json: js.slice(40) } }],
        ["content_block_stop", { type: "content_block_stop", index: 0 }],
        ["message_delta", { type: "message_delta", delta: { stop_reason: "tool_use", stop_sequence: null }, usage: { output_tokens: 900 } }],
        ["message_stop", { type: "message_stop" }]]);
      return;
    }
    const isContinuation = data.messages.length >= 2 && data.messages[data.messages.length - 1].role === "assistant";
    const base = { id: "msg_1", type: "message", role: "assistant", model: data.model, content: [], stop_reason: null, stop_sequence: null };
    if (!isContinuation && data.tools && data.tools.length) {
      // turn 1: thinking, a search, a result, a text with a citation, then pause
      sse(res, [
        ["message_start", { type: "message_start", message: { ...base, usage: { input_tokens: 12000, cache_creation_input_tokens: 9000, cache_read_input_tokens: 0, output_tokens: 1 } } }],
        ["content_block_start", { type: "content_block_start", index: 0, content_block: { type: "thinking", thinking: "", signature: "" } }],
        ["content_block_delta", { type: "content_block_delta", index: 0, delta: { type: "thinking_delta", thinking: "Let me check the news." } }],
        ["content_block_delta", { type: "content_block_delta", index: 0, delta: { type: "signature_delta", signature: "SIG123" } }],
        ["content_block_stop", { type: "content_block_stop", index: 0 }],
        ["content_block_start", { type: "content_block_start", index: 1, content_block: { type: "server_tool_use", id: "srvtoolu_1", name: "web_search", input: {} } }],
        ["content_block_delta", { type: "content_block_delta", index: 1, delta: { type: "input_json_delta", partial_json: "{\"query\": \"ACME robotics news\"}" } }],
        ["content_block_stop", { type: "content_block_stop", index: 1 }],
        ["content_block_start", { type: "content_block_start", index: 2, content_block: { type: "web_search_tool_result", tool_use_id: "srvtoolu_1", content: [
          { type: "web_search_result", url: "https://news.example.com/acme", title: "ACME wins contract", encrypted_content: "ENC1", page_age: "1 day ago" },
          { type: "web_search_result", url: "https://www.sec.gov/acme-8k", title: "ACME 8-K", encrypted_content: "ENC2", page_age: null }] } }],
        ["content_block_stop", { type: "content_block_stop", index: 2 }],
        ["content_block_start", { type: "content_block_start", index: 3, content_block: { type: "text", text: "", citations: [] } }],
        ["content_block_delta", { type: "content_block_delta", index: 3, delta: { type: "text_delta", text: "ACME won a contract this week. " } }],
        ["content_block_delta", { type: "content_block_delta", index: 3, delta: { type: "citations_delta", citation: { type: "web_search_result_location", url: "https://news.example.com/acme", title: "ACME wins contract", encrypted_index: "IDX1", cited_text: "ACME wins contract" } } }],
        ["content_block_stop", { type: "content_block_stop", index: 3 }],
        ["message_delta", { type: "message_delta", delta: { stop_reason: "pause_turn", stop_sequence: null }, usage: { input_tokens: 12000, cache_creation_input_tokens: 9000, cache_read_input_tokens: 0, output_tokens: 60, server_tool_use: { web_search_requests: 1 } } }],
        ["message_stop", { type: "message_stop" }],
      ]);
      return;
    }
    // final turn (or no tools): markdown answer with usage
    sse(res, [
      ["message_start", { type: "message_start", message: { ...base, usage: { input_tokens: 3000, cache_creation_input_tokens: 0, cache_read_input_tokens: 9000, output_tokens: 2 } } }],
      ["content_block_start", { type: "content_block_start", index: 0, content_block: { type: "text", text: "" } }],
      ["ping", { type: "ping" }],
      ["content_block_delta", { type: "content_block_delta", index: 0, delta: { type: "text_delta", text: "## View\n\n**Favorable**, about **55%** to beat the index (base rate 47%).\n\n- Bull: three insiders bought\n- Bear: guidance cut\n\n| Side | Odds |\n|---|---|\n| Bull | 55% |\n\n```\nsize <= 5% of portfolio\n```\n\nNot <script>alert(1)</script> advice." } }],
      ["content_block_stop", { type: "content_block_stop", index: 0 }],
      ["message_delta", { type: "message_delta", delta: { stop_reason: "end_turn", stop_sequence: null }, usage: { output_tokens: 500 } }],
      ["message_stop", { type: "message_stop" }],
    ]);
  });
}).listen(8899, "127.0.0.1", () => console.log("mock claude on 8899"));
