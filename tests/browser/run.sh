#!/usr/bin/env sh
# Browser tests of the chat box and the app (themes, stock page, records, self-check): builds a demo site, serves it, starts a mock Claude API, runs Playwright.
# Needs: python deps (requirements.txt + reportlab), node with playwright and chromium (npx playwright install chromium).
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
SITE=${SITE_DIR:-/tmp/smr-demo-site}
python3 "$HERE/demo_site.py" "$SITE"
(cd "$SITE" && python3 -m http.server 8767 --bind 127.0.0.1 >/dev/null 2>&1) & S1=$!
node "$HERE/mock_claude.js" >/dev/null 2>&1 & S2=$!
trap 'kill $S1 $S2 2>/dev/null' EXIT
sleep 2
rm -f "$HERE/mock_claude_requests.json"
SITE="http://127.0.0.1:8767/" node "$HERE/test_chat.js"
SITE="http://127.0.0.1:8767/" node "$HERE/test_app.js"
