# Smart Money Radar

A phone app that tracks what **corporate insiders, hedge funds, activist investors, members of Congress, ARK's active
ETFs** (and optionally **eToro's top traders**) buy and sell, straight from the official filings and the funds' own daily
files, and puts an analyst next to every move. Every trade gets:

* **Claude's analysis** (with an Anthropic API key): Claude Opus 5.5 (or Fable 5.1) at the effort level you choose, up to
  `max`, researches the trade on the web (news, next earnings date, guidance, financing) and writes a stance
  (Favorable / Worth watching / No clear edge / Unfavorable), its probability that the stock beats the S&P 500 over ~3 months,
  the bull and bear case, catalysts with dates, what to watch for, what would prove it wrong, a sizing note and its sources.
* **Measured predictability**: how often signals like this beat the S&P 500 before (with a 95% range and sample size), and,
  for insider purchases, a model that's only used if it beat the base rate on filings it never saw.
* **A typical 3-month price range** from the stock's own volatility (a range, not a call on direction).
* **A signal score (0–100)** with plain-English reasons for and against, based on what published research says makes these
  disclosures informative (open-market buys over sales, senior executives, several insiders buying together, small companies,
  not pre-planned, concentrated long-term funds, the person's own track record…).
* **A risk meter (1–5)** from volatility, worst drop, beta, company size, profitability and liquidity.
* **A position-size ceiling** worked out from *your* rules (set on the Sizing tab, stored only on your phone).
* **All the basic facts**: who, role, dates, filing delay, shares, price, value, holdings after, 10b5-1 plan, footnotes, and a
  button to open the original filing.
* **Next earnings date, short interest, daily short-selling pressure and unusual volume** (FINRA, Massive), the most common
  sources of sudden gaps and a hint that something is going on.
* **Win rates**: the app measures how each insider's, member's and fund's past disclosed buys did over the next ~3 months vs. the S&P 500.

**Ask Claude** is a chat box inside the app (the *Ask* tab): any question, answered by Claude Opus 5.5 at maximum effort
(or Claude Fable 5.1) with web research and the app's live data as context: the day's signals, prices and risk, the measured
base rates and track records, the market backdrop and the Ideas screens. Open it from a trade's page and that trade is in the
conversation. It uses your own API key, kept only on your phone (see *Ask Claude* below).

Tap any stock for its **research page**:

* **Forecast chart** for the next day, week, month, 3 months and year: the likely range (50%, 80% and 90% bands), the middle
  outcome and the chance it ends higher, from a model that simulates 2,000 price paths with the stock's own volatility,
  fat tails and known events (earnings, Fed meetings, CPI). Claude's own forecast (researched on the web) is drawn on top, with
  a combined forecast weighted by each one's measured skill.
* **Peaks and dips:** the chance of touching +10% or −10% along the way, the typical worst drop, the riskiest day (usually
  earnings) and when the 3-month low tends to come. As probabilities and windows, because nobody can name the day of a peak.
* **Valuation:** cheap or expensive versus companies with similar growth, margins and industry (from SEC financial data), plus
  the growth rate the price assumes.
* **Bubble risk:** a checklist from research on past bubbles (big run-up, acceleration, volatility, issuance, valuation…), and
  how tightly the stock moves with the AI trade. The **Ideas** tab has an **AI bubble gauge** for the whole theme.
* **News:** material SEC 8-K events (with red flags such as restatements or delisting notices) and recent headlines, plus
  Claude's digest when it has forecast the stock.
* **Rating:** Top conviction, Positive, Neutral, Caution or Red flag. *Top conviction* needs six checks to pass at once
  (Claude's researched view, measured evidence, valuation, bubble risk, liquidity, no red-flag filings), and it has its own
  graded track record.

Plus an **Ideas** tab (market backdrop, the AI bubble gauge, Top conviction stocks, three transparent screens with measured track
records, and Claude's weekday morning brief with setups to watch, each with a trigger and an invalidation), and a **Records**
tab that grades every probability and every price forecast the app has ever shown, Claude's included, so you can see whether
it's getting better (see *How accurate is it?* below), and a **Results** tab: the forecasting tool's overall win rate, what
it expected against what happened, and every miss taken apart, with why it missed and what the app is changing because of it
(see *Results* below). Light and dark themes follow your phone, or pick one under *More*.
Every refresh runs a **self-check** of the data and the app.

It refreshes about every **5 minutes** (GitHub's minimum): new filings on EDGAR's live feed usually show up within
5–10 minutes of the SEC accepting them, the open app checks for new data every minute, and a once-a-day check against
the SEC's index catches anything the live feed missed. It pushes alerts to your phone for strong signals or anything on
your watchlist, and works offline from the Home Screen.

> Not financial advice. Filings are delayed by law (insiders up to 2 business days, Congress up to 45 days, funds up to ~4.5 months)
> and can contain errors. Nobody can reliably predict individual stocks: the best published models explain well under 1% of
> monthly return variation, so even good signals move the odds only a few points. The app shows you those odds, measured, and
> grades itself (see [`docs/METHODS.md`](docs/METHODS.md)).

Everything runs on free tiers: GitHub Actions + GitHub Pages, SEC EDGAR, the House Clerk, Massive's free plan, and Telegram/ntfy.
Optional paid add-ons: the Claude API for the analyst (you set a hard daily budget; default $3/day, and Claude Max 5x/20x
plans include $100/$200 a month of API credit); Quiver (~$25–30/mo) for reliable Senate trades.

---

## Setup (about 20 minutes, once)

### 1. Create the repository
1. Make a free account at **github.com**.
2. Create a **new public repository** (e.g. `smart-money-radar`). It must be *public* for free Pages hosting and free Actions minutes.
   The data it publishes is public filings; the sizing settings and watchlist you set in the app never leave your phone
   (alert watch-lists you type into `config/config.yaml` are part of the public repository).
3. Put these files in it:
   * **Easiest from a phone:** connect GitHub in Claude (Settings → Connectors → GitHub), then ask Claude to push the project
     to your new repository.
   * **From a computer:** unzip, then `git init && git add . && git commit -m init && git branch -M main && git remote add origin https://github.com/YOU/smart-money-radar.git && git push -u origin main`.
     (If you use the web uploader instead, make sure the hidden `.github` folder gets uploaded too.)

### 2. Turn on the website
Repository **Settings → Pages → Build and deployment → Source: GitHub Actions**.

### 3. Add your keys (Settings → Secrets and variables → Actions → *New repository secret*)

| Secret | Needed? | Where to get it |
|---|---|---|
| `SEC_CONTACT_EMAIL` | **Required** | Your personal e-mail. The SEC requires automated tools to identify themselves; it's only sent to sec.gov. Until it's set, the SEC sources wait (the Data tab says so) rather than contact the SEC anonymously |
| `MASSIVE_API_KEY` | **Recommended** | Free at massive.com (formerly Polygon.io) → Dashboard → API keys. Powers price history, risk meters and all win rates |
| `TELEGRAM_BOT_TOKEN` + `TELEGRAM_CHAT_ID` | Recommended for alerts | Telegram: message **@BotFather** → `/newbot` → copy the token. Send your new bot any message, then open `https://api.telegram.org/bot<TOKEN>/getUpdates` and copy `"chat":{"id": …}` |
| `NTFY_TOPIC` (+ `NTFY_TOKEN`) | Alternative alerts | Install the **ntfy** app, subscribe to a long random topic name (it acts as a password). The free tier is per-IP and GitHub's runners share IPs, so a free ntfy account token (`NTFY_TOKEN`) is more reliable |
| `ALPACA_KEY_ID` + `ALPACA_SECRET_KEY` | **Recommended** | Free paper account at alpaca.markets → API keys. Adds an indicative real-time price (IEX) and, once, every US stock's daily prices back to 2016 (with the Massive key too, delisted ones included), so the backtest and the learning reach back about ten years instead of two |
| `OPENFIGI_API_KEY` | Optional | Free at openfigi.com. Maps fund holdings to tickers ~10× faster |
| `ANTHROPIC_API_KEY` | Recommended | Turns on Claude's analysis, idea write-ups and the morning brief. Create a key at platform.claude.com. A claude.ai subscription doesn't include API usage, **but Max 5x/20x plans include $100/$200 a month of API credit**, claimed by linking a Console organization to your plan (Anthropic help article "Monthly API credits for Max and Team plans"). Web search must be allowed for your Console organization (it is by default). See *Claude analyst* below for costs |
| `FRED_API_KEY` | Optional | Free at fred.stlouisfed.org (My Account → API keys). Adds rates, the yield curve and VIX to the market backdrop |
| `ALPHAVANTAGE_API_KEY` | Optional | Free at alphavantage.co. Adds the next earnings date to every stock (one request a day) |
| `QUIVER_API_KEY` or `FMP_API_KEY` | Optional | Senate trades. The official Senate site blocks most cloud servers, so the app falls back to Quiver (paid, reliable) or FMP (check that your plan includes Senate data) |
| `ETORO_API_KEY` + `ETORO_USER_KEY` | Optional, off by default | See *eToro* below |
| `HEARTBEAT_URL` | Optional | A free check URL from healthchecks.io (or similar). Every refresh pings it, and the service alerts you if the refresh ever stops running, which nothing inside the app can notice |

### 4. First run
**Actions** tab → enable workflows if asked → **Refresh data & publish app** → **Run workflow** (type `all` in the box).
It takes about 5–10 minutes. After that it runs by itself every ~5 minutes.
(If a scheduled run is already going, yours waits; if it then shows *cancelled*, the next scheduled run took its place and does
the same work, because every job also runs on its own timer. Press **Run workflow** again only if you need `all` forced.)

* The live feed (insiders, sale notices, activists, Congress, funds) appears after the first run.
* ARK's daily trades start on the second trading day (the first run only stores the funds' current holdings).
* Price history backfills over the first few hours, so risk meters and win rates fill in progressively.
* Stock pages and forecasts appear once a stock's price history is stored; valuations need about 120 companies' SEC financials,
  which load over the first day (150 companies every half hour). The backtest gives the accuracy page numbers within a day.

### 5. Put it on your Home Screen
Open `https://YOUR-USERNAME.github.io/smart-money-radar/` in **Safari** → Share → **Add to Home Screen**
(Android/Chrome: menu → *Install app*). It opens full-screen like a normal app and keeps the latest data offline.

### 6. Make it yours: `config/config.yaml`
Edit it on github.com (pencil icon). Saving triggers a refresh.
* `alerts.watch_tickers` / `alerts.watch_people`: always alert on these.
* `alerts.min_score`: alert threshold (70 = strong signals only).
* `funds.list`: the superinvestors to follow (each CIK is checked against the SEC's records every run).
* `etf.funds`: the transparent active ETFs whose daily holdings files are turned into trades (ARK's six by default).
* `congress.leadership_extra`: party leaders (Speaker, floor leaders, whips, caucus chairs) are detected automatically from the
  @unitedstates legislators data by their Bioguide ID, refreshed daily. Add anyone else's Bioguide ID here.
* `bubble.core` / `bubble.platform` / `bubble.power`: the AI basket behind the AI bubble gauge and each stock's AI exposure.
* `analyst.forecasts_per_day`: how many stocks Claude forecasts each day (your watchlist goes first).
* `forecast`, `valuation`, `conviction`: the model's assumptions and the Top conviction thresholds, explained in place.

---

## How it works
```
Refresh workflow (every ~5 min) ──► pipeline/ (Python)
   SEC EDGAR live feed ─ Form 4 · Form 144 · 13D/13G · 13F ─┐
   House Clerk PTR PDFs · Senate eFD (or Quiver/FMP)        │   score · risk · win rates · base rates
   ARK daily holdings files (trades = day-over-day change)  ├─► tested model · price ranges · ideas  ─► web/data/*.json ─► GitHub Pages ─► your phone
   Massive EOD history · Alpaca/Yahoo latest price          │   forecast ledger (grading)                └─► Telegram / ntfy alerts
   FINRA short interest + daily short volume · earnings · FRED
   SEC company facts (financials) · 8-K events · GDELT news │   valuation · bubble checklist · price forecasts
   Claude's latest analyses and forecasts (analyst cache)  ─┘   grading · learning · self-check (audit.json)

Analyst workflow (every ~20 min) ──► Claude (Opus 5.5 / Fable 5.1, effort up to max, web search) ──► analyses, brief,
                                     price forecasts, post-mortems of missed forecasts (lessons) ──► cache
Ask tab (in the app) ──► your browser calls the Claude API directly with your own key (nothing passes through a server)
```
* **Data sources, freshness and the audit trail:** see [`docs/DATA_SOURCES.md`](docs/DATA_SOURCES.md).
* **How predictions, the model, the ledger and Claude's analysis work:** see [`docs/METHODS.md`](docs/METHODS.md).
* **Health:** the app's **Data** tab shows each source's last success, errors and what's connected.
* **Tests:** `pip install -r requirements.txt pytest reportlab && python -m pytest`. They include parsers run against real
  filings, the model's planted-signal and pure-noise checks, the analyst against a simulated Claude API, the sources end to end,
  the forecast's coverage on simulated fat-tailed prices, the walk-forward learning on pure noise, the valuation maths and the
  self-check. Browser tests (`sh tests/browser/run.sh`) drive the app in a real browser: themes, the stock page and its chart,
  an on-device Claude forecast and the chat box against a mock of the streaming API, the records page and the self-check.
  They run on every push and every night, and the app's self-check reports the latest result.

## Claude analyst
A separate workflow (*Claude analyst*) runs every ~20 minutes and spends at most `analyst.daily_budget_usd` a day.

| What | When | Model / effort (defaults) | Typical cost each* |
|---|---|---|---|
| Deep analysis with web research | strong signals (score 70+) and anything on your watchlist | Opus 5.5, `max` | ~$0.40–1.20 |
| Quick read (no web research) | notable signals (50–69) | Opus 5.5, `low` | ~$0.03–0.08 |
| Idea write-ups | up to 3 a day from the Ideas screens | Opus 5.5, `max` | ~$0.40–1.20 |
| Market brief + setups to watch | weekdays 7:45 am ET | Opus 5.5, `max` | ~$0.50–1.50 |
| Price forecasts (1 week to 1 year, with news research) | up to 3 a day, watchlist first, each stock at most weekly | Opus 5.5, `high` | ~$0.20–0.60 |
| Post-mortems of missed forecasts (lessons) | up to 2 a day, once forecasts have been graded | Opus 5.5, `medium` | ~$0.05–0.20 |

\*Estimates from list prices ($4/$20 per million tokens for Opus 5.5, $10/$50 for Fable 5.1, $10 per 1,000 searches);
the Data tab shows what each analysis actually cost from the API's own usage reports. Change the model (`claude-fable-5-1`
costs 2.5x), effort, thresholds and budget in `config/config.yaml` under `analyst`. Anything over the budget waits for tomorrow
(deep requests fall back to quick reads first), each analysis has its own cost cap, and every billed call is counted even
when it fails, so the budget is the real spend. Below the thresholds, trades still get the signal score, measured
predictability and price range, which cost nothing.

Claude is told to anchor on the measured base rate, adjust only for evidence it can name, give both sides, and never present a
view as certain. Its probabilities are recorded the moment they're shown and graded after ~3 months on the Records tab, so you
can see whether listening to it would have helped.

## Ask Claude (the chat box)
Open the **Ask** tab, paste a Claude API key once, and ask anything. The default is Claude Opus 5.5 at effort `max` with up to
5 web searches per answer; Claude Fable 5.1 and lower efforts are in Settings. Every answer shows its sources and what it cost.

* **Where your key lives:** only in your browser's local storage on that device. The page sends it straight to
  `api.anthropic.com` over HTTPS (with Anthropic's header that permits direct browser calls); the app has no server, and a
  content-security policy stops the page from talking to any other site. "Forget key & chats" in Settings wipes it.
  One caveat of free GitHub Pages hosting: every project site of one account shares the origin `YOU.github.io`, so
  JavaScript in another project you host under the same account could read that storage. Keep other projects trustworthy,
  use a custom domain, or set a spending limit on the key in the Claude Console (recommended anyway).
* **What Claude sees:** the app's current data (strongest and newest signals with their scores, measured odds and Claude's
  existing views; prices and risk; base rates and the live scorecard; market backdrop; Ideas and the latest brief; your
  watchlist) plus, if you opened the chat from a trade, that trade's full record. Previous turns of the same chat are sent
  back as text.
* **Cost:** billed to your key at list prices (Opus 5.5 $4/$20 per million input/output tokens, Fable 5.1 $10/$50, web search
  $10 per 1,000). The app data is cached by the API, so a typical question costs a few cents to tens of cents; a long,
  heavily researched answer at max effort (which can think for tens of thousands of tokens) can cost a dollar or two. The
  chat shows the running total, and a spending limit on the key in the Claude Console is a sensible backstop.
* It's research, not personal financial advice, and it can be wrong: check its sources and the original filings.

## How accurate is it? (Records → Forecasts)
Every price forecast is written down when it's made, never revised, and graded when its date arrives. The page shows, for each
horizon and for the model, Claude, the combined forecast and the backtest, with a trend over time:

* **Did the ranges hold?** The 80% range should contain the outcome about 80% of the time. This is the target the app holds
  itself to (80%+, not 70%), and it is achievable: below target means the ranges were too narrow, so the next forecasts are
  widened; above it, narrowed.
* **Did it beat a random walk?** A proper scoring rule compares every forecast with a naive one centred on today's price. Above
  zero means real skill; expect a few percent at best.
* **Was the direction right?** Shown next to the "always say up" baseline. Expect roughly 50–56%. No forecaster, human or
  machine, gets 80–90% of stock directions right over weeks or months; anyone claiming that is overfitting or selling something.
  So the app aims its 80% target at what can be measured honestly (range coverage and skill) and shows direction as it is.

**How it learns from its mistakes:**
* **Range calibration:** widths are re-calibrated per horizon from graded results (from a backtest on day one, then from live
  grades). Evidence is counted in independent months, and a correction is used only if it would have improved the newest
  forecasts it wasn't fitted on.
* **Range learner (machine learning):** gradient-boosted trees test whether the right range width depends on the stock's
  situation (volatility rising or falling, a recent big move, an earnings report in the window, and how often that stock's
  own ranges have missed before). They're used at a horizon only if they beat the single width on recent forecasts they
  weren't fitted on.
* **Coverage keeper (adaptive conformal prediction):** after each day's results it widens the next ranges a little when
  more than 20% missed their 80% range, and narrows them when fewer did, so coverage stays near 80% when markets change.
  It's used only where replaying it on the live record beat leaving the ranges alone.
* **Signal learner (machine learning, a weekly competition):** nine candidates compete to predict which stocks will beat
  their forecast relative to the others: ridge regression, robust (Huber) regression, spline regression, gradient-boosted
  trees, a random forest, CatBoost, a neural network, the average of all of them, and the **fused model** (below). Each
  catches a different kind of pattern: straight lines, lines that ignore a few huge outliers, effects that bend,
  thresholds and combinations. They use the app's signals: momentum, recent moves, volatility, volume, smart-money
  trades, valuation, Claude's stance, insider buying and bubble score, and now **the news, filings and financial
  statements** known at each forecast's close: how much the company was in the headlines that week and their tone,
  material 8-K filings and red flags, revenue growth, operating margin, free-cash-flow yield and share issuance. Each
  candidate is tested month by month on forecasts it never saw, and the one with the best record so far is used. A nudge
  is applied only if that record passes a strict significance test, twice. With weak signals the honest answer is often
  "no nudge", and the app shows that test as it runs.
* **The fused model (the app's own algorithm):** it averages only the three algorithms that have been working best,
  chosen month by month from what was known at the time, so it is always tested on months it didn't help choose. It
  can be put in use only after nine test months as a real fusion, and only if it passes the same test as the others.
* **Ten years to learn from:** with the free Alpaca keys set, the app fetches every US stock's daily prices back to 2016
  once, so the backtest (and the tests above) cover about ten years of months instead of two. More months is what lets
  the tests tell a real signal from luck. With the free Massive key as well, stocks that were delisted since 2016 are
  included, so the backtest isn't flattered by looking only at the survivors.
* **Model and Claude:** their mix follows their measured skill on the same forecasts.
* **Claude's misses:** Claude reviews its worst misses, and the causes that at least two of those reviews agree on (from fixed
  lists, never free text) go into its next forecasts.

The loop was checked on simulated markets against the same forecasts without learning:
* With nothing to learn, the signal learner stayed off.
* With a planted signal, the competition picked a fitting kind of model (robust regression for a straight-line effect,
  the average or CatBoost for one that bends) and improved 1-week to 3-month forecasts in every quarter, by up to about 6%.
* The fused model picked the right three algorithms for each market (straight-line ones for a straight-line effect,
  ones that can bend for one that bends) and came second in both, ahead of almost every single algorithm.
* When earnings jumps grew bigger than in the history, the range learner improved 1-week and 1-month forecasts by about 2%.
* A weak signal looked harmful with two years of backtest and helpful with ten. It stayed off in both, because it
  wasn't clear enough, but more years is what lets the test see such effects at all.
Details, statistics and the simulation results are in [`docs/METHODS.md`](docs/METHODS.md) (sections 7–9).

Accuracy needs time: the first 1-day grades arrive the next trading day, 1-month grades after a month, and live calibration
switches on after about 200 graded forecasts from 20+ days. Until then the backtest (the model re-run on past prices) fills in.

## Results: the win rate, and why it missed
The **Results** tab is the forecasting tool's scorecard. A forecast *wins* when the price ends inside its 80% range, so
about 80% should win by design: well above that means the ranges are wider than needed, well below means too narrow.
* **The overall win rate** over every graded forecast (the same totals as the Records tab), for the model, Claude, the
  combined forecast and the backtest (never mixed): per horizon, over the last 90 days, and month by month, with an
  honest 95% interval that allows for overlapping forecasts.
* **Expected against actual:** the direction calls are shown next to what the forecasts themselves expected (a "55% up"
  forecast expects to be right 55% of the time), compared on the same calls, and next to "always say up".
* **Where it does better or worse:** with or without an earnings report in the window, calm or volatile stocks, thinly
  or heavily traded, the market rising or falling, nudged by the signal learner or not, and by industry.
* **Every miss, taken apart** (the newest of each horizon listed): the range it gave in prices, where the price ended
  and by how much, and why: the whole market moving, an earnings reaction, a one-day shock (with the 8-K filed that
  day, if any), far more volatility than expected, or a steady trend. Each part is measured from the daily prices over
  the forecast's window.
* **What changes because of it:** next to every miss and every cause, what the learning loop is actually doing about
  that kind of miss right now (wider ranges from the calibration or the coverage keeper, the range learner's test, the
  signal learner's test and the fused model), taken from the learners' latest decisions rather than promised.

The misses feed back into the forecasts through the learners: every miss counts in the calibration and the coverage
keeper; each stock's own record of misses, and of misses this breakdown put down to earnings, are inputs to the range
learner; and the news, filing and financial-statement signals are inputs the signal learner tests. Each change is used
only once it has beaten "no change" on forecasts it wasn't fitted on. The tab loads when you open it, so it doesn't
slow the rest of the app. Details: [`docs/METHODS.md`](docs/METHODS.md), section 9b.

## The self-check
Every refresh, the app checks itself: how old the data is, every file loaded, the trades and prices well-formed, every module
loaded, storage working, script errors, and the server's own check from its last run (files, prices, every stock's forecast and
valuation, forecasts up to date with the latest close, stuck grades, failing sources, the Claude analyst and budget, cache sizes,
config, the latest automated test run). When the network is too slow or you're offline, the header says it's showing the saved
copy, and a phone clock that disagrees with the data's timestamp shows as "Clock mismatch" rather than a wrong age.
Results are on the *Data* tab, a dot on the refresh button flags a problem, and a server failure sends a push notice (at most
one every 6 hours and never in quiet hours; a held-back notice is sent later, not dropped). If the refresh stops running
altogether, the header shows the data going stale after two hours, and the optional `HEARTBEAT_URL` secret gets you an alert.

## eToro (optional)
eToro's official API exposes Popular Investors' win rates and live positions, and the app can follow the top ones.
It ships **disabled** for two reasons: eToro says its API connectors aren't currently offered to US clients, and its terms allow
personal use only, while this app's website is public. If your account has API access and you accept that trade-off, add both keys
and set `etoro.enabled: true`.

## Where the "live trades" come from
| Source | Who | What you get | Delay |
|---|---|---|---|
| SEC Form 4 (live EDGAR feed) | corporate insiders | every buy, sale, grant, exercise | minutes after filing; filed ≤2 business days after the trade |
| SEC Form 144 (live EDGAR feed) | insiders about to sell restricted/control stock | advance notice of a sale, broker, planned date, 10b5-1 plan | minutes after filing, *before* the sale |
| SEC 13D/13G (live EDGAR feed) | activists and 5%+ holders | stakes, intent, amendments | minutes after filing; 13D ≤5 business days |
| SEC 13F | hedge funds / superinvestors | quarterly holdings changes | up to 45 days after quarter end |
| House Clerk / Senate eFD | members of Congress | trades with amount ranges | up to 45 days |
| ARK daily holdings files | ARK's active ETFs (Cathie Wood) | yesterday's buys, sells, new positions, exits | next morning |
| FINRA daily short-sale volume | the market | share of each stock's off-exchange volume sold short, vs its 20-day average | same evening |
| eToro Public API (optional) | top copy-traders | live positions | real time (off by default, see above) |

Websites that show "live trades" (insider trackers, Congress trackers, 13F sites) display these same official sources with
a delay of their own; the app reads the sources directly, so it is as fresh as the source allows and never depends on a
third party's terms. Options-flow and dark-pool "whale" feeds are paid, licensed data and aren't included.

## Good to know
* GitHub may delay scheduled runs at busy times. For steadier timing, a free service like cron-job.org can call the
  `repository_dispatch` endpoint (event type `refresh`) with a fine-grained token.
* GitHub pauses schedules in repositories with no activity for 60 days. The included *keep-alive* workflow prevents that.
* Official congressional reports may not be used for commercial purposes (Ethics in Government Act). Keep this personal.
* Headlines come from the GDELT Project (free, credited and linked on each stock page). Massive's news feed is off by default:
  its personal plans don't allow showing articles on a public website.
* The repository is public (that's what makes the free hosting and unlimited Actions minutes work), so anything you put in
  `config/config.yaml` (watch-lists, fund list) is visible to others. Keys go in **Secrets**, never in files.
* The refresh workflow has two jobs. The first reads all the outside data with a read-only token and hands over only the JSON
  data files it produced. The second, which holds the right to publish, builds the site itself from a fresh checkout of the
  app files exactly as committed plus those JSON files. A problem in a parser or a dependency therefore can't change the app's
  code or reach your browser-stored API key.
* The app's memory lives in GitHub's Actions cache. If saved memory (the small state, or the large cache that holds the price
  history and the forecast record) exists but can't be restored, the run stops instead of starting over, so a hiccup can't wipe
  your history, reset the accuracy record or re-send old alerts. Stock splits are applied to stored price history
  twice a day, so a 10-for-1 split never shows up as a 90% crash. Each run saves a fresh copy of the small state (a few MB);
  GitHub keeps the newest and evicts old copies once the repository's cache passes 10 GB, which is normal.
