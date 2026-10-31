# How the app's predictions work, and how honest they are

Every signal probability in the app answers the same question, measured the same way (price forecasts, which are ranges
rather than one probability, are in sections 7–9):

> **Will this stock beat the S&P 500 (SPY) over the next 63 trading days (~3 months)?** Measured from the close of the first
> trading day *after* the filing or disclosure became public (what a copier could actually do), with SPY measured over exactly
> the same days.

Nobody can reliably predict individual stocks. In the best published study, machine-learning models explained about 0.4% of
monthly individual-stock return variation out of sample (Gu, Kelly & Xiu 2020), which works out to a directional hit rate a few
points above 50%. So the app's job is to show **measured odds and their uncertainty**, not certainty, and to **grade itself**.

---

## 1. Base rates
* **What:** the share of past signals of a given kind that beat the S&P 500 over the next 63 trading days, with a 95%
  interval and the sample size. Rows with fewer than 30 cases are greyed out as "too few cases to trust yet".
* **Why the intervals are wide:** signals from the same few months share the market's and sectors' moves over overlapping
  3-month windows, so 6,000 filings are not 6,000 independent tries. The interval is cluster-robust over 13-week blocks, with
  a t critical value on half the blocks' degrees of freedom (neighbouring blocks overlap too), and never narrower than Wilson.
  In a simulation with a realistic shared factor (100 weeks × 60 filings), a plain Wilson interval covered the true rate only
  **31%** of the time (±1.3 points); this interval covered it **94%** of the time (±6 points).
* **Delisted stocks count.** A stock that stops trading inside the window (taken over, delisted, bankrupt, renamed) is graded
  to its last traded close against SPY over the same days, instead of being dropped, which would hide many of the worst
  outcomes.
* **Data:** the SEC's quarterly insider-transaction data sets (last 16 quarters) and whole-market daily prices (Massive) for
  insiders; the app's own congressional history; and, for kinds of signal with no long official history (fund moves,
  activist stakes…), the app's own graded ledger once it has 30+ cases.
* **Why many are below 50%:** single-stock returns are skewed (a few big winners, many laggards), so most stocks trail the index
  over short windows. A signal is interesting if it beats *its* base rate, not 50%.

## 2. The insider-purchase model
* **Unit:** one sample per stock per filing day (several insiders on one day count once, with their features combined). Live
  predictions are built the same way: every purchase filed that day in that stock is combined, and the 30-day cluster counts
  only filings up to that day. The 10b5-1 feature uses the Form 4 checkbox only (as the SEC data sets do), not footnotes.
* **Features, all known at the filing-day close:** seniority of the buyer, insiders on the filing, purchase size, insiders
  buying the same stock in the past 30 days, routine vs. opportunistic buyers (Cohen, Malloy & Pomorski 2012), 10b5-1 plan,
  1- and 3-month price moves, 6-month momentum, volatility, distance from the 52-week high and low, and trading volume (a
  size proxy). Market-wide features (index trend) are deliberately left out: they're identical for every filing on a given
  day, so a few months of test data can't validate them and they'd mostly learn noise about market timing.
* **Routine vs opportunistic** needs the same calendar month in each of the 3 prior years to be inside the data; when the month
  three years back is before the data starts, the buyer is "unknown", not "opportunistic".
* **Fair test:** train on the oldest ~70% of filings, skip 100 days (so no 3-month outcome overlaps the training period), and
  test on the newest filings. One pre-registered model (regularised logistic regression) is judged; a gradient-boosting model
  is reported alongside for comparison only, so the winner isn't picked on the test data.
* **The gate. All of these must hold, or the app shows base rates and says "No measurable predictability":**
  1. a lower Brier score on the test period than simply quoting the training base rate;
  2. the 95% interval of its ranking accuracy (AUC) stays above 0.5, counting only pairs of filings **from the same week and
     the same size third** (by trading volume). Same-week pairs remove the market's shared move, which no filing predicts;
     same-size pairs stop a few months of small caps beating large caps (or the reverse) from passing for stock-picking. The
     interval resamples weeks and stocks independently (a "pigeonhole" bootstrap), because filings in the same week share the
     market, and filings in the same stock (a buying spree, several insiders in a row) share one outcome;
  3. ranking accuracy above 0.5 in **both** halves of the test period;
  4. at least 300 test filings, over at least 10 weeks, in at least 60 different stocks.
* **Decided once a quarter.** The data and the model are rebuilt daily, but the pass/fail decision is taken once per calendar
  quarter and then held: re-deciding every day on slightly different data would give luck about 90 chances a quarter to pass.
  The Records tab shows when it was decided and what today's re-check says.
* **Self-tests on simulated markets** (`tests/test_quant.py`, `tests/test_review2.py`, and larger sweeps run during the audit):
  | World | Result |
  |---|---|
  | Planted signal (24 trials) | found 24 of 24 |
  | Pure noise (24 trials) | wrongly passed 0 of 24 |
  | Noise with insider buying sprees: several filings, one outcome (40 trials) | wrongly passed 0 of 40 (the previous gate: many) |
  | Noise where volatile, high-beta stocks beat SPY in rallies (30 trials, plus 30 controls) | 0 of 30, 0 of 30 |
  | Noise, re-tested at 10 dates three weeks apart (12 worlds) | passed at none of the 120 looks |
  | A real but modest edge (held-out AUC ≈ 0.57) at production size | passed 3 of 6 |
  | A clearer edge (AUC ≈ 0.62) | passed 6 of 6 |
  So the gate is deliberately conservative: a small real edge is caught about half the time, noise essentially never.
* **Shown with:** the held-out result for filings the model ranked in the same fifth ("filings ranked in the top fifth: 58%
  beat the index in the test"). The live model is refit on all data, so a live score is placed by its rank among the refit
  model's own scores, not by its raw number. Plus its ranking accuracy and a plain label (None detected / Low / Moderate).

## 3. Typical 3-month price range
From the stock's own one-year volatility, assuming no drift: 2 in 3 outcomes inside ±1 standard deviation, 9 in 10 inside
±1.645. A description of how much it swings, **not** a forecast of direction.

## 4. Claude's analysis
* **Models:** Claude Opus 5.5 by default (Anthropic says it matches Claude Fable 5.1 on most work at 40% lower running cost
  than Opus 5; list price $4/$20 per million input/output tokens), or Claude Fable 5.1 ($10/$50). Effort is set with the
  API's `output_config.effort` (`low`, `medium`, `high`, `xhigh`, `max`); deep analyses default to `max`.
* **Research:** Anthropic's server-side web search ($10 per 1,000 searches), up to 6 searches per deep analysis, for current
  news, the next earnings date, guidance, analyst actions and financing. The sources listed in the app are the pages the API
  actually returned (not URLs typed by the model).
* **Inputs:** every fact of the filing, the signal score and its reasons, measured predictability and base rate, price and
  risk, short interest, next earnings date, other recent signals in the stock, the market backdrop, and the app's sizing
  ceilings.
* **Instructions (the "playbook", `pipeline/analyst.py`):** anchor on the measured base rate and adjust only for evidence it
  can name (typically a few points; beyond ±10 needs exceptional evidence; stay within 25–75% unless something is truly
  extraordinary); give both sides; separate facts from judgment; never promise gains; speak about sizing only relative to
  the app's ceiling; prefer primary sources. It also carries a summary of the evidence below and standard risk practice
  (volatility-based sizing, pre-defined invalidation, event risk, liquidity, regime, no averaging down on a broken thesis).
* **Output:** a strict schema (stance, probability, confidence, headline, thesis, bull/bear case, catalysts, what to watch,
  what would prove it wrong, position note, facts checked, gaps). The app clamps probabilities and flags any that depart more
  than 20 points from the measured base rate.
* **Costs:** computed per call from the API's usage report and capped by a daily budget; the Data tab shows today's and the
  month's spend. Every billed turn is charged even if a later turn fails; a request is never re-sent automatically once it may
  have reached the API (a lost response is charged at the tier's estimated cost, since it may have been billed); each task has
  its own cost cap ($2.50 deep, $0.40 quick by default); a rate limit without a retry time (a spend limit) stops the run; and a
  failure that cost money counts as a strike (two strikes and the task is dropped), so a stubborn error can't be paid for every
  20 minutes. Output is capped at 20,000 tokens per call so a non-streamed call stays well inside the API's ~10-minute guidance;
  a view that runs out of tokens is retried once at effort `high`.
* **Graded** like everything else (section 6). If Claude's calls don't beat the base rates over time, the scorecard will show it.

### 4b. Ask Claude (the chat box)
The *Ask* tab is the same analyst, interactive: the browser sends your question to the Claude API with your own key, with
two cached system blocks: a short playbook (the honesty rules above, condensed) and the app's current data (the strongest
and newest signals with scores, measured odds and existing Claude views; prices and risk; base rates and the live scorecard;
the market backdrop; the Ideas screens and latest brief; your watchlist; and, when opened from a trade, that trade's full
record). Claude Opus 5.5 at effort `max` by default, streaming, with up to 5 web searches per answer; the sources it read
and cited are listed under each answer, with the cost computed from the API's usage report. Chat answers are not written to
the forecast ledger (they aren't the app's standing view of a trade), so they are not graded.

### 4c. Two newer signal kinds
* **Form 144 sale notices** score low and bearish (roughly 10–40 of 100): a notice of intent, filed before the sale, usually for
  compensation shares or under a 10b5-1 plan. Points for size, share of all shares outstanding, seniority and several insiders
  selling together; deductions for a plan and for compensation shares. Tracked in the ledger (`insider:sale_notice`) so a base
  rate accumulates.
* **Active ETF daily trades** score like 13F changes but lower (daily funds trade a lot; a trim is often a rebalance):
  new positions and large adds with a meaningful fund weight score highest, more when several of the manager's funds traded
  the same way the same day, and the price context counts half. The one-day lag is a plus; "the file shows what changed, not
  why" is always listed against. Tracked in the ledger (`etf:new/add/trim/exit`), so each action's base rate builds up.

## 5. Ideas (setups that aren't a single trade)
| Screen | Rule | Track record shown |
|---|---|---|
| Smart-money convergence | 2+ independent informed buyers (insiders, funds, 13D activists, Congress) in the same stock within ~45 days | none yet for this exact mix: no probability until the app's own graded record reaches 30; the closest measured pattern (3+ insiders within 30 days) is shown, labelled as related |
| Insiders buying into weakness | 2+ insiders buying within 30 days, the stock down 15%+ over the 3 months before the latest filing (measured at that filing's close) | the base rate of exactly that rule |
| Momentum near highs | top-decile 6-month momentum (skipping the last month), within 5% of the 52-week high, volatility under 70% | monthly backtest on the stored history, vs. all liquid stocks, including stocks that later delisted; 95% interval allows for the overlapping 3-month windows (Newey-West, conservative degrees of freedom: a null screen's interval excluded zero in 2.7% of 150 simulations, versus 19% for the naive interval) |

Claude's weekday brief reads these lists, the market backdrop and the strongest new signals, researches the week's calendar,
and writes 4–8 setups, each with a trigger and an invalidation. Setup probabilities are graded too.

## 6. The forecast ledger (the report card)
* Every probability the app shows (model, base rate, Claude, screens) is written down the first time it's shown, dated the day
  it's written (never backdated), and **can't be revised afterwards**. A corrected filing (Form 4/A) keeps the original's entry,
  and an idea that drops off a list and returns within 30 days keeps its first date, so nothing gets a fresh start after a bad
  stretch. Trades shown without a probability are tracked too, so base rates for every kind of signal build up.
* Each entry is graded after its own horizon (63 trading days unless a setup said otherwise). Stocks that stopped trading are
  graded to their last trade; only entries that were never priced at all are voided.
* The Records tab's Accuracy view shows, per source: beat rate with a 95% interval that allows for forecasts in the same months
  moving together, how often the call's direction was right (only where the probabilities vary), Brier score and its skill
  versus simply quoting the observed frequency, a calibration chart ("when it said 55–60%, how often did it happen?"), and
  Claude's results by stance.

## 7. Price forecasts (the stock page)
**What is forecast:** the *distribution* of a stock's closing price 1 trading day, 1 week (5 sessions), 1 month (21), 3 months
(63) and 1 year (252) ahead, drawn as a fan (the middle 50%, 80% and 90% of outcomes and the median), plus the chance of
finishing higher, the chance of touching +10% or −10% on the way, the typical worst drop from a peak, and when within 3 months
the low is most likely to land. There is deliberately no single "target price".

**Why ranges and windows, not a line and dates.** Over days to months, direction is close to a coin flip for everyone: the best
published machine-learning models explain about 0.4% of monthly single-stock return variation (Gu, Kelly & Xiu 2020), which is a
directional hit rate a few points above 50%. What a forecaster *can* get right is the spread: how far the price may travel and
how likely each range is. And for prices that move like a random walk, the day of the high (or low) inside a window follows the
arcsine law: it is most likely near the start or the end of the window and spread thinly in between. So the app gives
probabilities, event windows (earnings, Fed, CPI) and the share of risk on each event day, never "it peaks on the 14th".

**The model: filtered historical simulation** (Barone-Adesi, Giannopoulos & Vosper 1999), per stock, from about two years of
daily closes:
1. *Volatility now:* an exponentially weighted average of squared daily log returns (λ = 0.94, RiskMetrics), skipping
   earnings-reaction days, which are modelled separately.
2. *The stock's own shocks:* each past day's return divided by the volatility predicted for that day. Future days resample
   these, so each stock keeps its own fat tails and skew. They are demeaned but not rescaled to unit variance (rescaling made
   the ranges too narrow in tests on simulated fat-tailed, volatility-clustering prices).
3. *Volatility path:* reverts from today's level toward the stock's long-run level with a 40-session half-life.
4. *Known events:* an earnings day's variance is multiplied by the stock's own measured earnings-day reaction (the root mean
   square of its past earnings-day shocks, held between 1.5× and 6×; 3× without history). A report after the close lands on the
   next session, including one released after the close on the forecast's own anchor day. For past reports the timing comes
   from when the SEC accepted the earnings 8-K: before that day's close (the SEC's timestamp is New York time, despite its
   "Z"), the same session reacts; after it, the next one. Fed decision days ×1.25 and CPI
   release days ×1.15. Dates: the next earnings date (Alpha Vantage if connected, otherwise the stock's filing rhythm), the
   Federal Reserve's FOMC calendar and the BLS CPI schedule, back to 2024 so the backtest sees the same events (extra dates can be
   added in `config/config.yaml`).
5. *Drift:* the cash rate plus beta × a 5% equity premium (beta against SPY over a year, held between 0.3 and 2.5), minus the
   stock's trailing dividend yield (the price history isn't dividend-adjusted, so a price drops by its dividend), minus half the
   variance so the median isn't biased upward. This is an assumption, not a view; over a month it is tiny next to the range. A
   learned "tilt" is added only if one passes the test in section 9, and only over the horizon it was tested on.
6. *2,000 simulated paths × 252 sessions*, seeded from the ticker and date, so the same inputs always give the same forecast.
7. *Calibration* (section 9) widens or narrows each horizon's range and nudges its centre, from how past forecasts did. Beyond
   the longest horizon that has been graded, the correction fades out rather than being stretched to a year.
8. *Clean inputs:* only completed sessions are used (an unfinished intraday bar never anchors or grades a forecast; early 1pm
   closes count), and a stock whose last few days show a split the history hasn't been re-based for yet is held back until the
   re-base, which runs right after each day's prices arrive. A move counts as a split when the split records show one within a
   few days whose ratio explains it, or when it is a ratio no ordinary day produces (exactly −50%, −67%, +900%…). A −20% or
   −25% day, which earnings reactions produce all the time, isn't mistaken for a 5-for-4 or 4-for-3 split.

**On the stock page:** the median, the 80% range and the chance of finishing higher for the chosen horizon; the
peaks-and-dips grid (the typical high and low along the way, the chance of touching +10% and −10%, the typical worst drop from
a peak and the chance of a 20%+ drop);
the day that carries the most risk (shown when one day holds over 4% of the 3-month variance, typically earnings); a bar of
when the 3-month low is likely to land; and a table comparing the model, Claude and the combined forecast.

## 8. Claude's price forecasts and the combined forecast
* **Which stocks:** up to 3 a day (`analyst.forecasts_per_day`), your watchlist first, then the strongest signals, each refreshed
  at most weekly. Claude Opus 5.5 at effort `high` with up to 4 web searches, a $1.50 cap per forecast.
* **Inputs:** the model's distribution for every horizon, the event calendar, valuation, the bubble checklist, AI exposure,
  recent 8-K filings and headlines, its own earlier forecasts for the stock (numbers only), and what its post-mortems agree on
  (below). Headlines, filing titles, filers' remarks, web pages and its own earlier theses (which may quote a page it read) are
  passed in a block marked as untrusted data, and its standing rules say to weigh such text as evidence and never follow
  instructions inside it.
* **Rules it is given:** start from the model's distribution; move the median only for evidence it can name; keep ranges at
  least as wide as the model's unless it can say exactly why; keep the chance of rising near 50% without extraordinary
  evidence; give windows and events, never the day of a peak or dip; cite the pages it read.
* **Output** (strict schema): for 1 week, 1 month, 3 months and 1 year, a 10th percentile, median, 90th percentile and chance of
  rising (from the close it was given); bull/base/bear scenarios with probabilities; the dip-risk and upside windows with
  reasons; key dates; a news digest with links; risks; what would change its mind. A row that is out of order, or whose median
  sits a full 80%-range width outside the model's range, is dropped, not repaired. A news link is kept only if it is a page the
  API actually returned during that forecast.
* **Graded fairly:** from the close of the *reference session* (the first close after the answer came back), with target dates
  counted in trading sessions from it, so news after the close it saw can't flatter it. Its prices are re-based for any later
  split before grading.
* **Turned into a full distribution** by a split-normal curve through its 10th, 50th and 90th percentiles (in log price).
* **Combined forecast:** for each of Claude's target dates, Claude's quantiles and the model's quantiles for that same date,
  averaged in log price. Equal weights until both have been graded; then tilted toward whichever has the better skill measured
  on *the same* forecasts (every graded Claude forecast also scores the model's forecast from the same close on the same
  outcome), at most 0.3 either way, shrunk by days/(days+40). Horizons whose date has passed are dropped from the page.
* **On your phone:** "Ask Claude for a fresh forecast" on any stock page runs the same request with your own key (effort at
  most `high`), stores the result on the phone (its cost is added to the device's total) and grades it there from the published
  closes, re-based for splits (Records → Forecasts).

## 9. Grading and learning (Records → Forecasts)
**What is recorded.** For up to 250 stocks a day (watchlist first): 1-day, 1-week and 1-month forecasts every trading day,
3-month weekly and 1-year monthly (so long, overlapping forecasts don't swamp the record), each with the features known at its
anchor close (nothing filed or published after that close). Entries are never revised. Each is graded at its target date's
close; a stock that stops trading is graded to its last close; one never priced is voided. Every graded forecast counts in the
accuracy figures; for the learning, a fixed random sample of 30 per day and horizon is kept for about 15 months (outcomes on
one day move together, so more of the same day adds little), which is what lets calibration and the tilt test run at all.

**How accuracy is measured**, per horizon, all-time and over the last 3 months, with a monthly trend chart:
| Measure | What it tells you | What "good" is |
|---|---|---|
| **Range coverage** | how often the outcome landed inside the 50%, 80% and 90% ranges | the stated level: 80% for the 80% range. Below means over-confident, above means too cautious. This is the 80%+ target the app holds itself to, and it can be met |
| **Skill vs a random walk** | the quantile (pinball) score against a naive forecast centred on today's price with the stock's own volatility (a proper scoring rule: Gneiting & Raftery 2007) | above 0; expect a few percent at best |
| **Direction** | how often "more likely up than down" was right, shown next to "always up" | expect roughly 50–56%; nobody sustains 80% |
| **PIT histogram** | where outcomes fell inside the forecast distribution | flat; a U shape means ranges too narrow |

**Why the target is coverage, not 80–90% direction accuracy.** A forecaster who called direction right 80% of the time over
months would beat every fund on record by a wide margin; the published evidence puts the achievable level a few points above
50%. Coverage and skill are the measures that can honestly reach their targets, and they're what the learning loop optimises.

**How it learns from its mistakes:**
1. *Calibration:* per horizon, a width correction (scale) and centre correction (shift) estimated from the last 200 days of
   graded forecasts. It starts from the walk-forward backtest (item 5) and switches to live grades once there are 200+ from 20+
   different days. Evidence is counted in days, not rows: the correction is shrunk toward "no change" by days/(days+20); the
   scale stays within 0.6–2.0 and the shift within ±¼ of a standard deviation. The accuracy page shows its before → after
   coverage on the most recent 30% of days, with the correction fitted only on forecasts whose outcome date (the horizon's
   number of trading sessions later, holidays included) falls on or before the first of those days, so no fitted outcome
   overlaps a tested one (a hold-out test, not the fit grading itself), and says which source it's using. The stock page shows
   how often the 80% range has held at that horizon (live grades, or the backtest until there are enough) and warns that the
   ranges read too narrow or too wide only when an interval that allows for overlapping forecasts excludes 80% and rests on at
   least three independent periods; the warning quotes that interval's own rate.
2. *Signal tilts:* each forecast records signal features as of its anchor close (the stock's signal score and direction, 12-1
   month momentum, 1-month reversal, valuation at that close, Claude's stance as known then, insider buying minus selling, bubble
   score). A ridge regression on graded 1-month outcomes (measured against the median *before* any tilt, so an applied tilt
   can't feed back into its own target) proposes a small nudge, used only if a walk-forward test passes: refit each month on
   earlier months and tested on the next, the average monthly gain must be positive with t ≥ 2.5 (standard error inflated by √2
   because neighbouring months overlap; 2.5 because the test is repeated weekly), above 0.2%, positive over the last 3 months,
   and passed at two weekly checks in a row. Once on, it switches off when t falls below 1.5 or the last 3 months turn
   negative. It is applied only over the 21 sessions it was tested on, never extrapolated to a year, and capped. On simulated
   data, a single weekly test passed in 0 of 40 worlds with no real signal and found a planted signal in 20 of 20.
3. *Blend weights* between the model and Claude follow their skill on the same forecasts (section 8).
4. *Claude's post-mortems:* up to 2 a day, Claude reviews one of its biggest recent misses (the outcome outside its 90% range)
   with what actually happened, names the cause and the range advice from fixed lists, and writes a one-line note. Its later
   forecasts see only the patterns that at least two post-mortems agree on (for example "earnings gap, ranges wider: 3 times"),
   built from those fixed lists. The free-text notes (stripped of links and tickers, and rejected if they read like
   instructions) are shown on the Records tab but never fed back into a prompt, so one planted web page can't steer later
   forecasts. The measured record decides how much its forecasts count. A forecast stopped by the daily budget isn't counted
   as a failure.
5. *Walk-forward backtest:* the model re-run at 40 past dates (every 10 sessions) on a sample of up to 150 stocks that were
   liquid *on each date*, including ones that later delisted (so it isn't flattered by survivors), with only what was known
   then: prices, beta, Fed and CPI dates, and earnings dates (from filings, or for stocks without a filing record, estimated from
   the volume spikes that mark earnings days). Accuracy is visible from the first day and calibration has data to start from.
   It's labelled "Backtest" and never mixed with live grades.

## 10. Valuation: cheap or expensive versus similar companies
* **Data:** the SEC's XBRL company facts (what companies report in their 10-Ks and 10-Qs): revenue, gross profit, operating
  income, net income, operating cash flow, capex, dividends paid, cash, debt and shares. Trailing twelve months = latest fiscal
  year + this year-to-date − last year-to-date (falling back to the last four quarters, or the annual figure if it is under 460
  days old). When a value is restated, the latest filing wins. US-dollar facts only. Balance-sheet items are all read at one
  date (the latest balance sheet), so a tag a company stopped using years ago can't creep in, and debt is never counted twice
  (long-term debt already includes its current part; "debt current" already includes short-term borrowings).
* **Shares:** every share class on the cover page is counted (two classes with the same count are two classes). Classes that
  don't add up to the diluted count have different economics (Berkshire's A and B shares, say), so one price can't value them
  and the stock isn't valued; neither are foreign companies filing 20-F, whose US shares are usually ADRs with a ratio the SEC
  data doesn't give. Share counts reported before a later split are re-based with the split records, so a 10-for-1 split
  doesn't make a company look ten times cheaper until its next filing; a split counts only from the day the price used
  reflects it, so a pre-split price is never multiplied by post-split shares.
* **Market value** from shares × the last close (the page says which close); **enterprise value** = market value + debt − cash.
* **Relative valuation:** across every company with data (at least 120 are needed, so it fills in over the first days), a ridge
  regression of log(EV/sales) on revenue growth, growth², operating margin, margin × growth, gross margin and industry (2-digit
  SIC groups with 15+ members). Where the stock's actual multiple falls among the regression's residuals gives the label: Very
  cheap (bottom 10%), Cheap (10–30%), Fairly valued (30–70%), Expensive (70–90%), Very expensive (top 10%). That is,
  expensive *for its growth, margins and industry*.
* **Banks, insurers and REITs** (SIC 6000–6799): price/earnings and price/book percentiles among financials instead, since
  sales multiples mean little for them.
* **Reverse DCF:** the 10-year free-cash-flow growth rate the current enterprise value implies, at a 9% discount rate and 3%
  growth afterwards. Useful as a reality check ("the price assumes 35% a year for a decade").
* **Limits:** relative, not absolute (if a whole industry is overpriced, "fairly valued" within it can still be expensive);
  accounting one-offs and acquisitions distort TTM growth; it ignores balance-sheet quality beyond debt and cash.

## 11. Bubble risk and the AI theme
* **The checklist** follows Greenwood, Shleifer & You (2019, *Journal of Financial Economics*, "Bubbles for Fama"). Like the
  study, a run-up counts only as far as both the raw and the net-of-market return clear the bar. Across US
  industries from 1926 to 2014, after a two-year run-up of 100% net of the market, the chance of a 40%+ crash within two years
  was 53% (20% after a 50% run-up, 80% after 150%); among run-ups, the ones that crashed more often had rising volatility, heavy share
  issuance, young firms and an accelerating price path (turnover was high in run-ups that crashed and in those that didn't,
  so it counts little). Points: the two-year net run-up (0.5 at 25%, 1.5 at 50%,
  2.5 at 100%, 3 at 150%+), acceleration 1, volatility 0.5, turnover 0.5, issuance 1, young firm 0.5, valuation (Very expensive
  1.5, Expensive 0.75), price/sales ≥ 25 0.5, net insider selling 0.5, and the AI theme when it is Elevated or High (1 for
  core AI stocks, 0.5 otherwise). Levels: Low, Moderate (1.5+), Elevated (3.5+), High (5.5+). Every sign is listed on the page.
* **Honest scope:** the study measured industries, not single stocks, and found that run-ups alone did not predict lower
  *average* returns (some bubbles keep inflating). The app shows the industry crash rate as historical context, never as this
  stock's probability.
* **The AI bubble gauge** (Ideas tab): an equal-weight basket of AI chip and infrastructure ("core") and AI platform stocks
  (`bubble` in `config/config.yaml`), rebalanced daily (an average of simple returns, compounded), scored with the same signs
  plus breadth (how many members are near highs) and the distance from the basket's high. Phase: Inflating (run-up and still rising), Wobbling (run-up but off its high or narrowing),
  Deflating (falling well off its high), Calm.
* **Each stock's AI link:** a regression of its daily returns on SPY and on the AI basket's return over SPY; "linked to the AI
  trade" when the AI loading is at least 0.5 with t ≥ 3. That tells you how much of a stock's moves are the AI trade, whatever
  its business says.

## 12. Ratings and "Top conviction"
Each stock gets one rating: **Top conviction**, Positive, Neutral, Caution or Red flag. Top conviction needs all six checks:
1. Claude's research view (last 30 days) is Favorable, with a chance of beating the S&P 500 over ~3 months of at least 58% and at
   least 8 points above the measured base rate (for a price forecast, the measured share of all stocks that beat the index),
   at medium or high confidence;
2. measured evidence agrees, and thin evidence doesn't count: a recent bullish signal from the insider model that passed its
   out-of-sample gate and predicts a lift of 3+ points, a kind of signal whose beat rate is above 50% across its whole 95%
   interval (30+ cases), or a place on an Ideas screen whose track record beats the index with confidence;
3. valuation is not Very expensive; 4. bubble risk is not High; 5. risk level 4 of 5 or lower and at least $5M traded a day;
6. no red-flag 8-K in 90 days (bankruptcy, accelerated debt, a delisting notice, or financial statements that can't be relied
   on). A red-flag filing makes the rating Red flag.

Each new Top conviction call is written to the ledger and graded against SPY after 63 sessions; its record is on Records →
Forecasts, and until it has one the app says so next to the rating. It means "every check the app can measure lines up", not
"this will go up", and it is research, not advice.

## 13. The self-check
* **On the server, every run:** data files present, valid and rewritten this run; whether the data itself is moving (newest
  trade and newest price, in business days); the feed's schema and unique ids; prices positive and plausible; every stock
  page published, its forecast ordered, positive and anchored on a real price, its Claude and combined forecasts ordered, and
  valuations finite; forecasts up to date with the latest close (most of them two or more sessions old is a failure: the
  forecast job has stalled; skipped when forecasts are switched off, and then no old forecasts are published); no ledger entries stuck past their date; data sources, including jobs that run every time such as
  prices and stock pages, each judged against the cadence it runs at: failing repeatedly is a warning, still being tried but down for three days is a failure (a market-data
  key that's refused fails loudly rather than looking like the end of the plan's history); the Claude analyst running within
  budget; cache and site sizes; config values in range; the latest finished test run on the main branch. Published as `audit.json`. A failure sends one push notice; a notice held back
  (at most one every 6 hours, never in quiet hours) stays pending until it is sent, and a failure that clears and comes back
  alerts again.
* **Dead-man's switch (optional):** if the refresh stops running altogether, nothing inside the app runs to notice. Set the
  `HEARTBEAT_URL` secret to a free monitoring check (healthchecks.io, say): each run pings it, and the service alerts you when the
  pings stop. The app also shows stale data in the header and the self-check after two hours.
* **In the app, on every open and refresh:** how old the data is (and whether its timestamp is in the future, a sign of a wrong
  clock; the header then says "Clock mismatch" instead of showing an age), whether the network answered or the saved offline
  copy did (the header says "saved copy", and the next refresh tries again), the feed and prices, every file loaded and from the same refresh, whether the server's audit is from this refresh, the
  phone's storage, offline mode, every module loaded, script errors since opening, and the server's result (shown beside the
  phone's checks, not counted twice). The Data tab lists each check; a dot on the refresh button flags a failure.
* **Every night:** the full test suite runs on GitHub Actions; its result appears in the self-check.

## 14. Evidence the playbook is built on (summarised; magnitudes are historical averages)
| Finding | Source | Holds up today? |
|---|---|---|
| Insider purchases carry information, sales mostly don't; purchases earned ~40 bp/month, a third in the first month | Lakonishok & Lee (RFS 2001); Jeng, Metrick & Zeckhauser (REStat 2003) | Yes, mostly in smaller firms; more of it now happens right at the filing |
| Opportunistic insiders ~82 bp/month (value-weighted), routine insiders ~0 | Cohen, Malloy & Pomorski (JF 2012) | Yes |
| 13D activism: ~7% abnormal return around the announcement (2001–06) | Brav, Jiang, Partnoy & Thomas (JF 2008) | Weaker: later work finds about half arrives *before* the filing and long-run value-weighted returns are insignificant |
| Managers' "best ideas" outperform; copying 13Fs after disclosure roughly matches active funds after costs | Cohen, Polk & Silli (WP); Frank et al. (JLE 2004); Verbeek & Wang (JBF 2013) | Mixed |
| Congress: 1990s studies found outperformance; later work found the average member lagged, and post-2012 House buys underperformed; newly appointed leaders outperformed in a 2025 working paper | Ziobrowski et al. (2004, 2011); Eggers & Hainmueller (JOP 2013); Belmont et al. (JPubE 2022); Wei & Zhou (NBER WP 2025) | Weak for rank-and-file; leaders result not yet peer reviewed |
| Momentum ~1%/month historically; can crash in panic rebounds (−74% in Aug 1932, −45% in Apr 2009) | Jegadeesh & Titman (JF 1993); Daniel & Moskowitz (JFE 2016) | Exists; net-of-cost returns smaller |
| Nearness to the 52-week high explains much of momentum | George & Hwang (JF 2004) | Internationally mostly insignificant after costs |
| Profitability and quality earn higher risk-adjusted returns | Novy-Marx (JFE 2013); Asness, Frazzini & Pedersen (RAS 2019) | Robust across decades and countries |
| Low volatility / betting against beta | Ang et al. (JF 2006); Frazzini & Pedersen (JFE 2014) | Much of it sits in tiny stocks (Novy-Marx & Velikov 2022) |
| Post-earnings drift | Bernard & Thomas (1989/1990) | Essentially gone in large caps since ~2006 (Martineau 2022) |
| Accruals; drift after buybacks and offerings | Sloan (1996); Ikenberry et al. (1995); Pontiff & Woodgate (2008) | Decayed (Green, Hand & Soliman 2011; Fu & Huang 2018) |
| Heavily shorted stocks with low institutional ownership underperform | Asquith, Pathak & Ritter (JFE 2005); Boehmer, Jones & Zhang (JF 2008) | Mostly small caps, equal-weighted |
| Published edges shrink: −26% out of sample, −58% after publication; 65% of 452 anomalies fail once tiny stocks are handled | McLean & Pontiff (JF 2016); Hou, Xue & Zhang (RFS 2020) | Counterpoint: Jensen, Kelly & Pedersen (JF 2023) find most major factors replicate |
| Trend rules (10-month average) cut drawdowns sharply, with a modest return edge | Faber (2007/2013); Moskowitz, Ooi & Pedersen (JFE 2012) | Drawdown reduction robust; extra return weak |
| Forecast quality is measured with the Brier score and calibration | Brier (1950); Good Judgment Project (Mellers et al. 2014) | Standard |

## 15. What this can't do
* See the future. Expect small edges, long losing streaks, and regime changes that break past patterns.
* Forecast direction or name the day of a peak. The forecasts are ranges with probabilities; a calibrated 80% range still
  misses one time in five, and the misses cluster in crashes and squeezes that history underweights.
* Undo disclosure lags: insiders up to 2 business days, Congress up to 45 days, funds up to ~4.5 months after the quarter.
* Fix bad inputs: filings can contain errors. Delisted stocks are kept (graded to their last trade), but the $5 minimum price
  for the momentum screen is applied to split-adjusted prices, so a stock that later did a big reverse split can look pricier
  in the past than it really was.
* Make Claude infallible: it can misread a source or be wrong. Sources, the strict schema, the base-rate anchor and the
  ledger are there to catch that, not to guarantee against it.
* Give personal advice: it doesn't know your finances, taxes or goals. The sizing ceilings are rules of thumb you set.
