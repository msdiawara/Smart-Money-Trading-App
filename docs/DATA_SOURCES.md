# Data sources: what the app uses and why

Smart Money Radar only uses the **original source** of each disclosure when one is reachable. Aggregator websites
re-publish the same government filings later, so they can never be fresher than the source and sometimes add
errors. Third-party services are used only where the official source can't be reached from a cloud server,
or for data that has no official source (prices, eToro rankings).

Last researched and audited: 1 Nov 2026. The *Audit log* at the bottom records how each claim was checked and what changed as a result.

## 1. Corporate insiders (CEOs, directors, 10% owners): SEC Form 4
| | |
|---|---|
| Source | SEC EDGAR, official and free, no key |
| Live feed | `https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&type=4&owner=include&count=100&output=atom` (real-time list of accepted filings). Each filing appears twice (issuer + reporting owner), so 100 entries ≈ 50 filings ≈ 15–30 minutes on a weekday. The app pages back (up to 1,500 entries) until it reaches the last filing it already has |
| Backstop | Nightly daily index `Archives/edgar/daily-index/YYYY/QTRn/form.YYYYMMDD.idx` catches anything the live feed missed |
| Document | Filing folder `index.json` → raw ownership XML (`ownershipDocument`) |
| Freshness | Filings appear within minutes of SEC acceptance. Insiders must file within **2 business days** of the trade |
| Rules | Max 10 requests/second; a declared `User-Agent` with a contact e-mail is required (we use ≤ 8 req/s) |
| Notes | The feed's `type` filter is a prefix match (it also returns "4/A" and other forms), so forms are filtered exactly in code. Since April 2023 Form 4 has a Rule 10b5-1 checkbox (`aff10b5One`), which we read, plus a footnote text fallback |

## 2. Activist / large stakes (more than 5% of a company): Schedules 13D and 13G
| | |
|---|---|
| Source | SEC EDGAR current feed, `type=SCHEDULE 13` (structured XML is required since 18 Dec 2024) plus the legacy `SC 13` prefix |
| Freshness | 13D: due within **5 business days** of crossing 5%; amendments within 2 business days. 13G (passive): 5 business days to 45 days after quarter-end depending on filer type |
| Parsing | XML parsed by local element names (resilient to schema version 2.x tweaks); falls back to metadata + link if a field is missing |

## 3. Superinvestors / hedge funds: Form 13F-HR
| | |
|---|---|
| Source | SEC `data.sec.gov/submissions/CIK##########.json` per tracked fund → information-table XML |
| Freshness | Quarterly; due **45 days after quarter end** (Feb 14/17, May 15, Aug 14, Nov 14/16). Positions can be up to ~4.5 months old when seen |
| Units | Values are in **whole dollars** for filings since 3 Jan 2023 (thousands before). Real filings break this rule both ways (one 2022 filer used dollars; one 2023 filer kept thousands), so units are detected per filing from the implied price per share |
| Tickers | 13F lists CUSIPs; mapped to tickers with **OpenFIGI** (Bloomberg's free API: 25 requests/min and 10 jobs/request without a key, 25 per 6 s and 100 jobs with a free key). Cached forever |
| Fund list | Curated in `config/config.yaml`; every CIK is re-validated against the SEC's own entity name (and former names) on each run. Mismatches are skipped and shown in Data Health, and so are funds whose latest filing is a 13F-NT (holdings moved to another filer). Corrected after audit: Pershing Square → CIK 0002026053; Greenlight → DME Capital Management 0001489933; Scion (Burry) removed, since it deregistered in Nov 2025 |

## 4. Members of Congress: STOCK Act Periodic Transaction Reports
| | |
|---|---|
| House | **Official House Clerk**: yearly index `public_disc/financial-pdfs/{YEAR}FD.zip` (XML, refreshed daily) and one PDF per report at `public_disc/ptr-pdfs/{YEAR}/{DocID}.pdf`. E-filed PDFs are parsed; scanned paper filings are linked but not parsed |
| Senate | **Official Senate eFD** (`efdsearch.senate.gov`) is tried first (free), but it blocks many cloud IPs: a GitHub Actions job got HTTP 403 on every run from 23–29 Oct 2026. Dependable fallback: **Quiver Quantitative** Hobbyist API ($25–30/month, `Authorization: Token …`). **FMP** (`/stable/senate-latest`) is also supported, but which FMP plan includes it is unverified, so test with your key |
| Freshness | Members must report within **45 days** of a trade (30 days after notification). Amounts are disclosed only as ranges ($1,001–$15,000 etc.) |
| Who's who | **@unitedstates/congress-legislators** (`legislators-current.json`, public domain, refreshed daily): party, state, district, Bioguide ID and `leadership_roles`. Party leaders (Speaker, floor leaders, whips, caucus/conference chairs) are recognised by Bioguide ID, never by surname, so a trade by Rep. Dusty Johnson isn't mistaken for Speaker Mike Johnson's. Checked 30 Oct 2026: 539 current members; the file lists Johnson (J000299), Jeffries (J000294), Scalise (S001176), Thune (T000250), Schumer (S000148) and the other leaders with their current titles |
| Legal | The Ethics in Government Act forbids using these reports for commercial purposes (other than news media), credit ratings or soliciting money. Personal use is fine |
| Status | H.R. 7008 (restricting member trading) passed the House on 22 Jul 2026 (232–198) but **failed Senate cloture on 30 Sep 2026 (53–47)**. The STOCK Act disclosure regime is still in force |

## 5. High win-rate traders: eToro Popular Investors (optional, **off by default**)
| | |
|---|---|
| Source | **eToro official Public API** (`public-api.etoro.com`), which needs your own verified eToro account. Headers: `x-api-key`, `x-user-key`, `x-request-id` (UUID) |
| Leaderboard | `GET /api/v2/portfolios/rankings`: win ratio (% of profitable closed trades), gain, risk score 1–10, copiers, profitable months %, peak-to-valley drawdown |
| Trades | `GET /api/v1/user-info/people/{username}/portfolio/live`: open positions with open time and price. New and closed positions are detected by diffing snapshots |
| Limits | 60 requests/minute per user key for reads (20/min for writes) |
| Caveats | eToro's 8 Oct 2026 release says its API connectors are *not currently available to US clients*. eToro's terms allow personal use only, with no redistribution, and this app's website is public. So the integration ships **disabled**; enable it only if your account has API access and you accept that trade-off |
| Why not TipRanks / Quiver rankings? | TipRanks has no free data API (its MCP free tier is 10 calls/month) and its terms forbid scraping. Win rates for insiders, Congress and funds are instead **computed by the app itself** from the official filings (see 7) |

## 6. Prices, risk metrics and company size
| | |
|---|---|
| Daily history | **Massive** (formerly Polygon.io) free Basic plan: the *grouped daily* endpoint returns every US stock's end-of-day bar in one request (5 requests/min, past days only, ~2 years). It backfills in a few hours, then needs one request per night. This powers volatility, drawdown, beta, 52-week range and every win-rate calculation |
| Deep history (for the backtest) | **Alpaca** free Basic plan, historical bars back to 2016 from all US exchanges (the consolidated SIP feed, free for everything but the latest 15 minutes), 200 requests/min. Built once: Alpaca's asset list (listed stocks and ETFs, and the inactive ones it knows) plus, with the Massive key, every stock, ADR, ETF and ETN Massive's reference list shows as delisted since 2016 (`/v3/reference/tickers?active=false`; Alpaca has no list of older delistings). Those whose company still trades on a main exchange under a new ticker (same SEC CIK, from the SEC's ticker file) are left out, since their bars come with the new ticker. A monthly sweep keeps every stock that traded at least $100M in some month (a net wide enough for every stock the backtest could pick on any date, so the choice never depends on how a stock did later), then their split-adjusted daily bars are fetched (`/v2/stocks/bars`, `feed=sip`, `adjustment=split`; delisted ones with `asof` set to the month before they went, so a reused ticker isn't mixed up). Only the backtest reads it; nothing from it is published |
| Latest price | **Alpaca** free account: IEX real-time snapshots, labelled *indicative* (IEX is ~2–3% of volume). The free plan's *real-time* data is IEX only; its full-market SIP data is free only once it's 15 minutes old, which suits the deep history above but not a live price. Without Alpaca, a rotating set of tickers is refreshed from Yahoo's unofficial chart endpoint (labelled as such), and otherwise the last close is shown |
| Splits | **Massive** `/stocks/v1/splits` (included in every plan, 2 years of history on the free plan, updated daily; the older `/v3/reference/splits` is deprecated). The app records the day each price row was downloaded and re-bases only rows fetched before a split, verifying against the first post-split day, so a split never looks like a crash in risk meters or win rates |
| Size / profitability | SEC XBRL **frames** API: one request returns a concept for every company (shares outstanding, net income) |

## 7. Win rates computed by the app (official data + prices)
* **Insiders**: SEC *Insider Transactions Data Sets* (quarterly ZIPs of every Form 3/4/5). The download path moved in 2026 (`/files/datastandardsinnovation/...` from 2026 Q2; older files under `/files/structureddata/...`); both are tried. The `AFF10B5ONE` column isn't in the official readme, so it's used only when present. For each open-market purchase: buy at the close on the first trading day *after the filing date* (what a copier could actually do) and measure the return 63 trading days (~3 months) later and the return vs. S&P 500 (SPY) over exactly the same days (SPY's trading calendar is used for both legs).
* **Congress**: the same method applied to disclosed purchases, using the disclosure date. Every purchase the app sees is added to a long-term history at the moment it's ingested, so records keep growing after trades leave the 120-day feed.
* **Funds**: "copy the 13F" backtest: the top positions bought at the filing date and held to the next filing, vs. SPY.
* Survivorship: tickers without price history (often delisted) are excluded, and the share excluded is shown. This flatters win rates a little, and the app says so.

## 8. Evidence behind the signal score
* Open-market **purchases** carry information; most **sales** don't (diversification, taxes, plans): Lakonishok & Lee (2001).
* **Opportunistic** (non-calendar) insider trades earned ~82 bp/month while **routine** trades earned ~0: Cohen, Malloy & Pomorski (2012, *J. Finance*).
* **Cluster** purchases (several insiders) earn roughly twice the abnormal return of solitary buys. Executives' clusters are stronger than directors'.
* The signal is stronger in **smaller companies** (information asymmetry).
* Congress: rank-and-file members don't systematically beat the market, but members in **leadership** have outperformed after taking office (CEPR/VoxEU, 2025). Returns fell after the 2012 STOCK Act.
* 13F copying works best for **concentrated, low-turnover** managers and **large positions** (~7.5%+ of the fund); high-turnover funds are stale by filing time.

## 9. Market context, short interest, earnings dates and the Claude API
| | |
|---|---|
| Market backdrop | Computed from the whole-market history: S&P 500 vs its 50/200-day averages, % of liquid stocks above their 200-day average, new 52-week highs vs lows, 11 sector ETFs plus QQQ, IWM, TLT, GLD and HYG |
| Macro (optional key) | **FRED** (St. Louis Fed) API, 120 requests/minute: VIXCLS (Cboe, citation required), DGS10 and DGS2 (Federal Reserve H.15, public domain), T10Y2Y (citation required), DFF (public domain). ICE's high-yield spread (BAMLH0A0HYM2) is deliberately **not** used: its licence forbids reproduction and this app's page is public, so credit stress is read from the HYG ETF price instead. The app shows FRED's required notice and source, keeps only the latest values, and doesn't train any model on FRED data (both restricted by FRED's API terms) |
| Short interest | **FINRA**, free, no key: one file per settlement date for every listed stock (`cdn.finra.org/equity/otcmarket/biweekly/shrtYYYYMMDD.csv`), positions as of mid-month and month-end, published ~7 business days later. FINRA's proposal to report weekly was withdrawn, so twice-monthly stands. The parser accepts either delimiter and matches column names flexibly; the exact live header couldn't be test-downloaded from the build sandbox |
| Earnings dates (optional key) | **Alpha Vantage** `EARNINGS_CALENDAR` (free key, 25 requests/day; the app uses 1/day for every company). No reputable free source of point-in-time consensus estimates exists, so the app doesn't show "beats/misses" |
| Claude API | Messages API with `output_config.effort` (low → max), Anthropic's web search tool, strict tool use for the answer, prompt caching for the playbook. Prices (checked on Anthropic's pricing page, 30 Oct 2026): Opus 5.5 $4/$20, Fable 5.1 $10/$50 per million input/output tokens; cache reads $0.20/$0.25; web search $10 per 1,000. A claude.ai subscription doesn't include API usage, but Max 5x/20x plans include $100/$200 a month of API credit |

## 10. Delivery
* Workflow actions: GitHub removed Node 20 from its runners on 16 Sep 2026, so every action is pinned to a Node 24 major (`checkout@v6`, `setup-python@v6`, `cache@v5`, `configure-pages@v6`, `upload-pages-artifact@v5`, `deploy-pages@v5`). Python packages are pinned to the exact versions the tests passed with.
* Hosting: GitHub Actions (cron every 5 min, the platform minimum, best-effort, may be delayed under load) + GitHub Pages (custom-workflow deploys are exempt from the 10-builds/hour limit; 1 GB site, 100 GB/month soft bandwidth). Scheduled workflows are disabled after 60 days without repository activity, so a monthly keep-alive commit is included.
* Alerts: **Telegram** bot (most reliable) or **ntfy** (free; 250 messages/day per *IP address*, and GitHub runners share IPs with other users, so add a free ntfy account token for your own quota). Alerts are bundled per run and capped per day.
* GitHub's terms say Actions shouldn't be used for activity unrelated to building and publishing the repository's project. This pipeline builds and publishes this repository's app, which is a common pattern ("git scraping"), but it's a gray area at a 5-minute cadence. Slow it to every 15–30 minutes in `.github/workflows/refresh.yml` if you'd rather be conservative.
* Phone app: installable web app (PWA). iOS 16.4+ / Android Chrome "Add to Home Screen"; iOS 26 opens home-screen sites as web apps by default.

## 11. Live-trade sources added in round four
| | |
|---|---|
| **Form 144 (planned insider sales)** | SEC EDGAR, official and free. Required before an insider sells more than 5,000 shares or $50,000 of restricted/control stock in 3 months; filed electronically as XML since April 2023 (root `edgarSubmission`, `submissionType` 144, elements `issuerInfo`, `securitiesInformation` with `noOfUnitsSold`, `aggregateMarketValue`, `noOfUnitsOutstanding`, `approxSaleDate`, `brokerOrMarketmakerDetails`, `securitiesToBeSold`, `securitiesSoldInPast3Months`, `noticeSignature/planAdoptionDates`). Live feed `type=144` plus the daily index backstop. Form 144 carries no ticker, so the issuer CIK is mapped through the SEC's ticker file. Mostly routine (compensation shares, 10b5-1 plans), which the scorer says; its value is timing: it arrives before the sale and before the Form 4 |
| **Transparent active ETFs (daily trades)** | The funds' own daily holdings files. ETFs relying on SEC Rule 6c-11 must post their full portfolio on their website each business day, so for an actively managed fund it is an official daily record of what the manager did. Default: ARK's six active funds (`assets.ark-funds.com/fund-documents/funds-etf-csv/…_HOLDINGS.csv`; columns `date, fund, company, ticker, cusip, shares, market value ($), weight (%)`, disclaimer lines after the table; the date in the file is the *next* trading day, i.e. positions at the start of that day). Only the ARKK file name could be verified from the build sandbox; the other five are tried under their current and former fund names and reported on the Data tab if none works. Trades are the day-over-day change net of fund flow (the median share change across holdings); ARK's own trade e-mails are the only exact record and aren't public files. **Terms:** ARK's holdings documents carry a reproduction notice and its website terms could not be read from the sandbox (the site returned an error), so this is the one source whose terms are unverified. The app republishes only the changes it computes (facts about a public fund's trades), with a link to ARK's file; the file is checked every two hours with a conditional request (`If-None-Match`), so it is downloaded only when it changes, about once a day; set `etf.enabled: false` if you'd rather not |
| **FINRA daily short-sale volume** | FINRA Reg SHO daily files, official and free (`cdn.finra.org/equity/regsho/daily/CNMSshvolYYYYMMDD.txt`, pipe-delimited `Date|Symbol|ShortVolume|ShortExemptVolume|TotalVolume|Market`, posted by 6pm ET the same day). FINRA's caveats, shown in the app: off-exchange (TRF/ADF/ORF) trades only, not consolidated with the exchanges, and not the same thing as the twice-monthly short interest; so the ratio is compared with its own 20-day average, never with 50%. ~500 KB per day; the app keeps 25 days for the stocks it shows |
| **Unusual volume** | Yesterday's consolidated volume vs its 20-day average, from the whole-market history already stored (Massive, end of day). A one-day lag; the free Alpaca feed is IEX-only (2–3% of volume), so it isn't used for this |
| **Claude in the app (Ask tab)** | The browser calls the Claude Messages API directly (`stream: true`, `output_config.effort`, web search tool `web_search_20250305`, prompt caching on the app context) with the user's own key and Anthropic's `anthropic-dangerous-direct-browser-access: true` header, which is the documented way to call the API from a browser for bring-your-own-key tools. The key is kept in the browser's local storage only; a content-security policy limits the page's connections to its own origin and api.anthropic.com. Costs are computed from the stream's usage reports with the same price table as the analyst |

## 12. Valuation, news and forecast inputs added in round five
| | |
|---|---|
| **SEC company facts (financials)** | `data.sec.gov/api/xbrl/companyfacts/CIK##########.json`: every XBRL fact a company has reported in its 10-Ks and 10-Qs, each with its period, form and filing date. Official and free, under the same fair-access rules as the rest of EDGAR (declared `User-Agent` with contact e-mail; the app stays under 8 requests/second across all SEC hosts). US GAAP concepts first, IFRS (`ifrs-full`) for foreign filers that use it; US-dollar facts only. Up to 150 companies per half-hour run, so the full set loads within the first day; a company's facts are re-fetched when it files a new 10-Q/10-K, or every 30 days. Since round nine, revenue growth, operating margin, free-cash-flow yield and share issuance from the latest statements filed before a forecast's date are forecast signals |
| **SEC submissions** | `data.sec.gov/submissions/CIK##########.json`: industry code (SIC), all tickers and share classes, first filing date (firm age for the bubble checklist), share offerings (S-1/S-3/424B) over 12 months, and every 8-K with its item numbers. Re-fetched every 3 days, or sooner when a new filing appears in the live feed |
| **8-K material events** | Item numbers come from the SEC itself (submissions `items`), not from parsing text, and are labelled in plain English. The live EDGAR feed (`type=8-K`) is polled so a new 8-K shows up within minutes. Red flags: 1.03 bankruptcy, 2.04 accelerated debt, 3.01 delisting notice, 4.02 non-reliance on past financial statements. Since round nine, 8-Ks with a material item in the 10 days before a forecast's close, and any red flag in the 90 days before, are forecast signals too |
| **Headlines: the GDELT Project** | DOC 2.0 API (`api.gdeltproject.org/api/v2/doc/doc`, `mode=ArtList`), free and without a key. GDELT's terms allow any use, commercial included, with a citation and link back to GDELT, which every stock page shows. Searched by company name in English-language sources, reputable financial outlets listed first. GDELT asks for at most one request every 5 seconds and answers rate limits as plain text with HTTP 200, which the app detects and backs off from. Each check asks for up to 250 articles (GDELT's most), and the page shows the best 8. Since round nine the headlines also feed two forecast signals: how many distinct headlines GDELT had seen in the week before the app's last check before a forecast's close, and their tone from a short word list (METHODS section 9). For that the app keeps, for each of the last few checks, only the count, the average tone and whether the answer was full, not the articles |
| **Massive news (not used)** | Massive's `/v2/reference/news` is in its plans, but individual plans are a personal, non-display licence and this app's website is public, so the app doesn't fetch it at all |
| **Event calendar** | FOMC decision dates from the Federal Reserve's meeting calendars and tentative-schedule releases, and CPI release dates from the BLS archive and schedules, built in from 2016 (for the backtest) through 2027; extra dates can be added in `config/config.yaml`. The two emergency Fed decisions of March 2020 are included (the Sunday one on the next session), as are the two CPI reports released on Good Friday (on the next session). Next earnings from Alpha Vantage if connected, otherwise estimated from the company's quarterly filing rhythm and labelled "estimated" |
| **Prices for forecasts** | The same split-adjusted daily history that powers the risk meters (section 6), so a split never looks like a crash in the model, the backtest or the grading |
| **Fonts** | IBM Plex Sans (SIL Open Font License 1.1), bundled with the app so it works offline; the licence is in `web/fonts/OFL.txt` |

## 13. Long price history for the learning (researched 1 Nov 2026)
The learners' tests need many independent months, so the free sources of long daily history were compared:
| Source | Free history | Coverage | Verdict |
|---|---|---|---|
| **Alpaca** Basic plan | back to 2016, 200 requests/min | US stocks and ETFs; its asset list has only the delisted stocks it once traded, so Massive's reference list fills in the rest | **Used**: the deep history (section 6) |
| **Massive** Basic plan | 2 years, end of day | every US stock, one request per day | Used: the main daily history |
| **SEC Insider Transactions Data Sets** | 2006 to now, quarterly, public domain | every Form 3/4/5 | Used for insider win rates (section 7). The next step for the learning: historical insider-buying signals for the backtest, which first needs a complete store of live insider trades so the live and historical definitions match |
| MarketParquet | the last ~365 days free; the full archive back to 2000, delisted stocks included, for a one-time $79 | US stocks, ETFs, futures | Not used (paid); the simplest upgrade beyond 2016 |
| Tiingo free plan | decades per stock | 500 different symbols a month | Too few symbols for a whole-market backtest |
| Stooq | decades, as free bulk downloads | stocks, ETFs, indices, many markets | No API (CSV files only), so it can't be refreshed automatically |
| Yahoo Finance | decades per stock | listed stocks only | Unofficial and drops delisted stocks, so a backtest on it is flattered by survivors; used only as the per-stock fallback |

## Audit log
How the data choices were checked (30 Oct 2026):

1. **Primary-source research.** I researched each data category against official documentation first (SEC EDGAR fair-access and API pages, SEC filing calendars and 13D/G rule changes, the House Clerk and Senate eFD sites, and the eToro, Alpaca, Massive, OpenFIGI, ntfy and GitHub docs), then compared third-party alternatives on freshness, cost and reliability. Aggregators were rejected wherever an official source is reachable, because they can't be fresher than the source.
2. **Independent audit.** A separate research agent that hadn't seen my reasoning re-verified 20 specific claims plus all 27 fund CIKs against primary sources, looking for contradictions. Result: 9 verified as written, 3 contradicted, the rest partly right or unconfirmed. Every correction was applied:
   * Alpaca paper accounts get IEX-only data → history moved to Massive's whole-market endpoint; Alpaca now only supplies an *indicative* real-time price. (Refined 1 Nov 2026: that holds for real-time data; Alpaca's free plan does serve full-market SIP bars once they're 15 minutes old, which the deep history uses.)
   * eToro API not currently offered to US clients, and its terms forbid redistribution → integration ships disabled.
   * Senate eFD returned 403 to GitHub Actions for a full week → Quiver documented as the dependable fallback; Quiver's auth header corrected to `Token`.
   * SEC moved the insider data-set path in 2026 → new path first, old path as fallback.
   * Fund CIKs: Pershing Square and Greenlight moved filers; Scion deregistered → fixed. Name checks now accept former names and ignore "LLC/MA"-style suffixes.
   * Senate cloture failure on H.R. 7008 (30 Sep 2026) added. Cron moved off the top of the hour. ntfy shared-IP caveat added.
3. **Real-document tests.** A third agent collected 50 real filings from public test suites (SEC Form 4s across schema versions X0306/X0508/X0609, 13F tables in both units, 2024–2026 Schedule 13D/13G XML, a real encrypted House PTR PDF, real Senate eFD pages, live EDGAR Atom feeds). Every parser was run against them. That caught and fixed: a 13G with a blank line before `<?xml`, mixed `true/false` vs `0/1` 10b5-1 flags, wrapped House asset names and amounts, and 13F unit mislabelling.
4. **Independent code review.** A fresh agent that hadn't seen the build reviewed the whole pipeline and app cold and reported 21 issues (5 high, 9 medium, 7 low). All were fixed, each with a regression test (`tests/test_fixes.py`; 57 tests pass in total). The important ones: one malformed filing could stall a feed; the GitHub cache rule silently included the large price archive in the small state cache (confirmed by running GitHub's own `@actions/glob` library on both patterns); splits weren't applied to stored history; a fund's first 13F would have shown every holding as "new"; Congress leaders were matched by surname; the Senate bot wall could return a web page that looked like "no trades"; win-rate windows could drift from SPY's on stocks with gaps. Fixing them turned up one more: the missed-filing check could skip a business day; it now walks forward day by day until it's caught up.
5. **Second live check (same day).** Re-verified facts that change: found GitHub's Node 20 removal (would have broken the workflow) and moved to Node 24 actions; confirmed Massive's current splits endpoint and free-plan access; downloaded the legislators file to confirm leadership IDs and titles.
6. **Third round: the analyst and predictability layer (same day).**
   * *Primary-source checks:* Anthropic's pricing, effort, web-search and structured-output docs (models, IDs, prices,
     `pause_turn` handling, effort levels per model, Max-plan API credits).
   * *Independent research audit:* a separate agent checked 5 data sources and 18 research findings against primary sources
     (journal pages, NBER/SSRN, official data pages). Changes it caused: ICE's credit-spread series dropped (licence), FRED
     notice and minimal caching added (FRED terms), activism, post-earnings drift, buyback drift and Congress claims
     rewritten to reflect their decay or mixed evidence, the 2025 Congress-leaders result labelled as an unreviewed working
     paper, FINRA short interest added (free, official), and no consensus-estimate data used (no reputable free source).
   * *Statistical self-tests:* the model gate was run on simulated markets with planted signals and with pure noise. The first
     version claimed skill on noise in 2–3 of 12–16 trials; same-week comparison, week-clustered bootstrap, the both-halves
     rule, one pre-registered model and a minimum test size brought that to 1 in 24 while still catching 24 of 24 planted
     signals. The independent review below then found noise designs it still fell for; after the fixes it is 0 of 24 on noise
     and 24 of 24 on planted signals (`docs/METHODS.md` §2).
   * *Independent code review* of the new code, with every finding fixed (item 8).
7. **Automated checks every run.** Each job's last success, error and item count are published to the app's *Data* tab. Fund CIKs are re-validated against SEC names, units are sanity-checked, and every trade links to its original filing so any number can be checked against the source.

8. **Second independent code review (analyst and predictability layer).** A fresh agent that hadn't seen the build reviewed
   the new code cold and backed each claim with a runnable experiment. It reported 21 issues (2 high, 10 medium, 9 low). All were
   fixed, each pinned by a regression test (`tests/test_review2.py`; 101 tests pass in total), and its own experiments were
   re-run against the fixes:
   * **Claude billing (high).** A billed search turn followed by a failed turn lost its cost from the budget; a timed-out
     request could be re-sent automatically (billed twice); and 48,000-token non-streamed calls could outlive their timeout.
     Now every billed turn is charged, nothing is re-sent once it may have reached the API, lost responses are charged at an
     estimate, output is capped at 20,000 tokens. Re-run: the persisted spend matched the true cost exactly ($24.00 of $24.00;
     before, failed tasks were charged $0 for $2.82 of real spend).
   * **Budget drift (high).** Reservations for running calls were released at a re-estimated amount; now released exactly
     and never saved. Re-run: drift $0.00.
   * **Over-confident statistics (medium).** Base-rate and ledger intervals treated filings in the same months as independent
     (true-rate coverage 31% → now 94%); the momentum backtest's interval ignored overlapping windows (false "significant"
     19% → now 2.7%); the model gate could be fooled by buying sprees, size-factor swings and daily re-testing (now 0 of 40,
     0 of 60 and 0 of 120 looks; a real edge of AUC ≈ 0.62 still passes 6 of 6).
   * **Data alignment (medium).** Live model features didn't match the training rows (per-day aggregation, 30-day cluster,
     10b5-1 source); the routine/opportunistic test called buyers "opportunistic" when the data didn't reach back 3 years;
     the held-out calibration band was looked up by raw score on a differently-scaled refit model; the "buying into weakness"
     screen showed a track record of a different rule and measured the fall with today's price.
   * **Ledger integrity (medium).** Entries could be backdated to when a trade was first seen, a corrected Form 4/A or a
     returning idea could get a second entry. Now dated when written, with stable ids and a 30-day idea registry.
   * **Workflow (medium).** The analyst could start from an empty memory after a cache hiccup and reset the day's budget; it
     now checks whether a saved copy exists and skips the run instead.
   * **Low:** delisted stocks were dropped (now graded to their last trade, and kept in the screen's backtest universe), the
     train/test gap is 100 days, NaN can no longer reach the published JSON, setups are graded at their own horizon, hit rates
     only appear when probabilities vary, the refresh workflow no longer receives the Claude key, an unquoted brief time in the
     YAML config works, idea write-ups get their own prompt, and four display details were corrected.

9. **Fourth round: the chat box and the live-trade sources (same day).**
   * *Primary-source checks:* Anthropic's streaming event reference (event order, `input_json_delta`, `pause_turn`, usage in
     `message_delta` being cumulative), the web search tool reference (`max_uses`, result and error shapes, citations,
     `encrypted_content` must be returned unchanged, the 2026 tool versions' dynamic filtering and its extra billing), the
     browser-access header, the Form 144 XML structure (several real filings), ARK's file location and columns (an October 2026
     report of the live file), FINRA's daily file layout and posting time.
   * *Decisions:* the analyst now defaults to the basic web search tool (the 2026 versions run code to filter results, which
     adds code-execution billing; still selectable in config); aggregator websites were not added as sources, because they
     display the same official filings later and their terms forbid automated use; options-flow and dark-pool feeds were not
     added, because they are paid, licensed data.
   * *Tests:* parsers against fixtures that mirror the real files (both Form 144 namespace styles, ARK's trailing disclaimer
     and cash rows, FINRA's pipe format), the sources end to end offline, and the chat box in a real browser against a mock of
     the streaming API (search pause and continuation, citations, cost arithmetic, error messages, persistence, the
     content-security policy).
   * *Independent code review* of the new code (a fresh agent, cold, with its own experiments against a mock of the
     streaming API and real EDGAR filings): 1 high, 6 medium, 13 low findings, all fixed and pinned by tests:
     - **High:** the new content-security policy blocked the app's own "‹ Back" links (they were `javascript:` URLs) → a
       real button.
     - **Medium:** a short or partial holdings file would have produced a wave of false "sold out" events and poisoned the
       snapshot (now ignored when it has under 70% of yesterday's rows); an older file served after a newer one rolled the
       snapshot back and re-emitted trades (now ignored); a Form 144/A could replace the wrong notice when a person had
       several open (now uses the `previousAccessionNumber` the amendment names); the chat's output limit was too small for
       maximum effort, which thinks a lot (now 64k tokens at `max`/`xhigh`, streamed); the chat's cached context block
       carried a millisecond timestamp, so the cache never hit between questions (the time now travels in the user turn
       and the context stays byte-identical until the data refreshes); a stream that died before the API said why was
       shown as a complete answer (now flagged "connection dropped: may be incomplete").
     - **Low:** a Stop button (the only way to stop an answer was to wipe the chat), no ghost "Stopped." after New chat,
       the unusual-volume average used 21 days instead of 20, ETF trade dates now use the market calendar (holidays), a
       late or failed FINRA file is retried for two days before it's treated as a holiday and stocks that enter the feed
       later get their short-volume history backfilled, the learned base rate for sale notices is now shown, same-family
       ETF "consensus" counts less and only within one manager, fenced code and tables render in the chat, the stream
       reader is cancelled after an error, the chat warns Claude that app data and web pages are untrusted, the key's
       shared-origin caveat on GitHub Pages is documented, and the config comments and refresh keys were corrected.
   * *Second audit pass:* a second fresh reviewer checked every fix with its own experiments and found one regression
     the first fixes had introduced plus seven smaller issues, all fixed and tested: the ETF trade date relied on the
     stored market calendar, which always lags a day or two, so an ordinary Monday file was dated to the Friday before
     (now: the calendar decides for the days it covers; beyond it, weekends and NYSE holidays by rule are skipped);
     stopped chat answers were saved with no cost (now priced from the usage seen so far, marked "at least"); the Back
     button could leave the app when the tab had earlier history (now bounded); a mildly cut-off holdings file could still
     pass the 70% row gate (now the weights must also add up, relative to the previous file) and a fund that really
     shrank would have been frozen for ever (now accepted once the same file has been served for two hours); a chain of
     144/A amendments naming the original would have duplicated cards (now matched through the `amends` link, and the
     frozen price context carried over); a continued search turn could glue a heading onto the previous sentence; the
     short-volume backfill skipped the very first run's tickers; placeholder answers were being sent back as Claude's own
     words in later turns; the ETF file is now fetched with a conditional request (`If-None-Match`) instead of a full
     download every two hours. Final run: 114 Python tests and the browser test pass; linters clean.

10. **Fifth round: valuation, bubble risk, price forecasts and their report card (30–31 Oct 2026).**
   * *Primary-source checks:* Greenwood, Shleifer & You (2019, "Bubbles for Fama") for the run-up thresholds behind the
     bubble checklist; Gu, Kelly & Xiu (2020) for what direction accuracy is realistic for single stocks; Gneiting & Raftery
     (2007) for the proper scoring rule used to grade ranges; Barone-Adesi, Giannopoulos & Vosper (1999) for filtered
     historical simulation; the BLS CPI release schedule and the Federal Reserve's FOMC calendar for event dates; the GDELT
     terms (citation and link required, shown on every stock page); the SEC company-facts and submissions APIs; Massive's
     licence (its news feed is personal, non-display, so it isn't fetched); the current major versions of every GitHub action.
   * *Experiments before trusting the learning rules:* the signal-tilt gate was run on simulated markets: one weekly test
     passed in 0 of 40 worlds with no real signal and found a planted signal in 20 of 20.
   * *First review round:* three fresh reviewers (forecasting method, pipeline, app) found 55 problems (17, 16 and 22), all
     fixed and pinned by `tests/test_review3.py` and the browser tests. Among them: the backtest only used today's
     survivors (now each past date uses the stocks that were liquid then, delisted ones included); an earnings-date proxy
     peeked at later data; valuations mixed balance-sheet dates and ignored second share classes; share counts weren't
     adjusted for splits; Claude's forecasts could be graded from a close that came before news it had read; free-text
     lessons from post-mortems could carry instructions from a web page into later prompts (now only patterns from fixed
     lists); a slow stock page could paint over the next one; chart dates were off by a day west of UTC.
   * *Second review round:* two more fresh reviewers re-checked every fix. Backend: two of 33 fixes were incomplete and
     two new defects had crept in. App: 12 fixed, 4 fixed with caveats, 6 partly fixed, 9 new (mostly small). All were
     fixed and pinned by `tests/test_review4.py` and new browser checks: the calibration hold-out now leaves a gap so no
     fitted outcome overlaps a tested one; a refused market-data key fails loudly; the self-check now catches stalled
     forecasts and failing every-run jobs; the large cache (price history and the forecast record) can't be overwritten by
     an empty one; the site is built in the publishing job from a clean checkout; market caps use split-adjusted share
     counts; a paid phone forecast keeps running if you leave the page, can't be started twice, and shows its cost even when
     it fails; the app says when it's showing the saved offline copy and when the phone's clock disagrees with the data;
     keyboard focus stays put when a control redraws the page; red text on dark chips now meets 4.5:1 contrast.
   * *Third review round:* two more fresh reviewers checked the second round's fixes with their own experiments and
     found 16 more problems (8 pipeline, 8 app), all fixed, pinned by tests, and the app reviewer's experiments re-run on
     the fixed code. Pipeline: on a split's execution day a pre-split price could be paired with post-split shares (a
     10× market cap for a day); the coverage interval wasn't clustered when it rested on fewer than three periods and came
     from a different sample than the number beside it; the split pause fired on ordinary −20%, −25% and +33% earnings days
     (now only with the split records' confirmation or an unmistakable ratio); earnings timing read the SEC's New York
     timestamps as UTC; switching forecasts off would have raised a false "stalled" alarm and kept old forecasts up; some jobs
     weren't judged against their own cadence; the hold-out gap ignored holidays; Claude's earlier wording still reached
     post-mortem prompts outside the untrusted block. App: on a slow connection it re-downloaded every data file every
     minute (re-measured: one small request per check); a partly cached update could leave old prices on screen under a
     green header (re-measured: redrawn with the new prices); a phone forecast that failed midway under-reported its cost
     and failed silently if you'd left the page (re-measured: cost counted, failure announced); a failed refresh said
     "You're up to date"; restoring keyboard focus could pull it out of the chat box; a slow stock page could pin an old
     copy; the range warning quoted a number from a different sample; and three smaller focus gaps.
   * *Final run:* 196 Python tests, both browser tests (chat, and the app in light and dark with the forecast chart,
     records, self-check, offline mode and a paid forecast against a mock API), pyflakes, actionlint and node syntax checks
     all pass.

11. **Sixth round: does the forecasting actually learn? (31 Oct 2026).**
   * *Method:* the learning loop was traced end to end in the code (grading → calibration, signal learning and blend weights →
     stored → used by the next forecasts), then run exactly as the pipeline runs it over 400 trading days of simulated markets
     (36 stocks; a market factor, volatility clustering, fat tails, earnings jumps the model isn't told about, volatility steps),
     with every forecast also made without learning and both scored on the same outcomes. Worlds: no real signal, bigger
     earnings jumps than the history shows, a sustained volatility step-up, and weak and strong planted signals.
   * *Found:* the wiring was complete, but the learning didn't reliably improve accuracy. Calibration counted every day as new
     evidence, though a month of daily forecasts shares most of its outcome and every stock rides the same market, so it chased
     noise and made 1-month forecasts up to 4.7% worse in some quarters. The signal learner could not start for about ten
     months, kept only half of any effect however much data it had, filled missing values differently when fitting and when
     applying, and in the no-signal world switched on by chance.
   * *Fixed:* evidence is counted in independent months; a calibration is used only if it would have improved the newest
     forecasts it wasn't fitted on; the signal learner predicts how each stock does relative to the others that day (removing
     the market's own move), uses momentum and last month's move relative to the S&P 500, learns from the backtest from the
     first day, keeps two years of 1-month records, and shrinks less as evidence grows; older feature definitions are never
     mixed in. Pinned by tests in `tests/test_forecast.py` and `tests/test_review4.py`; the before/after results are in
     METHODS section 9.

12. **Seventh round: the strongest learning the data can support, audited three times (31 Oct – 1 Nov 2026).**
   * *Change:* the single ridge regression became a weekly competition between ridge regression, gradient-boosted trees, a
     neural network and their average. Each is tested month by month on forecasts it never saw, and the choice of
     algorithm is itself tested that way (nested walk-forward selection). A second learner was added for range widths
     that depend on the stock's situation, and the signal set gained volatility, volume, 52-week-high and short-term
     reversal features. Every stock with a forecast is now listed on a searchable *All forecasts* page
     (`data/forecasts.json`).
   * *Audit 1 (independent statistics review):* no leakage that mattered. The weak points were:
     * too few independent months for the test to tell a weak signal from luck;
     * a target and scoring not fully stock-versus-stock (beta could learn one period's market move; predictions weren't
       centred or capped as deployed);
     * a significance bar too loose when test months are few, and a second weekly pass that could re-test the same months;
     * period counts that flipped as the window moved;
     * an in-sample constant in the range learner, which would make ranges too narrow;
     * the backtest seeing earnings dates up to 35 days ahead.

     All fixed. Records are kept longer and the S&P 500's always. The target removes each stock's beta times the market's
     own surprise and the day's average. Scoring matches use. The bar is Student's t at 99.5% (4.0 with 6 test months)
     and the second pass needs a newly completed month. The constant is computed out of fold. The backtest now runs
     weekly on calendar-fixed dates, with earnings dates known two weeks ahead.
   * *Audit 2 (independent app review):* the new list's search box broke phone keyboards that compose words, the list
     jumped to the top on refreshes and Back, a saved offline copy could stick, chances read like price moves ("+50%"),
     and a learner that failed to rebuild was cached silently. All fixed.
   * *Audit 3 (independent check of the fixes):* confirmed them and found more:
     * a day without the S&P 500's own record let beta back into the target (now estimated from that day's stocks);
     * the S&P 500 itself got a stock-versus-stock nudge;
     * live and backtest rows for the same stock and day counted twice;
     * a one-day stub of a new month counted as a test month;
     * the row cap split days and months;
     * the coverage warning rested on up to four years of data;
     * the backtest queue stored row positions that shift when history is added.

     All fixed. Each finding is pinned by a test in `tests/test_ml.py`, `tests/test_review3.py` or `tests/browser/test_app.js`.
   * *Simulation check:* the same four simulated markets were run with the previous version and this one, with every
     forecast also made without learning (METHODS section 9 has the full table):
     * with no real signal, the signal learner never switched on;
     * with planted signals, the competition picked ridge for a straight-line effect and the average of the three for one
       that bends, improving every quarter (1-month forecasts by 4.1% and 2.7%, against 3.9% and 2.0% before);
     * with bigger earnings jumps than in the history, the range learner improved 1-week and 1-month forecasts by about
       1.8%, where the previous version changed nothing.

13. **Eighth round: more algorithms, a coverage keeper and ten years of prices, audited four times (1 Nov 2026).**
   * *Change:* four candidates joined the signal learner's competition: robust (Huber) regression, spline regression, a
     random forest and CatBoost (eight, with the average). A third learner, the coverage keeper (adaptive conformal
     inference), adjusts range widths after each day's results. With free Alpaca keys the backtest reaches back to 2016
     (the deep history, section 6), with stocks delisted since then from Massive's reference list. The Fed and CPI
     calendar now starts in 2016 to match.
   * *Research:* free sources of long whole-market price history were compared (section 13). Alpaca's free plan is the
     only one found that is free, automatable and covers every US stock back to 2016; the SEC's insider data sets
     (2006 on) are the next step for the learning.
   * *Audit 1 (independent statistics review of the learners):*
     * the keeper's multiplier replaced the calibration of a horizon that had none of its own instead of multiplying
       it, and leaked into neighbouring horizons;
     * the algorithm chosen for a month used the month just before it, whose outcomes weren't all in yet;
     * the significance test's error bar missed slower runs of luck;
     * live and backtest forecasts of the same day were centred together, though they're different sets of stocks;
     * the spline's penalty was six times too weak for its six curve pieces, and missing values piled up on one knot;
     * the keeper gave a half-empty day full weight and scored every year of its replay equally;
     * calibration could switch on and off from one day to the next.

     All fixed: the keeper is its own factor, the choice uses only months whose outcomes were in, the error bar is
     also estimated with Newey-West, centring is per day and source, and calibration switches back on only with a
     0.2% edge. A realistic no-signal test (slow-moving signals, overlapping outcomes, fat tails) now runs with the
     suite.
   * *Audit 2 (independent review of the deep history and the backtest):*
     * Alpaca has no list of companies delisted before it knew them (the docs said it had), so delisted stocks now
       come from Massive's reference list, asked for as of the month before they went;
     * on older dates the filing record doesn't reach, filing-based earnings dates meant no earnings at all (now
       volume-spike estimates there, using only spikes up to that date);
     * the Fed and CPI calendar began in 2024; the NYSE holiday rules counted Juneteenth before 2022 and missed the
       two national days of mourning;
     * a long gap in a ticker's prices could join two companies' histories;
     * an Alpaca error that wasn't about one symbol still split the batch, a (nearly) empty download could be marked
       complete, and the matrix on disk wasn't tied to the build that made it;
     * the backtest made rows older than the record keeps (dropped at the next save);
     * on split-adjusted prices, the $5 floor kept out stocks that have split since.

     All fixed.
   * *Audit 3 (independent app review):* a broken CatBoost install would have stopped the whole pipeline; a learner
     decision made by an older version of the test was shown as on; the suite ran the full-size models (it now runs
     light ones, with one test of the full settings); the deep history had no row on the Data tab, ran every
     refresh, failed silently and read its arrays every run. All fixed.
   * *Audit 4 (independent check of the fixes):* confirmed the learner fixes and found nine more problems, all in the
     deep history:
     * keeping each year's 2,500 most traded stocks used how much they traded later, a look-ahead that kept the stocks
       that went on to do well. Now every stock that traded $100M in some month is kept: a net wide enough for every
       stock the backtest could pick on any date, so the pick depends only on what was known then;
     * a bankrupt company whose stock moved over the counter counted as renamed, and vanished;
     * one error from Massive ended its delisted list early;
     * a hole could open between the deep weeks and the Massive weeks as the Massive history grows;
     * a network error could store part of the Massive key in the build's progress file;
     * a damaged file was never fetched again;
     * delisted ADRs and ETNs were left out;
     * a failed sweep was retried every hour, forever;
     * without the SEC's ticker list, a renamed company could be counted twice.

     All fixed. Every finding of this audit, and the main ones of the first three, is pinned by a test in
     `tests/test_deep.py`, `tests/test_ml.py` or `tests/test_review3.py`.
   * *Simulation check* (METHODS section 9 has the full table): the four simulated markets of round seven, with the
     previous version and this one, every forecast also made without learning:
     * with no real signal, the signal learner never switched on;
     * with a straight-line signal, robust (Huber) regression won every week and 3-month forecasts improved 6.4%
       (5.9% before); with one that bends, the average won most weeks and CatBoost the rest (1 month: 2.4%, against
       2.7% before);
     * with bigger earnings jumps than in the history, forecasts improved in every quarter at all three horizons;
     * a weak signal was measured as harmful with two years of backtest and as helpful with ten (both stayed off: at
       t 1.0–1.9 it isn't clear enough to act on).
14. **Ninth round: a Results tab, a fused model, and news, filings and financial statements as inputs, audited three
    times (1 Nov 2026).**
   * *Change:* a Results tab grades every forecast against what happened and takes every miss apart (market, earnings,
     a one-day shock, volatility or trend), with what the learning loop does about each kind of miss. A ninth
     candidate, the fused model, averages the three algorithms with the best record so far and can be put in use after
     nine test months as a fusion. Headlines (GDELT), 8-K filings and the latest financial statements became signal
     learner inputs, and each stock's own record of misses (and of misses caused by earnings) range learner inputs.
   * *Audit 1 (independent review of the learning changes)* found ten problems:
     * SEC acceptance times are New York time with a misleading "Z", so an 8-K filed after the close counted as known
       at it;
     * the headline count depended on news after the close (GDELT's answer holds only the newest articles, and a
       check after the close let later ones push earlier ones out);
     * the nine-month rule held back the fused model from use but not from the month-by-month choices the test
       scores, so the test could pass on months the deployed model never had;
     * a range-learner decision from the previous release counted as "already on", lowering the bar to switch on;
     * the miss record took an outcome as known by the calendar, though a stock with gaps is graded later;
     * filing signals were zero, not blank, for companies without a filing record;
     * the statements' date came from a different record than their numbers;
     * cached forecasts were rebuilt whenever the would-be fused trio changed, even when it wasn't in use;
     * the fold cache and the decision check didn't cover the list of signals;
     * a CatBoost failure at fit time was dropped silently inside the fused model, and scored as a zero gain in the
       test.
   * *Audit 2 (independent review of the Results tab)* found twelve:
     * backtest misses were taken apart with the live price history, so the deep history's years were marked
       "no prices" for good;
     * the win rate came from the sample the ledger keeps, not every graded forecast, so it disagreed with the
       Records tab;
     * backtest misses used the S&P 500's move, not its surprise;
     * Claude's misses assumed a beta of 1;
     * the lists were cut across horizons before the page filtered them, hiding the slower horizons' misses;
     * misses not yet taken apart were counted as "no data" in one view and left out in another;
     * the "what changes" text claimed learners at horizons where they don't run, and applied the model's corrections
       to Claude's misses;
     * the month chart ignored the horizon filter;
     * a one-off volume spike counted as an earnings day;
     * nothing used the breakdown to improve forecasts;
     * a passing failure was stored as a permanent "no data";
     * and small wording, rounding and loading problems (the page's data was fetched on every app start).

     All of both audits' findings were fixed: SEC times are read as New York time; the news signals come from the last
     news check before the close; the nine-month rule applies to the tested choices, and the algorithm put in use must
     also pass on its own record; versions, dates and blanks are handled as described; the page counts every graded
     forecast from the monthly totals, keeps each horizon's newest misses, counts the undone ones apart, estimates
     beta from prices, accepts only spikes with a quarterly rhythm (marked as estimated), retries misses that lacked
     prices, and loads only when its tab is opened. The breakdown now feeds the range learner (each stock's share of
     misses caused by earnings). Each finding is pinned by a test in `tests/test_review.py`.
   * *Audit 3 (independent check of the fixes)* confirmed 15 of the 22 fixes, found 7 partly done, and found new
     problems:
     * the earnings-miss input wasn't point-in-time: the volume spikes standing in for earnings days were picked with
       later spikes in view;
     * on a weaker but still positive week, the signal learner kept running a model that week's test hadn't measured;
     * the "likely" interval and the "expected" direction were computed on different forecasts than the rates shown
       beside them;
     * a red flag older than a company's 30 newest 8-Ks (all the SEC's list keeps) was missed;
     * a saved copy of the results could be shown as the latest;
     * a CatBoost that kept failing would have stopped the weekly test for good;
     * a run without the stocks' context would have stored half-done diagnoses for good.

     All fixed, with tests. The simulations then found one more: shrinking a stock's earnings-miss rate toward the
     record's rate made it a stand-in for how long the stock had been followed (ranges at 1 month came out too wide);
     it is now shrunk toward zero.
   * *Simulation check* (METHODS section 9 has the full table): the four simulated markets of rounds seven and
     eight, re-run with this version, every forecast also made without learning:
     * with no real signal, the signal learner never switched on, and the worst quarter was −0.2% (−0.7% before);
     * with a straight-line signal, forecasts improved in every quarter (3 months: 6.5%); robust regression won every
       week and the fused model came second (+6.47% a month on unseen months, against +6.55%);
     * with a signal that bends, every quarter improved (3 months: 4.3%); the average of all won most weeks and the
       fused model, which had picked the three algorithms that can bend, came second (+5.60% against +5.69%);
     * with bigger earnings jumps than in the history, forecasts improved in every quarter at all three horizons.

Still unconfirmed (and handled defensively): which FMP plan includes Senate data; Tiingo/Finnhub limits (not used); the EDGAR feed's maximum page size (the app uses 100); whether `AFF10B5ONE` appears in every data-set file (optional column); the exact file names of five of ARK's six holdings files and ARK's website terms (see section 11).
