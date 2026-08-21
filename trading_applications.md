# Trading Applications of GSJ Bandwidth Selection

Comprehensive list of applications in order flow, Level 2 DOM, and options analytics where multivariate (or 1D weighted) KDE with data-driven bandwidth applies.

---

## Completed Demos

| # | Application | d | Data Source | Status | Notebook |
|---|------------|---|-------------|--------|----------|
| 1 | Volume Profile (1D weighted) | 1 | yfinance 1-min | ✅ Done | `volume_profile_notebook.ipynb` |
| 2 | Dynamic Volume Profile (rolling) | 1 | yfinance 1-min | ✅ Done | `dynamic_volume_profile_v2.ipynb` |
| 3 | Prominence-encoded levels | 1 | yfinance 1-min | ✅ Done | `volume_profile_v3.ipynb` |
| 4 | Options OI surface (strike × DTE) | 2 | yfinance options | ✅ Done | `options_visual_notebook.ipynb` |
| 5 | Order flow regimes (4D bar features) | 4 | yfinance 1-min | ✅ Done | `orderflow_notebook.ipynb` |
| 6 | Dealer Gamma Exposure map | 1-2 | yfinance options | 🔨 Building | `gamma_exposure_notebook.ipynb` |

---

## Options-Based Applications

### Directly Buildable from yfinance

| Application | d | Features | What modes reveal | Trading signal |
|------------|---|----------|-------------------|----------------|
| **Options OI surface** | 2 | (moneyness%, DTE) weighted by OI | Where positions are concentrated | Support/resistance from hedging flow |
| **Dealer GEX (gamma exposure)** | 1 | Price levels weighted by gamma×OI | Where dealer hedging is most intense | Pin levels, resistance/support from hedging |
| **Put/call OI split** | 2 | (moneyness%, DTE) separate surfaces for P vs C | Asymmetry in positioning | Directional bias (more puts = fear, more calls = speculation) |
| **IV clustering** | 2 | (moneyness%, impliedVol) | Where vol is cheap/expensive in strike space | Relative value: sell rich vol clusters, buy cheap ones |
| **Options flow freshness** | 3 | (moneyness%, DTE, volume/OI ratio) | New activity vs stale positions | High vol/OI = fresh flow = smart money acting NOW |
| **Max pain landscape** | 2 | (strike, total_dollar_pain_at_strike) for each expiry | Where aggregate OI pain is highest per expiry | Pin targets for expiration week |
| **Cross-expiry gamma** | 2 | (strike, DTE) weighted by gamma×OI | Full 2D gamma surface, not just spot slice | Shows gamma walls that only exist at specific expiries |
| **Earnings implied move** | 3 | (moneyness%, DTE_to_earnings, OI) | Pre-earnings positioning clusters | Detect the "expected move" priced into options |
| **Skew term structure** | 2 | (DTE, 25Δ_put_IV − 25Δ_call_IV) | How fear evolves across time | Kink points = regime changes in skew |
| **Volatility regime** | 3 | (realized_vol, IV_percentile, put/call_ratio) | Joint density of vol indicators | Modes = complacency/hedging/panic regimes |
| **Net delta exposure** | 2 | (strike, net_delta×OI) per expiry | Directional tilt from options positioning | Where the market is "leaning" long or short |
| **Unusual activity detection** | 3-4 | (moneyness%, DTE, vol/avg_vol, size) | Anomaly: low-density points in activity space | Unusual trades = potential informed flow |
| **Spread strategy clustering** | 3-4 | (width, DTE, moneyness_center, debit/credit) | What spread structures are popular | Crowded trades = potential unwind risk |
| **Expiry concentration** | 1 | DTE values weighted by total OI at that expiry | Which expiries dominate | Gamma events approaching (OPEX timing) |

### Requires Tick/L2 Data (Polygon, Bookmap, Exchange Feeds)

| Application | d | Features | What modes reveal | Trading signal |
|------------|---|----------|-------------------|----------------|
| **Book imbalance clustering** | 3 | (price_level, bid_size/ask_size, depth_rank) | Structural order book imbalances | Hidden S/R from resting orders |
| **Iceberg detection** | 2-3 | (trade_size, refresh_rate, price_at_fill) | Patterns of hidden large orders | Institutional accumulation/distribution |
| **Trade classification (HFT/inst/retail)** | 4-5 | (size, inter-trade_time, price_impact, spread, aggressor) | Distinct flow archetypes | Know who you're trading against |
| **Spoofing detection** | 3-4 | (order_size, time_alive, distance_from_BBO, cancel_rate) | Patterns of manipulative orders | Avoid false signals from fake orders |
| **Liquidity regime** | 3 | (spread, depth_at_touch, trade_arrival_rate) | Distinct liquidity states | Adapt execution strategy to current state |
| **Price impact profile** | 2 | (trade_size, realized_impact_bps) | How size moves price | Optimal execution: size your orders per regime |
| **Quote stuffing detection** | 3 | (quote_rate, quote_depth, duration) | Abnormal messaging patterns | Identify noise vs real price discovery |
| **Dark pool activity** | 2-3 | (size, price_relative_to_NBBO, time_of_day) | Where off-exchange fills cluster | Institutional interest zones |
| **Aggressive vs passive flow** | 3 | (trade_side, size_percentile, time_clustering) | Distinct aggression patterns | Momentum vs mean-reversion signals |
| **Fill quality clustering** | 3 | (slippage, time_to_fill, size_fill_ratio) | Execution quality regimes | Detect when execution is degrading |

### Hybrid (Options + Price Action + Volume)

| Application | d | Features | What modes reveal | Trading signal |
|------------|---|----------|-------------------|----------------|
| **Delta-hedging flow prediction** | 2-3 | (spot_price, net_gamma_at_price, vol_regime) | Where dealers MUST trade as spot moves | Price magnetism (pin) and acceleration zones |
| **Volatility surface regime** | 3-4 | (ATM_vol, skew, term_slope, butterfly) | Distinct vol surface shapes | Regime change detection for vol trading |
| **Cross-asset correlation regime** | 5-10 | (returns of sector ETFs or correlated names) | Market state clustering | Portfolio hedging regime detection |
| **Intraday volume curve anomaly** | 3 | (time_of_day, volume_rank, price_change) | Normal vs abnormal intraday volume | Detect unusual activity pre-news |
| **Opening auction analysis** | 3-4 | (imbalance, indicative_price, size, time_to_open) | Opening rotation patterns | Predict opening direction from pre-open |
| **End-of-day positioning** | 3 | (price, volume_rank, minutes_to_close) | How closing flow clusters | MOC (market-on-close) flow detection |
| **Correlation breakdown detection** | 5+ | (multi-asset return vector) | Normal correlation regime vs breakdown | Cross-asset hedge failure warning |
| **Factor crowding** | 5-10 | (momentum, value, size, quality, volatility factor exposures) | Where stocks cluster in factor space | Crowded factors prone to unwind |
| **Pairs trade regime** | 3 | (spread_z, half-life_estimate, recent_vol) | Distinct spread behavior regimes | Enter pairs only in mean-reverting regime |
| **Event study classification** | 4-5 | (price_change, vol_change, spread_change, order_imbalance, t_since_event) | How different events cluster in impact space | Type events for future reaction prediction |

---

## What GSJ Specifically Adds vs Silverman

| Context | Silverman says | GSJ says | Why it matters |
|---------|---------------|----------|----------------|
| Tight ranging market | "One big support zone" | "3 distinct levels at $X, $Y, $Z" | Precision for limit order placement |
| Multiple expiry gamma walls | "Gamma concentrated near ATM" | "Separate walls at weekly/monthly/quarterly" | Different hedging dynamics per expiry |
| Vol regime detection | "2 regimes: calm and volatile" | "4 regimes: calm, hedging, panic, complacency" | Earlier detection of regime transition |
| Order flow types | "Big trades vs small trades" | "HFT, institutional sweep, retail, market-maker" | Different signal interpretation per source |
| Options positioning | "Puts at low strikes, calls at high" | "Specific clusters: protective puts at -5%, speculative calls at +3%, LEAPS at +10%" | Actionable levels vs vague direction |

---

## Key Insight: When Does GSJ Help Most?

GSJ helps most when the data is **genuinely multimodal in the feature space** — i.e., there are distinct clusters/modes that Silverman would merge. In trading contexts, this happens when:

1. **Multiple player types** are active simultaneously (institutional vs retail vs HFT)
2. **Multiple positioning strategies** coexist (protection vs speculation vs hedging)
3. **Multiple time horizons** overlap (weekly gamma vs monthly puts vs quarterly LEAPS)
4. **Distinct market regimes** exist within the observation period (range vs trend vs breakout)
5. **Multiple price levels** are simultaneously relevant (S/R levels, gamma pins, max pain)

For unimodal data (e.g., a steady trending market with consistent order flow), Silverman is fine because there's only one mode. GSJ's advantage appears precisely when the market has structural complexity worth resolving.

---

## Data Sources Summary

| Source | What it provides | Cost | Best for |
|--------|-----------------|------|----------|
| yfinance | OHLCV bars + options chains | Free | Volume profile, options OI/gamma surface |
| Polygon.io | Tick trades + quotes + options | Free tier (delayed) | Trade classification, quote analysis |
| Alpaca | Real-time trades + quotes | Free (paper account) | Live order flow proxy |
| Binance/crypto | Full trade history + order book snapshots | Free | Book imbalance, liquidity regime |
| CBOE DataShop | Historical options trades | $$ | Options flow research |
| LOBSTER (academic) | NASDAQ L3 order book reconstruction | Free (academic) | Full microstructure research |
| Bookmap / Jigsaw | Real-time DOM + reconstructed tape | $50-200/mo | Visual DOM analysis, iceberg detection |
| NYSE/ARCA ITCH feeds | Raw exchange message feed | $$$ | HFT research, spoofing detection |
