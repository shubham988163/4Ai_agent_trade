# Pre-Registered Research Plan v2: Systematic Strategy Viability

---

## 1. Problem Statement and What Has Been Ruled Out

Empirical research across 49 Nifty constituents over 100 days of 5-minute data (69 trading sessions) demonstrated **no gross statistical edge** for the ORB+200MA intraday strategy. Gross expectancy across all stop variants was non-positive (-0.02R to -0.29R in both train and test splits) and statistically indistinguishable from a random coin-flip baseline. Furthermore, at actual retail sizing (Rs 15,000 capital, Rs 3,000 position cap, Rs 75 risk per trade), fixed statutory taxes and turnover fees impose round-trip friction of ~0.208% of position value, requiring a structural stop distance of at least 0.69% of price to satisfy the 30% cost-risk threshold. Because typical 5m 1.5xATR stops on Nifty 50 large-caps are only 0.14% to 0.29%, exactly 0 out of 49 stocks passed the cost gate. Trailing stops showed no measurable effect on the EMA-crossover strategy (paired 95% CI straddled zero, proving exits cannot manufacture an edge where entry alpha is absent); trailing was never tested on ORB. Consequently, 5-minute opening range breakout scalping with tight structural stops has shown no viable gross edge.

---

## 2. Candidate Hypotheses

### Hypothesis A: Higher-Timeframe Intraday Trend Continuation (15m / 60m Bars)
- **Economic Rationale:** Institutional execution in large-cap equities unfolds gradually across the session, creating persistent intraday momentum on higher-timeframe consolidation breakouts while filtering out 5-minute opening noise.
- **Timeframe & Assumed Stop Scale:** 15-minute or 60-minute candles; typical structural stop distance is assumed to be **0.75% to 1.30% of price** (*unverified assumption; subject to Step Zero*).
- **Cost Gate Feasibility:** At an assumed 0.85% stop distance and Rs 3,000 position cap, round-trip friction (~Rs 6.24) is 24.5% of trade risk (Rs 25.50), clearing the 30% cost ceiling.
- **Trade-Offs:** Retains zero overnight gap risk and lower intraday STT; however, signal frequency on the 6 watchlist stocks is assumed to be extremely low (~10 trades over 69 sessions; *unverified assumption; subject to Step Zero*), risking sample starvation.

### Hypothesis B: Daily / Multi-Day Swing Pullback (Cash Delivery / CNC)
- **Economic Rationale:** Multi-day institutional rebalancing drives multi-session price drift that exceeds intra-day noise thresholds.
- **Timeframe & Typical Stop Scale:** Daily (EOD) bars; typical structural stop distance is **3.0% to 5.0% of price**.
- **Cost Gate Feasibility (Risk-Based Sizing at Rs 75 Risk):**
  - Target risk = Rs 75. Position size $V = \text{Rs } 75 / s_{\%}$.
  - Delivery round-trip friction = Turnover charges ($0.3223\% \times V$) + Flat CDSL DP charge (**Rs 15.93**).
  - Flat DP charges alone consume $\text{Rs } 15.93 / \text{Rs } 75.00 = \mathbf{21.24\%}$ of allowable risk.
  - Turnover friction can consume at most $30.0\% - 21.24\% = 8.76\%$ of risk, requiring:
    $$s_{\%} \ge \frac{0.3223\%}{8.76\%} \approx \mathbf{3.68\%}$$
  - **Verdict:** Passes cost gate *only* if stop distance is $\ge 3.68\%$ of price (e.g., at 4.0% stop, position is Rs 1,875, friction is Rs 21.97 = 29.3% of risk). Any stop narrower than 3.68% fails at Rs 75 risk due to the flat DP fee.
- **Trade-Offs:** Captures larger price moves without intraday screen time; but introduces overnight gap risk, locks capital over multiple days, and requires wide stops ($\ge 3.68\%$).

### Hypothesis C: Low-Frequency Volatility-Gated 5m Breakout (Wide Stops Only)
- **Status:** **REJECTED based on funnel empirical results.**
- In the 69-session funnel across the 6 watchlist symbols, 0 out of 429 raw breakout candidates satisfied both indicator filters and a $\ge 0.70\%$ structural stop; all 9 surviving signals had stops $< 0.70\%$. Narrow structural stops are an intrinsic property of 5m bars in liquid Indian large caps.

### Hypothesis D: Explicit "Do Nothing / Keep Paper-Only" (Control Baseline)
- **Economic Rationale:** Capital preservation strictly dominates active trading when systematic strategies show no demonstrable gross alpha after transaction friction.
- **Timeframe & Stop Scale:** N/A (zero market exposure).
- **Cost Gate Feasibility:** Optimal (0.0% friction, 0 drawdown).
- **Trade-Offs:** Preserves 100% of capital and avoids development fatigue; generates zero active return and mothballs execution infrastructure.

---

## 3. Cost Model per Hypothesis

> [!WARNING]
> All statutory rates below must be verified against Zerodha's current charge page prior to live deployment. Rates reflect standard schedules and change periodically.

| Cost Component | Intraday MIS (Hypothesis A) | Delivery CNC (Hypothesis B) | Verified in `trading.costs`? |
| :--- | :---: | :---: | :---: |
| **Brokerage** | 0.03% capped at Rs 20/order | Rs 0.00 (Zero brokerage) | Yes (MIS modeled) |
| **STT / CTT** | 0.025% on sell leg only | 0.10% buy + 0.10% sell (**4x MIS on sell leg**) | **No** (CNC unmodeled) |
| **Exchange Txn Fee** | 0.00297% on turnover (both legs) | 0.00297% on turnover (both legs) | Yes |
| **SEBI Turnover Fee** | 0.0001% on turnover (both legs) | 0.0001% on turnover (both legs) | Yes |
| **Stamp Duty** | 0.003% on buy leg only | 0.015% on buy leg only | **No** (CNC stamp higher) |
| **GST** | 18% on (Brokerage + Exch + SEBI) | 18% on (Brokerage + Exch + SEBI) | Yes |
| **DP Charges (CDSL)** | Rs 0.00 | **Rs 13.50 + 18% GST = Rs 15.93 flat per sell** | **No** (Unmodeled) |
| **Simulated Slippage** | 0.05% per leg (0.10% round trip) | 0.05% per leg (0.10% round trip) | Configured (`SLIPPAGE_PCT`) |

### Unmodeled Costs in `trading.costs`:
1. **Flat DP Charges:** Flat Rs 15.93 debit charge on every delivery sale (eats 21.24% of Rs 75 risk).
2. **Delivery STT:** 0.10% on buy AND 0.10% on sell (4x intraday sell STT).
3. **Overnight Gap Slippage:** Morning auction gaps skipping stop prices.
4. **MTF Interest:** Margin financing interest (~18% p.a.) if holding delivery on leverage.

---

## 4. Data, Statistical Power, and Sample Requirements

### Statistical Power Table
Assuming standard deviation of trade returns $\sigma_R = 1.0R$:

$$N = \left( \frac{z_{1 - \alpha/2} + z_{1 - \beta}}{\delta / \sigma_R} \right)^2$$

| Target Effect ($\delta$) | Power ($1-\beta$) | Required $N$ at 95% ($\alpha=0.05, z=1.960$) | Required $N$ at 97.5% ($\alpha=0.025, z=2.241$) |
| :---: | :---: | :---: | :---: |
| **+0.20R** | 50% ($z=0.000$) | 96 trades | 126 trades |
| **+0.20R** | 80% ($z=0.842$) | 196 trades | 238 trades |
| **+0.15R** | 50% ($z=0.000$) | 171 trades | 223 trades |
| **+0.15R** | 80% ($z=0.842$) | 349 trades | 423 trades |

### Minimum Detectable Effect (MDE) at $N = 100$
$$\text{MDE} = \frac{z_{1 - \alpha/2} + z_{1 - \beta}}{\sqrt{N}} \times \sigma_R$$
- At 95% Confidence ($\alpha = 0.05$): $\text{MDE}_{50\%} = \mathbf{+0.196R}$, $\text{MDE}_{80\%} = \mathbf{+0.280R}$.
- At 97.5% Confidence ($\alpha = 0.025$): $\text{MDE}_{50\%} = \mathbf{+0.224R}$, $\text{MDE}_{80\%} = \mathbf{+0.308R}$.
*Conclusion:* At $N = 100$, an experiment with 97.5% confidence can only detect large edges ($\ge +0.22R$ to $+0.31R$). Any true edge below $+0.22R$ will fail to achieve significance.

### Data Feasibility and Window Requirements
- **Watchlist Limitation:** The 6 watchlist stocks over 69 sessions cannot yield $N \ge 100$ higher-timeframe trades.
- **For Hypothesis A (Intraday 15m/60m):**
  - *Option 1 (Forward Test Window):* 30 forward market sessions (target window: `2026-09-29` to `2026-11-10`) across the 49 Nifty stocks.
  - *Option 2 (Extended History):* Use Yahoo Finance 60m data (~730 days of 1-hour candles) to secure multi-year sample depth.
- **For Hypothesis B (Daily Swing):** Requires 2 to 3 years of daily history (e.g., `2021-01-01` to `2024-06-30` for exploration) across 49 stocks to reach $N \ge 100$ closed swing trades.
- **Survivorship Bias Warning:** Testing historical windows on the *current* Nifty 50 constituent list introduces survivorship bias (poor performers demoted from the index are excluded; past winners are over-represented).

---

## 5. Pre-Registered Protocol

### Step-Zero Pre-Check (Mandatory Gate)
Before computing returns, P&L, or expectancy:
1. Scan the candidate universe on historical data.
2. Measure candidate trade count ($N$) and empirical stop distance distribution ($s_{\%}$).
3. **Hard Stop Rule:** If total closed trades $N < 100$, or if fewer than 50% of setups have stop distances meeting the cost gate ($s_{\%} \ge 0.69\%$ for MIS, $s_{\%} \ge 3.68\%$ for CNC), **terminate the research immediately**. Do not proceed to return calculations.

### Primary Metric
$$\bar{R}_{\text{gross}} = \frac{1}{N} \sum_{i=1}^N \frac{\text{Gross P\&L}_i}{\text{Initial Risk}_i}$$

### Matched Baselines
- **For Hypothesis A:** Matched random direction baseline on identical entry bars, identical hold duration, 500 bootstrap iterations ($N \ge 100$).
- **For Hypothesis B (Swing):** Matched-exposure baseline: same trade direction (long), random entry dates, identical hold duration ($D$ days) to isolate timing alpha from passive equity market drift / beta.

### Numerical Pass / Fail Criteria (Uniform 97.5% Confidence)
A hypothesis passes if and only if all three criteria are met at **97.5% two-sided confidence** (Bonferroni-adjusted for 2 candidate variants):
1. **Gross Alpha:** Lower bound of 97.5% CI must be positive:
   $$\text{CI}_{97.5\%,\text{low}}(\bar{R}_{\text{gross}}) > 0.00R$$
2. **Baseline Superiority:** Statistically superior to matched baseline:
   $$p < 0.025$$
3. **Net Cost Viability:** Lower bound of 97.5% Net CI must be positive:
   $$\text{CI}_{97.5\%,\text{low}}(\bar{R}_{\text{net}}) > 0.00R$$

### Day-Level Clustered Confidence Intervals
Confidence intervals must be generated via block-bootstrapping clustered by trading date ($d \in D$) to account for cross-sectional market correlations on trend days.

### Strict Data Partitioning
- **Quarantined Window:** `2026-08-28` to `2026-09-28` (the 21 held-out sessions from ORB research are permanently locked).
- **Hypothesis A:** Exploration on 48 sessions (`2026-06-22` to `2026-08-27`); out-of-sample testing on 30 forward sessions (`2026-09-29` to `2026-11-10`) or uninspected 60m yfinance history.
- **Hypothesis B:** Exploration on `2021-01-01` to `2024-06-30`; test split on `2024-07-01` to `2026-06-21` (evaluated strictly once).

### Minimum Economic Threshold
$$\text{Net Return} \ge +0.20R \text{ per trade} \implies \ge \text{Rs } 15.00 \text{ net profit per trade}$$
At Rs 75 risk, any strategy delivering $< +0.10R$ (< Rs 7.50/trade) is economically meaningless relative to API costs, compute infrastructure, and operational risk.

### Stop Rule (Project Termination)
Failure of the exploration set to achieve $\text{CI}_{97.5\%,\text{low}}(\bar{R}_{\text{gross}}) > 0.00R$ permanently ends work on that hypothesis.

---

## 6. Look-Ahead and Realism Checklist

### Required Modifications to `Backtester` (`trading/backtest.py`):
1. **Next-Open Entry Fill:** Modify `_try_enter` to fill at `Open[i+1]` after signal confirmation on `Close[i]` (currently enters at `bar["Close"]`).
2. **Gap Fill Handling on Stops:** If a candle opens beyond the stop-loss level (gap down/up), record exit at `Open[i]`, not the nominal stop price.
3. **Dual-Leg Slippage:** Enforce 0.05% slippage on both entry and exit legs.
4. **Exact Cost Model:** Wire `trading.costs.round_trip` for actual held quantity.
5. **Loss Limit Enforcement:** Halt backtest trading for the session if intraday realized + unrealized loss hits `DAILY_LOSS_LIMIT` (-Rs 500).

---

## 7. Decisions Needed from the User

Before proceeding with any implementation or data extraction, specify:

1. **Hypothesis Selection:** Which candidate approach do you want to explore?
   - `Hypothesis A`: 15m/60m intraday trend continuation (stops 0.75%-1.30%, MIS schedule).
   - `Hypothesis B`: Daily multi-day swing pullback (stops 2.50%-5.00%, CNC schedule + DP charges).
   - `Hypothesis D`: Do nothing / remain paper-only and preserve capital.
2. **Universe Scope:** Do you approve expanding testing to the **49 Nifty constituents**, acknowledging that the 6-stock watchlist cannot deliver the statistical minimum of $N \ge 100$ trades?
3. **Multi-Year Daily Data Acquisition:** If Hypothesis B (Daily swing) is selected, do you approve downloading 2021–2026 daily data via yfinance / Fyers?
4. **Cost Model Extension:** If Hypothesis B (CNC) is selected, should `trading.costs` be extended to include flat DP charges (Rs 15.93) and delivery STT schedules?
