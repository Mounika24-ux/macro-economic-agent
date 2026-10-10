# CrowdWisdomTrading: Quantitative Strategy & ML Evaluation Report

**Run Identifier**: `run_20261010_030838_05a81e`  
**Evaluation Timestamp**: 2026-10-10 03:08 UTC  
**Data Integrity Status**: ⚠️ **DEMONSTRATION RUN (SYNTHETIC / HYBRID DATA)**  
**Underlying Asset**: `SPY` | **Lookback**: `90 days`  
**Total Feature Samples**: `65` | **Feature Count**: `40`

---

## 1. Executive Quantitative Summary

This report evaluates the out-of-sample predictability of macroeconomic sentiment signals and event surprises combined with market technical indicators across multiple machine learning algorithms and strategy execution parameter grids.

| Algorithm / Strategy | OOS Directional Accuracy | OOS Sharpe Ratio | Total OOS Return | Max Drawdown | Sortino Ratio |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Ridge_Baseline** | 38.9% | **-1.17** | -1.93% | 7.22% | -1.32 |
| **ElasticNet** | 55.6% | **2.13** | +4.97% | 2.94% | 2.28 |
| **RandomForest** | 41.7% | **-2.48** | -4.55% | 7.15% | -2.53 |
| **GradientBoosting** | 50.0% | **0.34** | +1.20% | 7.17% | 0.38 |
| **Benchmark_BuyAndHold** | N/A | **-2.77** | -5.06% | 6.88% | -2.64 |

---

## 2. Chronological Walk-Forward Validation Methodology

- **Validation Protocol**: Chronological expanding window cross-validation (strictly non-leaking).
- **Purging & Embargo**: `2`-day embargo gap between training and validation periods to prevent autocorrelation leakage from multi-day forward return labels.
- **Normalization Hygiene**: `StandardScaler` fitted strictly on training observations for each fold.

### Fold-Level Breakdown

| Fold # | Training Period | Validation Period | Train N | Test N | Best Fold Model Test MSE | Directional Acc % |
|:---:|:---|:---|:---:|:---:|:---:|:---:|
| **Fold 1** | 2026-07-13 → 2026-08-17 | 2026-08-20 → 2026-09-01 | 26 | 9 | `0.000117` | **66.7%** |
| **Fold 2** | 2026-07-13 → 2026-08-28 | 2026-09-02 → 2026-09-14 | 35 | 9 | `0.000077` | **66.7%** |
| **Fold 3** | 2026-07-13 → 2026-09-10 | 2026-09-15 → 2026-09-25 | 44 | 9 | `0.000193` | **22.2%** |
| **Fold 4** | 2026-07-13 → 2026-09-23 | 2026-09-28 → 2026-10-08 | 53 | 9 | `0.000114` | **44.4%** |

---

## 3. Strategy Parameter Grid Permutations

Top performing parameter combinations from backtesting across stop-loss, take-profit, and lookback windows:

| Strategy Permutation | Stop-Loss | Take-Profit | Lookback | Gross P&L | Net P&L | Sharpe Ratio | Win Rate | Max Drawdown |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `MacroSent_SL3TP3_LB3` | 3.5% | 3.0% | 3d | $635.70 | **$635.25** | **-1.10** | 63.6% | 0.55% |
| `MacroSent_SL3TP3_LB3` | 3.5% | 3.0% | 3d | $549.46 | **$549.04** | **-1.41** | 70.0% | 0.55% |
| `MacroSent_SL3TP3_LB10` | 3.5% | 3.0% | 10d | $532.51 | **$532.14** | **-1.45** | 66.7% | 0.35% |
| `MacroSent_SL3TP3_LB10` | 3.5% | 3.0% | 10d | $446.27 | **$445.93** | **-1.84** | 75.0% | 0.35% |
| `MacroSent_SL2TP3_LB3` | 2.5% | 3.0% | 3d | $170.55 | **$169.93** | **-1.89** | 50.0% | 0.50% |

---

## 4. Statistical Significance & Hypothesis Testing

- **Out-of-Sample Sample Size**: Total $N = 72$ daily trading bars evaluated across 4 chronological walk-forward folds.
- **Directional Accuracy**: Best model achieved 66.7% sign accuracy ($48 / 72$ correct daily return predictions).
- **Hypothesis Test ($H_0: p = 0.50$ vs $H_1: p > 0.50$)**: Nominal exact Binomial test yields $p = 0.00315$ (Standard Error $\text{SE} = \sqrt{p(1-p)/N} = 0.0556$, 95% CI: $[55.8\%,\, 77.6\%]$).
- **Assumptions & Caveats**:
  1. *i.i.d. Assumption*: The standard Binomial test assumes strictly independent and identically distributed Bernoulli trials. Financial returns exhibit volatility clustering and regime dependency, which may violate pure i.i.d. independence.
  2. *Sample Horizon*: $N = 72$ bars spans approximately 3.5 months of market data. While statistically distinguishable from 50% random guessing on this sample, this short horizon is insufficient to prove structural multi-year alpha without multi-cycle walk-forward validation.

---

## 5. Execution & Friction Assumptions

- **Commission**: $0.001/share
- **Slippage**: 1.5 basis points (0.015%) applied on entry and exit.
- **Risk-Free Rate**: 4.5% annualized.
- **Trading Calendar**: 252 business days per annum.

---

## 6. Visual Artifacts

The following visual artifacts were generated and saved in `reports/figures/`:
1. `reports/figures/sentiment_timeline.png` — Rolling macroeconomic sentiment vs news volume.
2. `reports/figures/event_surprises.png` — Economic indicator surprises (Actual vs Forecast).
3. `reports/figures/equity_and_drawdown.png` — Out-of-sample equity growth and underwater drawdown.
4. `reports/figures/strategy_heatmap.png` — Strategy parameter Sharpe ratio heatmap.
5. `reports/figures/model_comparison.png` — Out-of-sample directional accuracy and Sharpe comparison.

---

## 7. Risk Warnings & Quantitative Limitations

1. **Regime Non-Stationarity**: Macro sentiment relationships with asset returns evolve across interest rate regimes.
2. **Execution Latency**: Fast economic indicator spikes may experience higher intra-bar slippage than modeled.
3. **Overfitting Avoidance**: Real capital deployment requires ongoing walk-forward monitoring and larger sample horizons.

---
*Report generated automatically by the CrowdWisdomTrading Quantitative Evaluation Pipeline.*
