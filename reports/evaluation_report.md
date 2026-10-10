# CrowdWisdomTrading: Quantitative Strategy & ML Evaluation Report

**Run Identifier**: `run_20261010_021714_a66a0c`  
**Evaluation Timestamp**: 2026-10-10 02:17 UTC  
**Data Integrity Status**: ⚠️ **DEMONSTRATION RUN (SYNTHETIC / HYBRID DATA)**  
**Underlying Asset**: `SPY` | **Lookback**: `180 days`  
**Total Feature Samples**: `130` | **Feature Count**: `40`

---

## 1. Executive Quantitative Summary

This report evaluates the out-of-sample predictability of macroeconomic sentiment signals and event surprises combined with market technical indicators across multiple machine learning algorithms and strategy execution parameter grids.

| Algorithm / Strategy | OOS Directional Accuracy | OOS Sharpe Ratio | Total OOS Return | Max Drawdown | Sortino Ratio |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Ridge_Baseline** | 47.2% | **-0.56** | -1.54% | 9.10% | -0.62 |
| **ElasticNet** | 48.6% | **-1.21** | -4.30% | 7.43% | -1.18 |
| **RandomForest** | 50.0% | **-3.61** | -13.61% | 16.35% | -2.98 |
| **GradientBoosting** | 56.9% | **-1.48** | -5.43% | 10.44% | -1.21 |
| **Benchmark_BuyAndHold** | N/A | **0.64** | +3.79% | 6.54% | 0.67 |

---

## 2. Chronological Walk-Forward Validation Methodology

- **Validation Protocol**: Chronological expanding window cross-validation (strictly non-leaking).
- **Purging & Embargo**: `2`-day embargo gap between training and validation periods to prevent autocorrelation leakage from multi-day forward return labels.
- **Normalization Hygiene**: `StandardScaler` fitted strictly on training observations for each fold.

### Fold-Level Breakdown

| Fold # | Training Period | Validation Period | Train N | Test N | Best Fold Model Test MSE | Directional Acc % |
|:---:|:---|:---|:---:|:---:|:---:|:---:|
| **Fold 1** | 2026-04-13 → 2026-06-26 | 2026-07-01 → 2026-07-24 | 55 | 18 | `0.000163` | **55.6%** |
| **Fold 2** | 2026-04-13 → 2026-07-22 | 2026-07-27 → 2026-08-19 | 73 | 18 | `0.000072` | **66.7%** |
| **Fold 3** | 2026-04-13 → 2026-08-17 | 2026-08-20 → 2026-09-14 | 91 | 18 | `0.000090` | **55.6%** |
| **Fold 4** | 2026-04-13 → 2026-09-10 | 2026-09-15 → 2026-10-08 | 109 | 18 | `0.000147` | **50.0%** |

---

## 3. Strategy Parameter Grid Permutations

Top performing parameter combinations from backtesting across stop-loss, take-profit, and lookback windows:

| Strategy Permutation | Stop-Loss | Take-Profit | Lookback | Gross P&L | Net P&L | Sharpe Ratio | Win Rate | Max Drawdown |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `MacroSent_SL3TP3_LB3` | 3.5% | 3.0% | 3d | $3,133.07 | **$3,132.02** | **0.92** | 64.0% | 0.74% |
| `MacroSent_SL3TP3_LB10` | 3.5% | 3.0% | 10d | $2,886.08 | **$2,885.11** | **0.67** | 65.2% | 0.74% |
| `MacroSent_SL3TP3_LB3` | 3.5% | 3.0% | 3d | $2,717.71 | **$2,716.71** | **0.48** | 62.5% | 0.74% |
| `MacroSent_SL3TP3_LB5` | 3.5% | 3.0% | 5d | $2,483.06 | **$2,482.01** | **0.22** | 60.0% | 0.74% |
| `MacroSent_SL3TP3_LB10` | 3.5% | 3.0% | 10d | $2,470.72 | **$2,469.80** | **0.22** | 63.6% | 0.74% |

---

## 4. Execution & Friction Assumptions

- **Commission**: $0.001/share
- **Slippage**: 1.5 basis points (0.015%) applied on entry and exit.
- **Risk-Free Rate**: 4.5% annualized.
- **Trading Calendar**: 252 business days per annum.

---

## 5. Visual Artifacts

The following visual artifacts were generated and saved in `reports/figures/`:
1. `reports/figures/sentiment_timeline.png` — Rolling macroeconomic sentiment vs news volume.
2. `reports/figures/event_surprises.png` — Economic indicator surprises (Actual vs Forecast).
3. `reports/figures/equity_and_drawdown.png` — Out-of-sample equity growth and underwater drawdown.
4. `reports/figures/strategy_heatmap.png` — Strategy parameter Sharpe ratio heatmap.
5. `reports/figures/model_comparison.png` — Out-of-sample directional accuracy and Sharpe comparison.

---

## 6. Risk Warnings & Quantitative Limitations

1. **Regime Non-Stationarity**: Macro sentiment relationships with asset returns evolve across interest rate regimes.
2. **Execution Latency**: Fast economic indicator spikes may experience higher intra-bar slippage than modeled.
3. **Overfitting Avoidance**: Real capital deployment requires ongoing walk-forward monitoring.

---
*Report generated automatically by the CrowdWisdomTrading Quantitative Evaluation Pipeline.*
