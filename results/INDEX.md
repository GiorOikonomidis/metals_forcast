# Results index

Reorganized from the original dump (`XCU/`, `res_with_news/`, `synthetic/` with
inconsistent, sometimes-misleading folder names) into three groups. Folder
names now encode: **target — head — diff mode — ctx/pred — epochs — covariates
— news** directly, so you don't need to open `config_snapshot.json` to know
what a run was.

- `real_XCU/` — 11 runs against the real XCU price series (2 are the same
  config re-run a day apart, kept as A/B for reproducibility comparison).
- `synthetic/` — 6 runs against synthetic targets (SINE, SINC, SQUARE,
  SINE_MOD, AR1_RETURNS ×2).
- `incomplete_runs/` — 3 runs that never finished (killed mid-training or
  crashed) — no `summary.json`, kept for the `run.log`/`config_snapshot.json`
  post-mortem only, excluded from the tables below.
- `sample_forecasts/` — five representative per-window forecast plots, chosen to show
  the contrast this project is about: XCU collapsing to persistence, versus
  SINE/SINE_MOD/AR1_RETURNS where the model genuinely tracks the signal.

**On pruned artifacts**: each run originally emitted one forecast plot per test window
(~360 PNGs per run, ~614 MB total). Those per-day dumps are excluded from the repo — every
*aggregate* diagnostic is retained (coverage summary, CRPS/MSE/WIS per window, PIT
histogram, reliability curve, skill-vs-naive, loss curves, per-horizon breakdowns),
alongside each run's `summary.json`, `config_snapshot.json` and `run.log` — so every
number in the tables below is verifiable from what's committed here.

**Naming caveat found during reorg**: the original folder `no_feat_covs__with_news`
(now `D_...`) actually had per-id XCU technical-indicator covariates (11
features) configured *in addition to* the 49 cross-asset global covariates —
i.e. it wasn't a "no feature covariates" ablation at all, the old name was
simply wrong. Renamed to reflect what it actually ran.

---

## Real XCU — ranked by MSE skill vs. naive (higher is better; naive = 0%)

| id | folder | head | diff | ctx→pred | epochs | covariates | news | windows | MSE | skill vs naive | cov50/80/90 | sharp80 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| J | `J_transformer_logdiff_ctx365_pred50_ep200_cross12_news` | transformer | log_diff | 365→50 | 200 | 12 cross-asset | on | 150 | 0.1969 | **+1.79%** | .07/.14/.17 | 0.107 |
| (rerun) | `J_..._RERUN_20260728` | — same config, run a day earlier — | | | | | | 150 | 0.1969 | +1.79% | .07/.14/.17 | 0.107 |
| I | `I_transformerMini_logdiff_ctx100_ep300_cross49_news` | transformer (mini) | log_diff | 100→7 | 300 | 49 cross-asset | on | 359 | 0.0360 | −0.49% | .08/.14/.17 | 0.044 |
| C | `C_transformer_logdiff_ctx30_ep100_xcuFeat11_cal6_noNews` | transformer | log_diff | 30→7 | 100 | XCU-own×11 + 6 calendar | off | 359 | 0.0359 | −0.19% | .11/.19/.24 | 0.056 |
| E | `E_linear_logdiff_ctx130_ep200_cross12_news` | linear | log_diff | 130→7 | 200 | 12 cross-asset | on | 150 | 0.0512 | −1.50% | .80/.96/.97 | 0.703 |
| G | `G_linear_logdiff_ctx100_ep200_cross49_news_do15` | linear | log_diff | 100→7 | 200 | 49 cross-asset | on | 359 | 0.0377 | −5.28% | .89/.99/.99 | 1.114 |
| H | `H_linear_logdiff_ctx100_ep200_cross49_news_do15_temp1.5` | linear | log_diff | 100→7 | 200 | 49 cross-asset | on | 150 | 0.0527 | −4.31% | .96/.97/.97 | 1.248 |
| A | `A_linear_nodiff_ctx100_ep300_xcuFeat11_cal6_noNews` | linear | no_diff | 100→7 | 300 | XCU-own×11 + 6 calendar | off | 359 | 0.0472 | −31.6% | .09/.18/.23 | 0.076 |
| F | `F_linear_nodiff_ctx130_ep200_cross12_news` | linear | no_diff | 130→7 | 200 | 12 cross-asset | on | 150 | 0.0580 | −14.9% | .69/.97/1.0 | **3.009** |
| B | `B_linear_nodiff_ctx100_ep100_xcuFeat11_cal6_noNews` | linear | no_diff | 100→7 | 100 | XCU-own×11 + 6 calendar | off | 359 | 0.0816 | −127.8% | .06/.14/.18 | 0.104 |
| D | `D_linear_nodiff_ctx100_ep300_xcuFeat11_cross49_news` | linear | no_diff | 100→7 | 300 | XCU-own×11 **+** 49 cross-asset | on | 359 | **0.1497** | **−317.6%** | .07/.10/.12 | 0.103 |

## Synthetic — sanity checks

| id | folder | target | ctx→pred | epochs | MSE | skill vs naive | interpretation |
|---|---|---|---|---|---|---|---|
| N | `N_SQUARE_transformer_logdiff_ctx30_pred10_ep30` | SQUARE | 30→10 | 30 | 6.8e-14 | +99.99999997% | trivial signal, essentially solved |
| L | `L_SINE_transformer_logdiff_ctx30_pred10_ep30` | SINE | 30→10 | 30 | 5.4e-9 | +99.99994% | clean periodic signal, essentially solved |
| M | `M_SINC_transformer_logdiff_ctx30_pred10_ep60` | SINC | 30→10 | 60 | 2.5e-9 | +99.99995% | clean periodic signal, essentially solved |
| Q | `Q_AR1_RETURNS_transformer_logdiff_ctx30_pred7_ep100_cal6` | AR1_RETURNS (φ=0.3) | 30→7 | 100 | 1.174e-4 | +1.18% | genuine causal return structure — model beats naive, matches theory |
| P | `P_AR1_RETURNS_transformer_logdiff_ctx30_pred7_ep100_noCovs` | AR1_RETURNS (φ=0.3) | 30→7 | 100 | 1.181e-4 | +0.59% | same signal, no covariates — beats naive by less than Q |
| O | `O_SINE_MOD_transformer_logdiff_ctx30_pred7_ep100` | SINE_MOD (trend+AM+FM) | 30→7 | 100 | 0.2422 | **−18.7%** | worse than naive — surprising given SINE_MOD is deterministic; worth re-checking (see observations) |

## Incomplete (excluded from the tables above — no summary.json)

| id | folder | status |
|---|---|---|
| R | `R_transformer_nodiff_ctx100_ep300_xcuFeat11_cal6_KILLED_epoch7` | killed/stopped at epoch 7 of 300 |
| S | `S_XAU_lstm_logdiff_ctx130_CRASHED_after_split` | died right after the temporal split, before epoch 0 — no traceback captured (5-line log) |
| T | `T_linear_nodiff_ctx100_ep200_cross49_news_KILLED_epoch27` | killed mid-training around epoch 27; `val_ce` had started climbing again (ep22-27: 3.846→3.913) right before it stopped — early stopper (patience 10) likely would have fired within ~5-10 more epochs anyway |

---

## Observations — capabilities and incapabilities

| # | Observation | Evidence |
|---|---|---|
| 1 | **The model can learn deterministic structure when it's actually there.** SINE/SINC/SQUARE all converge to near-zero error (skill ≈99.9999%). This rules out a basic architecture/training-loop bug as the cause of poor XCU performance. | N, L, M |
| 2 | **The model correctly detects genuine causal return structure and only there.** AR1_RETURNS (built with real φ=0.3 momentum) is the *only* real-return-generating-process signal that beats naive (+0.6 to +1.2%). Every real-XCU config, whose returns measured ≈0 autocorrelation at all lags, fails to beat naive. This is consistent, not contradictory — it's exactly what efficient-market statistics predict. | P, Q vs. A/B/D/E/F/G/H |
| 3 | **On real XCU, `log_diff` + `transformer`/small-ctx setups get closest to naive parity; `no_diff` is uniformly worse.** The two best real-XCU results (I: −0.49%, C: −0.19%) are both `log_diff`; the worst (D: −317.6%, B: −127.8%) are both `no_diff`. | I, C vs. B, D |
| 4 | **Calibration flips between badly overconfident and badly underconfident depending on `news`/`diff` combination — no config found a middle ground.** `no_diff`+no-news runs (A, B, D) are drastically overconfident: `cov50` ≈0.06–0.11 against a nominal 0.50 (the 50% interval only actually contains truth 6–11% of the time). News-enabled runs (E, G, H) are the opposite — `cov50` ≈0.80–0.96, badly underconfident/oversized. | cov50 column across all rows |
| 5 | **Interval width can blow up in `no_diff`+news mode.** Run F's `sharp80` (0.2-tail interval width) is 3.0 — roughly the full price range of XCU itself — and `wis90` is 40.6, an order of magnitude above every other run. This looks like MC-Dropout variance exploding specifically in the `no_diff` + `news`-enabled combination. | F |
| 6 | **The one config that beat naive on real XCU used a 50-day horizon, not 7.** Run J (`pred_len=50`, `ctx_len=365`) is the only real-XCU run with positive skill (+1.79%), and it's reproducible (identical result across the two dated reruns). Every 7-day-horizon real-XCU run underperforms naive. This may indicate longer-horizon log-diff cumulative structure is more learnable than 1-step direction — worth a deliberate follow-up sweep across horizons rather than treating this as a one-off. | J vs. all `pred_len=7` real rows |
| 7 | **`D`'s covariate count doesn't add up — worth debugging before trusting that run.** `config_snapshot.json` for `D` shows 11 per-id (XCU-own) + 49 global covariates configured (60 total), but both the output directory name and `summary.json`'s own `"covariates"` field say `49`. Either the per-id covariates silently weren't applied, or the count reported in `summary.json`/output-dir naming under-counts by omitting `n_long` — either way this is worth resolving, since `D` is currently your worst real-data result and you can't yet tell whether that's "feature covariates hurt" or "feature covariates were silently dropped and this is actually the cross-49-only result." | D's `config_snapshot.json` vs. `summary.json`/folder name |
| 8 | **`SINE_MOD` underperforming naive is inconsistent with its own visual forecast plot shown earlier and worth re-running.** A deterministic, fully periodic signal getting *worse* skill than persistence (−18.7%) is not what runs N/L/M would predict. Possible causes: only 100 epochs vs. the harder FM+AM+trend combination needing more; or an eval-time MC-sampling/temperature mismatch specific to this run. | O |
| 9 | **3 of 20 launched runs never finished** (R, S, T) — one crashed within seconds of the temporal split with no captured traceback (S), one was killed 27/200 epochs in with `val_ce` starting to climb again (T), one was killed early during a 300-epoch run (R). None of these look like the same failure mode, so each needs its own log inspection rather than a single fix. | incomplete_runs/ |
