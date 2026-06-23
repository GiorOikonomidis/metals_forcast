"""
Dataset validation — checks all three cases for issues that would break or
silently degrade a model: NaN/inf, shape errors, range violations, alignment
mismatches, constant columns, duplicate dates, and cross-case consistency.

Run from imple_ours/:  venv/Scripts/python dataset_validation.py
"""
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

CASES = ["case_mask", "case_interp"]
INDEX_FEATURES = [
    "EMA_12", "EMA_26", "MACD", "RSI", "Stoch_K", "Stoch_D",
    "Williams_R", "ROC", "Daily_Return", "Volatility", "Movement",
]
SENT_COLS = ["prob_positive", "prob_negative", "prob_neutral"]

EMBED_DIM = 768

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"
WARN = "\033[33mWARN\033[0m"
INFO = "\033[36mINFO\033[0m"

issues = []   # (severity, case, message)

def check(ok, severity, case, msg, detail=""):
    tag = PASS if ok else (FAIL if severity == "FAIL" else WARN)
    line = f"  [{tag}] {msg}"
    if detail:
        line += f"\n         {detail}"
    print(line)
    if not ok:
        issues.append((severity, case, msg + (" — " + detail if detail else "")))

def header(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


# ── load ──────────────────────────────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def load_case(case):
    base = os.path.join(SCRIPT_DIR, "..", "datasets", case)
    return {
        "tgt":  pd.read_parquet(os.path.join(base, "target.parquet")),
        "feat": pd.read_parquet(os.path.join(base, "feature_covariates.parquet")),
        "dyn":  pd.read_parquet(os.path.join(base, "dynamic_covariates.parquet")),
    }

print("Loading datasets...")
data = {}
for c in CASES:
    try:
        data[c] = load_case(c)
        print(f"  Loaded {c}")
    except Exception as e:
        print(f"  [FAIL] Could not load {c}: {e}")
        sys.exit(1)


# ══════════════════════════════════════════════════════════════════════════════
#  PER-CASE CHECKS
# ══════════════════════════════════════════════════════════════════════════════

for case in CASES:
    tgt  = data[case]["tgt"]
    feat = data[case]["feat"]
    dyn  = data[case]["dyn"]
    is_case2 = (case == "case_2_mask")

    dates_tgt  = pd.to_datetime(tgt["date"])
    dates_feat = pd.to_datetime(feat["date"])
    dates_dyn  = pd.to_datetime(dyn["date"])

    company_cols = [c for c in dyn.columns if "__" in c]
    close_cols   = [c for c in company_cols if c.endswith("__Close")]
    n = len(tgt)

    header(f"CASE: {case}  ({n} rows)")

    # ── 1. Shape sanity ───────────────────────────────────────────────────────
    check(len(feat) == n, "FAIL", case, "feat row count matches target",
          f"feat={len(feat)}, tgt={n}")
    check(len(dyn) == n, "FAIL", case, "dyn row count matches target",
          f"dyn={len(dyn)}, tgt={n}")

    # ── 2. Date alignment ─────────────────────────────────────────────────────
    dates_match_feat = np.array_equal(dates_tgt.values, dates_feat.values)
    dates_match_dyn  = np.array_equal(dates_tgt.values, dates_dyn.values)
    check(dates_match_feat, "FAIL", case, "target.date == feat.date (exact alignment)")
    check(dates_match_dyn,  "FAIL", case, "target.date == dyn.date  (exact alignment)")

    # ── 3. Date monotonicity & uniqueness ─────────────────────────────────────
    is_sorted = dates_tgt.is_monotonic_increasing
    check(is_sorted, "FAIL", case, "dates are monotonically increasing (no reordering)")

    n_dup = dates_tgt.duplicated().sum()
    check(n_dup == 0, "FAIL", case, "no duplicate dates",
          f"{n_dup} duplicate(s) found" if n_dup else "")

    # ── 4. Target Close ───────────────────────────────────────────────────────
    close_nan = tgt["Close"].isna().sum()
    if is_case2:
        trading_mask = feat["trading_day"] if "trading_day" in feat.columns else None
        if trading_mask is not None:
            trading_nan = tgt["Close"].isna() & (trading_mask == 1)
            check(trading_nan.sum() == 0, "FAIL", case,
                  "target Close not NaN on trading days",
                  f"{trading_nan.sum()} trading rows with NaN Close")
            non_trading_nan = tgt["Close"].isna() & (trading_mask == 0)
            print(f"  [{INFO}] target Close NaN on non-trading days: {non_trading_nan.sum()} (expected)")
    else:
        check(close_nan == 0, "FAIL", case, "target Close has no NaN",
              f"{close_nan} NaN values" if close_nan else "")

    inf_close = np.isinf(tgt["Close"].fillna(0)).sum()
    check(inf_close == 0, "FAIL", case, "target Close has no inf", f"{inf_close} inf" if inf_close else "")

    # ── 5. Feature covariates ─────────────────────────────────────────────────
    date_cols = ["sin_dow", "cos_dow", "sin_month", "cos_month", "sin_doy", "cos_doy"]
    for col in date_cols:
        if col in feat.columns:
            nan_c = feat[col].isna().sum()
            inf_c = np.isinf(feat[col]).sum()
            check(nan_c == 0, "FAIL", case, f"feat.{col} no NaN", f"{nan_c} NaN" if nan_c else "")
            check(inf_c == 0, "FAIL", case, f"feat.{col} no inf", f"{inf_c} inf" if inf_c else "")
            in_range = feat[col].between(-1.0, 1.0).all()
            check(in_range, "FAIL", case, f"feat.{col} in [-1, 1]",
                  f"min={feat[col].min():.4f} max={feat[col].max():.4f}" if not in_range else "")

    if is_case2:
        check("trading_day" in feat.columns, "FAIL", case, "feat has trading_day column (Case 2)")
        if "trading_day" in feat.columns:
            bad_vals = ~feat["trading_day"].isin([0, 1])
            check(bad_vals.sum() == 0, "FAIL", case, "trading_day only 0 or 1",
                  f"{bad_vals.sum()} bad values" if bad_vals.sum() else "")
            td_nan = feat["trading_day"].isna().sum()
            check(td_nan == 0, "FAIL", case, "trading_day no NaN", f"{td_nan} NaN" if td_nan else "")
            n_trading = int(feat["trading_day"].sum())
            print(f"  [{INFO}] trading_day: {n_trading} trading / {n - n_trading} non-trading rows")
    else:
        check("trading_day" not in feat.columns, "WARN", case,
              "feat has no trading_day column (Cases 1 & 3 only have trading days)")

    # ── 6. Index technical features ───────────────────────────────────────────
    present_idx_cols = [c for c in INDEX_FEATURES if c in dyn.columns]
    missing_idx_cols = [c for c in INDEX_FEATURES if c not in dyn.columns]
    check(len(missing_idx_cols) == 0, "FAIL", case, "all index features present in dyn",
          f"missing: {missing_idx_cols}" if missing_idx_cols else "")

    for col in present_idx_cols:
        nan_c = dyn[col].isna().sum()
        inf_c = np.isinf(dyn[col].replace([None], np.nan).astype(float)).sum()
        check(nan_c == 0, "FAIL", case, f"dyn.{col} no NaN", f"{nan_c} NaN ({nan_c/n*100:.1f}%)" if nan_c else "")
        check(inf_c == 0, "FAIL", case, f"dyn.{col} no inf", f"{inf_c} inf" if inf_c else "")

        # range checks for bounded indicators
        if col == "RSI":
            oob = ((dyn[col] < 0) | (dyn[col] > 100)).sum()
            check(oob == 0, "WARN", case, "RSI in [0, 100]",
                  f"{oob} values outside range" if oob else "")
        if col in ("Stoch_K", "Stoch_D"):
            oob = ((dyn[col] < 0) | (dyn[col] > 100)).sum()
            check(oob == 0, "WARN", case, f"{col} in [0, 100]",
                  f"{oob} values outside range" if oob else "")
        if col == "Williams_R":
            oob = ((dyn[col] < -100) | (dyn[col] > 0)).sum()
            check(oob == 0, "WARN", case, "Williams_R in [-100, 0]",
                  f"{oob} values outside range" if oob else "")
        if col == "Movement":
            oob = ((dyn[col] < -1.0) | (dyn[col] > 1.0)).sum()
            check(oob == 0, "WARN", case, "Movement (daily return) in [-1, 1]",
                  f"{oob} values outside range — extreme daily moves" if oob else "")

        # constant / zero-variance
        std = dyn[col].std()
        check(std > 1e-8, "WARN", case, f"dyn.{col} not constant",
              f"std={std:.2e}" if std <= 1e-8 else "")

    # ── 7. Company features ───────────────────────────────────────────────────
    check(len(company_cols) > 0, "FAIL", case, "company feature columns present",
          f"found {len(company_cols)}")
    if company_cols:
        total_nan = dyn[company_cols].isna().sum().sum()
        check(total_nan == 0, "FAIL", case,
              "company features no NaN (ffill+bfill should eliminate all)",
              f"{total_nan} total NaN across {len(company_cols)} columns" if total_nan else "")
        total_inf = np.isinf(dyn[company_cols].values.astype(float)).sum()
        check(total_inf == 0, "FAIL", case, "company features no inf",
              f"{total_inf} inf values" if total_inf else "")

    if close_cols:
        neg_prices = (dyn[close_cols] <= 0).sum().sum()
        check(neg_prices == 0, "FAIL", case, "company __Close columns all positive (>0)",
              f"{neg_prices} zero/negative values" if neg_prices else "")

        # per-company NaN report if any
        if dyn[close_cols].isna().sum().sum():
            per_co = dyn[close_cols].isna().sum()
            bad_co = per_co[per_co > 0]
            print(f"  [{WARN}] Companies with NaN closes:")
            for co, cnt in bad_co.items():
                print(f"           {co}: {cnt} NaN")

    # ── 8. News / sentiment ───────────────────────────────────────────────────
    for col in SENT_COLS:
        if col not in dyn.columns:
            check(False, "FAIL", case, f"dyn has {col} column")
            continue
        nan_c  = dyn[col].isna().sum()
        inf_c  = np.isinf(dyn[col].fillna(0)).sum()
        neg_c  = (dyn[col].fillna(0) < 0).sum()
        print(f"  [{INFO}] {col}: {nan_c} NaN ({nan_c/n*100:.1f}%), {nan_c} rows without news")
        check(inf_c == 0, "FAIL", case, f"{col} no inf", f"{inf_c} inf" if inf_c else "")
        check(neg_c == 0, "FAIL", case, f"{col} no negative probability",
              f"{neg_c} negative values" if neg_c else "")

        if case != "case_1_agg_news":
            oob = (dyn[col].dropna() > 1.0).sum()
            check(oob == 0, "WARN", case, f"{col} <= 1.0 (single-day prob, Cases 2 & 3)",
                  f"{oob} values > 1.0" if oob else "")

    # label/prob consistency
    if "label" in dyn.columns:
        label_notnull = dyn["label"].notna()
        prob_notnull  = dyn["prob_positive"].notna()
        mismatch = (label_notnull != prob_notnull).sum()
        check(mismatch == 0, "FAIL", case,
              "label non-null iff prob_positive non-null (no half-filled news rows)",
              f"{mismatch} mismatches" if mismatch else "")

        valid_labels = {"positive", "negative", "neutral"}
        bad_labels = dyn["label"].dropna()[~dyn["label"].dropna().isin(valid_labels)]
        check(len(bad_labels) == 0, "FAIL", case, "label only positive/negative/neutral",
              f"{len(bad_labels)} invalid: {set(bad_labels)}" if len(bad_labels) else "")

    # ── 9. Embedding integrity ────────────────────────────────────────────────
    if "embedding" in dyn.columns:
        emb_series = dyn["embedding"].dropna()
        n_emb = len(emb_series)
        n_none_where_label = (dyn["label"].notna() & dyn["embedding"].isna()).sum()
        check(n_none_where_label == 0, "FAIL", case,
              "embedding non-null wherever label is set",
              f"{n_none_where_label} rows with label but no embedding" if n_none_where_label else "")

        if n_emb > 0:
            dims = emb_series.apply(lambda x: len(x) if hasattr(x, "__len__") else -1)
            wrong_dim = (dims != EMBED_DIM).sum()
            check(wrong_dim == 0, "FAIL", case,
                  f"all embeddings have dim={EMBED_DIM}",
                  f"{wrong_dim} with wrong dim: {dims[dims != EMBED_DIM].value_counts().to_dict()}" if wrong_dim else "")

            # check for NaN/inf inside embedding vectors
            sample_idx = emb_series.index[:min(200, n_emb)]
            bad_emb = 0
            for idx in sample_idx:
                v = dyn.loc[idx, "embedding"]
                arr = np.asarray(v, dtype=float)
                if np.isnan(arr).any() or np.isinf(arr).any():
                    bad_emb += 1
            check(bad_emb == 0, "FAIL", case,
                  f"no NaN/inf inside embedding vectors (sampled first {len(sample_idx)})",
                  f"{bad_emb} bad embeddings found" if bad_emb else "")

            # L2 norm sanity (FinBERT CLS token should be in a reasonable range)
            norms = emb_series.iloc[:min(500, n_emb)].apply(
                lambda x: float(np.linalg.norm(np.asarray(x, dtype=float)))
            )
            check(norms.min() > 0.01, "WARN", case, "embedding L2 norm > 0.01 (not near-zero)",
                  f"min norm={norms.min():.4f}" if norms.min() <= 0.01 else "")
            print(f"  [{INFO}] embedding L2 norm: min={norms.min():.3f} mean={norms.mean():.3f} max={norms.max():.3f}")
    else:
        check(False, "FAIL", case, "embedding column present in dyn")


# ══════════════════════════════════════════════════════════════════════════════
#  CROSS-CASE CONSISTENCY
# ══════════════════════════════════════════════════════════════════════════════

header("CROSS-CASE CONSISTENCY")

# Both cases share the same date range (built from identical all_dates logic)
dates_mask  = pd.DatetimeIndex(data["case_mask"]["tgt"]["date"])
dates_interp = pd.DatetimeIndex(data["case_interp"]["tgt"]["date"])
check(dates_mask.equals(dates_interp), "FAIL", "cross",
      "case_mask and case_interp have identical date ranges",
      f"mask={len(dates_mask)} vs interp={len(dates_interp)}")

# Target Close values should be identical (same index, same unfilled source)
tgt_mask  = data["case_mask"]["tgt"].set_index("date")
tgt_interp = data["case_interp"]["tgt"].set_index("date")
common = tgt_mask.index.intersection(tgt_interp.index)
if len(common):
    diff = (tgt_mask.loc[common, "Close"] - tgt_interp.loc[common, "Close"]).abs().max()
    check(diff < 1e-6, "FAIL", "cross",
          "target Close identical in both cases on all shared dates",
          f"max diff={diff:.2e}" if diff >= 1e-6 else "")

# Company feature columns should be identical across both cases
dyn_mask  = data["case_mask"]["dyn"]
dyn_interp = data["case_interp"]["dyn"]
company_cols_mask  = sorted([c for c in dyn_mask.columns  if "__" in c])
company_cols_interp = sorted([c for c in dyn_interp.columns if "__" in c])
check(company_cols_mask == company_cols_interp, "FAIL", "cross",
      "Same company columns in both cases",
      f"diff: {set(company_cols_mask).symmetric_difference(company_cols_interp)}"
      if company_cols_mask != company_cols_interp else "")

# case_mask should have trading_day column; case_interp should not
feat_mask  = data["case_mask"]["feat"]
feat_interp = data["case_interp"]["feat"]
check("trading_day" in feat_mask.columns,  "FAIL", "cross", "case_mask has trading_day column")
check("trading_day" not in feat_interp.columns, "FAIL", "cross", "case_interp has no trading_day column")

# On trading days, case_interp company closes should equal case_mask closes
# (interpolation only affects non-trading rows; trading rows have real prices)
if company_cols_mask == company_cols_interp and len(company_cols_mask):
    trading_rows = feat_mask["trading_day"].values == 1
    sample_col   = company_cols_mask[0]
    diff_td = (dyn_mask.loc[trading_rows, sample_col].values
               - dyn_interp.loc[trading_rows, sample_col].values)
    max_diff = np.abs(diff_td).max()
    check(max_diff < 1e-6, "FAIL", "cross",
          f"company closes identical on trading days (checked {sample_col})",
          f"max diff={max_diff:.2e}" if max_diff >= 1e-6 else "")


# ══════════════════════════════════════════════════════════════════════════════
#  POTENTIAL DATA-LEAKAGE CHECKS
# ══════════════════════════════════════════════════════════════════════════════

header("LEAKAGE CHECKS")

# Feature covariates must NOT contain price-derived columns
price_like = ["Close", "Open", "High", "Low", "Volume", "EMA", "MACD", "RSI",
              "Return", "Volatility", "Movement"]
for case in CASES:
    feat = data[case]["feat"]
    leaky = [c for c in feat.columns if any(p.lower() in c.lower() for p in price_like)
             and c != "trading_day"]
    check(len(leaky) == 0, "FAIL", case,
          "feature_covariates contains no price-derived columns",
          f"leaky columns: {leaky}" if leaky else "")

# Movement in dynamic_covariates is fine (it IS a covariate, derived from past data)
# But warn if it has 0 variance (e.g., all zeros)
for case in CASES:
    dyn = data[case]["dyn"]
    if "Movement" in dyn.columns:
        mv = dyn["Movement"].dropna()
        print(f"  [{INFO}] {case} Movement: min={mv.min():.4f}  mean={mv.mean():.4f}  max={mv.max():.4f}  std={mv.std():.4f}")


# ══════════════════════════════════════════════════════════════════════════════
#  SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

header("SUMMARY")

fails = [(c, m) for s, c, m in issues if s == "FAIL"]
warns = [(c, m) for s, c, m in issues if s == "WARN"]

if fails:
    print(f"\n  {len(fails)} FAILURE(S) — must fix before training:\n")
    for case, msg in fails:
        print(f"    [{case}]  {msg}")
else:
    print(f"\n  No failures.")

if warns:
    print(f"\n  {len(warns)} WARNING(S) — review but may be acceptable:\n")
    for case, msg in warns:
        print(f"    [{case}]  {msg}")
else:
    print(f"  No warnings.")

print()
