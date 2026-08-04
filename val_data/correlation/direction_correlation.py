"""
Correlation between the target's pure price direction (up / down / flat) and
the day's news signal, for every dataset under produced_data/*/datasets/.

The news signal has two parts, measured two different ways:
- prob_positive / prob_negative / prob_neutral: plain scalars, correlated
  against direction with ordinary Pearson correlation.
- embedding: a 768-d vector per day, for which Pearson correlation against a
  scalar is undefined column-by-column (see correlation_attention.py's
  load_embedding_pca for the same issue). Reduced to its top principal
  components first, each of which IS a scalar series and can be correlated.
- label (positive/negative/neutral): categorical, correlated against the
  categorical direction (up/down/flat) via Cramer's V, plus a same-day
  agreement rate (label=="positive" & direction==+1, etc.).

Run from anywhere:
    python <path-to-this-file>/direction_correlation.py
"""
import sys
from pathlib import Path

REPO_ROOT = Path(r"C:\Users\GiorgosOikonomidis\Desktop\proakt\imple_ours")
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
from sklearn.decomposition import PCA

OUT_DIR = Path(__file__).parent / "results_direction"
EMBEDDING_PCA_COMPONENTS = 10
# Datasets to check: (name, target_id, target_parquet, global_parquet)
# `name` is the plot/file label — it must be the actual target id, not the
# dataset directory ("metals" is misleading since XCU is the only metal
# target built here; there is no XAU/XAG target anywhere in this repo).
DATASETS = [
    ("XCU", "XCU", REPO_ROOT / "produced_data" / "metals" / "datasets" / "target_variables.parquet",
     REPO_ROOT / "produced_data" / "metals" / "datasets" / "global_covariates.parquet"),
]


def load_direction(target_path: Path, target_id: str) -> pd.Series:
    """
    Pure day-over-day price direction for one target id: +1 up, -1 down, 0 flat.

    Parameters
    ----------
    target_path : Path
        Location of target_variables.parquet (long format: date, id, close, ...).
    target_id : str
        Which id to pull, e.g. "XCU" or "^nsdq".

    Returns
    -------
    pd.Series
        sign(close.diff()), indexed by date, first row (no prior value) dropped.
    """
    df = pd.read_parquet(target_path)
    df = df[df["id"] == target_id].copy()
    df["date"] = pd.to_datetime(df["date"])
    close = df.sort_values("date").set_index("date")["close"]
    return np.sign(close.diff()).dropna()


def load_news(global_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Split the global covariate panel's news columns into their three forms.

    Parameters
    ----------
    global_path : Path
        Location of global_covariates.parquet.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.Series]
        scalars : (T, 3) prob_positive/negative/neutral, date-indexed.
        embedding : (T, 768) raw embedding, date-indexed.
        label : (T,) categorical sentiment label, date-indexed.
    """
    df = pd.read_parquet(global_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    scalars = df[["prob_positive", "prob_negative", "prob_neutral"]]
    embedding = pd.DataFrame(
        np.stack(df["embedding"].to_numpy()), index=df.index,
        columns=[f"emb_{i}" for i in range(768)],
    )
    label = df["label"]
    return scalars, embedding, label


def cramers_v(direction: pd.Series, label: pd.Series) -> float:
    """
    Association strength between two categorical variables (0 = none, 1 = perfect).

    Parameters
    ----------
    direction : pd.Series
        Categorical, e.g. {-1, 0, 1}.
    label : pd.Series
        Categorical, e.g. {"positive", "negative", "neutral"}.

    Returns
    -------
    float
        Cramer's V computed from the chi-square statistic of their contingency
        table, bias-uncorrected (fine at these sample sizes).
    """
    table = pd.crosstab(direction, label)
    chi2 = chi2_contingency(table)[0]
    n = table.values.sum()
    r, k = table.shape
    return float(np.sqrt((chi2 / n) / min(r - 1, k - 1)))


def plot_correlation_bar(scalar_corr: pd.Series, embedding_corr: pd.Series,
                          name: str, out_dir: Path) -> None:
    """
    Bar chart of every prob_* and embedding-PCA correlation against direction.

    Parameters
    ----------
    scalar_corr : pd.Series
        Pearson r of prob_positive/negative/neutral against direction.
    embedding_corr : pd.Series
        Pearson r of each embedding PCA component against direction.
    name : str
        Dataset label, used for the title and filename.
    out_dir : Path
        Directory the PNG is written to.
    """
    combined = pd.concat([scalar_corr, embedding_corr]).sort_values()
    colors = ["#c53030" if v < 0 else "#2b6cb0" for v in combined.values]

    fig, ax = plt.subplots(figsize=(7, max(4, 0.35 * len(combined))))
    ax.barh(combined.index, combined.values, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Pearson r vs. direction")
    ax.set_title(f"{name}: news signal correlation with price direction")
    fig.tight_layout()
    fig.savefig(out_dir / f"{name}_direction_correlation_bar.png", dpi=150)
    plt.close(fig)


def plot_label_direction_heatmap(agreement: pd.DataFrame, v: float,
                                  name: str, out_dir: Path) -> None:
    """
    Heatmap of P(label | direction) — does the news label distribution shift
    depending on which way the target actually moved that day.

    Parameters
    ----------
    agreement : pd.DataFrame
        Row-normalized crosstab, direction (index) x label (columns).
    v : float
        Cramer's V for the pair, shown in the title.
    name : str
        Dataset label, used for the title and filename.
    out_dir : Path
        Directory the PNG is written to.
    """
    fig, ax = plt.subplots(figsize=(5, 1.2 + 0.8 * len(agreement)))
    im = ax.imshow(agreement.values, cmap="magma", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(agreement.columns)))
    ax.set_xticklabels(agreement.columns)
    ax.set_yticks(range(len(agreement.index)))
    ax.set_yticklabels([{-1.0: "down", 0.0: "flat", 1.0: "up"}.get(i, i) for i in agreement.index])
    for r in range(agreement.shape[0]):
        for c in range(agreement.shape[1]):
            val = agreement.values[r, c]
            ax.text(c, r, f"{val:.2f}", ha="center", va="center",
                    color="white" if val < 0.6 else "black", fontsize=9)
    ax.set_title(f"{name}: P(label | direction)   Cramer's V={v:.3f}")
    fig.colorbar(im, ax=ax, fraction=0.05)
    fig.tight_layout()
    fig.savefig(out_dir / f"{name}_label_direction_heatmap.png", dpi=150)
    plt.close(fig)


def analyze(name: str, target_id: str, target_path: Path, global_path: Path) -> None:
    """
    Run and print/save the direction-vs-news correlation for one dataset.

    Parameters
    ----------
    name : str
        Dataset label for output filenames ("metals", "index").
    target_id : str
        Target id within target_variables.parquet.
    target_path, global_path : Path
        Dataset parquet locations.
    """
    direction = load_direction(target_path, target_id)
    scalars, embedding, label = load_news(global_path)

    common = direction.index.intersection(scalars.index)
    direction = direction.loc[common]
    scalars = scalars.loc[common]
    embedding = embedding.loc[common]
    label = label.loc[common]

    print(f"\n{'=' * 60}\n{name} ({target_id})  N={len(common)}  "
          f"{common.min().date()} -> {common.max().date()}\n{'=' * 60}")
    print("direction counts:", direction.value_counts().to_dict())

    # --- scalars: plain Pearson ---
    scalar_corr = scalars.apply(lambda col: direction.corr(col))
    print("\n[prob_* Pearson correlation with direction]")
    print(scalar_corr.round(4))

    # --- embedding: PCA then Pearson, same fix as correlation_attention.py ---
    pcs = PCA(n_components=EMBEDDING_PCA_COMPONENTS, random_state=0).fit_transform(embedding.values)
    pcs_df = pd.DataFrame(pcs, index=embedding.index,
                          columns=[f"embedding_pc{i + 1}" for i in range(EMBEDDING_PCA_COMPONENTS)])
    embedding_corr = pcs_df.apply(lambda col: direction.corr(col)).sort_values(key=abs, ascending=False)
    print("\n[embedding PCA components Pearson correlation with direction]")
    print(embedding_corr.round(4))

    # --- label: categorical association (Cramer's V, all 3 classes) ---
    v = cramers_v(direction, label)
    print(f"\n[label categorical association]\nCramer's V(direction, label) = {v:.4f}")
    agreement = pd.crosstab(direction, label, normalize="index")
    print("P(label | direction):")
    print(agreement.round(3))

    # --- label as {-1, 0, 1}: plain Pearson between two discrete distributions ---
    # Both sides are now literally in {-1, 0, 1}, so this is the same Pearson
    # correlation used for prob_*/embedding above, just with label encoded to
    # match direction's sign convention instead of routed through chi-square.
    label_numeric = label.map({"negative": -1, "neutral": 0, "positive": 1})
    label_pearson_r = direction.corr(label_numeric)
    print(f"\n[label as {{-1,0,1}}] Pearson r(direction, label_numeric) = {label_pearson_r:.4f}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    scalar_corr.to_frame("pearson_r").to_csv(OUT_DIR / f"{name}_prob_correlation.csv")
    embedding_corr.to_frame("pearson_r").to_csv(OUT_DIR / f"{name}_embedding_pca_correlation.csv")
    agreement.to_csv(OUT_DIR / f"{name}_label_direction_crosstab.csv")
    plot_correlation_bar(scalar_corr, embedding_corr, name, OUT_DIR)
    plot_label_direction_heatmap(agreement, v, name, OUT_DIR)
    with open(OUT_DIR / f"{name}_summary.txt", "w") as f:
        f.write(f"N={len(common)}  cramers_v={v:.4f}  label_pearson_r={label_pearson_r:.4f}\n")
        f.write(f"direction counts: {direction.value_counts().to_dict()}\n")
        f.write(f"\nprob_* correlation:\n{scalar_corr}\n")
        f.write(f"\nembedding PCA correlation:\n{embedding_corr}\n")


def main() -> None:
    """Run the direction-vs-news analysis for every configured dataset."""
    for name, target_id, target_path, global_path in DATASETS:
        if not target_path.exists() or not global_path.exists():
            print(f"skip {name}: missing {target_path if not target_path.exists() else global_path}")
            continue
        analyze(name, target_id, target_path, global_path)


if __name__ == "__main__":
    main()
