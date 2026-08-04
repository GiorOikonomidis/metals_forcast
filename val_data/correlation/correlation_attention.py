"""
Cross-attention correlation diagnostic between the global covariate panel and
the target metals (XAU, XAG, XCU). Reads target_variables.parquet and
global_covariates.parquet from the repo root, differences both, and runs
scaled dot-product attention over the standardized diffed series treated as
per-feature token vectors (one token per column, embedding dim = number of
aligned dates). Two matrices come out: GLOBAL*GLOBAL (attention among all
global covariate columns) and GLOBAL*TARGET (attention from each global
column onto XAU/XAG/XCU). Results (CSV + heatmap PNG for each matrix) are
written to a results subfolder next to this script.

Run from anywhere:
    python <path-to-this-file>/correlation_attention.py
"""
import sys
from pathlib import Path

REPO_ROOT = Path(r"C:\Users\GiorgosOikonomidis\Desktop\proakt\imple_ours")
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from sklearn.decomposition import PCA

from model_impl.utils.data_loader_utils.transforms import apply_differencing

TARGET_PATH = REPO_ROOT / "produced_data" / "metals" / "datasets" / "target_variables.parquet"
GLOBAL_PATH = REPO_ROOT / "produced_data" / "metals" / "datasets" / "global_covariates.parquet"
OUT_DIR = Path(__file__).parent / "results_full"
TARGET_IDS = ["XCU"]
DIFF_MODE = "log_diff"
# `embedding` is a 768-d vector per day; Pearson correlation is only defined
# between scalar series, so it cannot be a column in the diffed GLOBAL panel
# like the rest. Its top principal components are scalar series and stand in
# as the "is this news signal helpful" measure — see load_embedding_pca.
EMBEDDING_PCA_COMPONENTS = 10
# Softmax temperature applied to the correlation scores before normalizing
# into attention weights. Scores are true Pearson correlations in [-1, 1],
# so a small temperature is needed for softmax to spread mass across keys
# instead of collapsing onto the single highest-correlation key.
ATTENTION_TEMPERATURE = 0.1
# Cells with value > HIGHLIGHT_THRESHOLD get a bold border. Cells with
# value < -HIGHLIGHT_THRESHOLD get one too, but only for matrices that
# actually contain negative values (e.g. the correlation matrices) — the
# softmax attention matrices are non-negative by construction, so a negative
# threshold there would never match anything and isn't drawn.
HIGHLIGHT_THRESHOLD = 0.4
# Separate, lower threshold for the GLOBAL*TARGET matrices: target relevance
# runs weaker than the internal GLOBAL*GLOBAL sector correlations (miners
# correlate ~0.6-0.9 with each other but only ~0.3-0.55 with XAU/XAG/XCU), so
# a single 0.4 cutoff hid real target signal.
TARGET_HIGHLIGHT_THRESHOLD = 0.25
# Above this |r| in the GLOBAL*GLOBAL matrix, two candidate features are
# treated as near-duplicates (e.g. same ticker's high/low/close, or two
# tickers moving as one sector factor) — only the one with stronger target
# relevance is kept.
REDUNDANCY_THRESHOLD = 0.85


def load_global(path: Path) -> pd.DataFrame:
    """
    Load the wide global-covariate panel indexed by date.

    Parameters
    ----------
    path : Path
        Location of global_covariates.parquet (wide format, one row per date,
        one column per feature).

    Returns
    -------
    pd.DataFrame
        All feature columns (opens, highs, lows, closes, oil, fx — every
        column except `date`), indexed by date and sorted ascending.
    """
    df = pd.read_parquet(path)
    df = df.set_index("date").sort_index()
    df.index = pd.to_datetime(df.index)
    # `embedding` (768-d vector per row) and `label` (string) are not
    # diffable/scalar columns — apply_differencing's astype(float64) fails on
    # them. `embedding`'s own correlation signal is measured separately via
    # its PCA projection (see load_embedding_pca); `label` is redundant with
    # the already-scalar prob_positive/negative/neutral columns kept here.
    return df.drop(columns=["embedding", "label"], errors="ignore")


def load_embedding_pca(path: Path, n_components: int) -> pd.DataFrame:
    """
    Reduce the daily news embedding to its top principal components.

    Pearson correlation is only defined between two scalar series, and each
    `embedding` cell is a 768-d vector, so there is no single "correlation
    with the embedding column" the way there is for a price column. PCA
    projects the daily embeddings onto their `n_components` directions of
    largest variance; each resulting component IS a scalar time series and
    can be correlated against the target exactly like any other covariate.
    This measures how much of the summarized news content moves with price —
    a real proxy for the embedding's helpfulness, not a substitute for
    feeding the model the full 768-d vector.

    Parameters
    ----------
    path : Path
        Location of global_covariates.parquet.
    n_components : int
        Number of principal components to keep.

    Returns
    -------
    pd.DataFrame
        (T, n_components) columns "embedding_pc1".."embedding_pcN", indexed
        by date, sorted ascending. Never differenced: the embedding is daily
        content, not a price level, so there is no level/diff distinction to
        preserve.
    """
    df = pd.read_parquet(path)[["date", "embedding"]].set_index("date").sort_index()
    df.index = pd.to_datetime(df.index)
    matrix = np.stack(df["embedding"].to_numpy())  # (T, 768)
    pcs = PCA(n_components=n_components, random_state=0).fit_transform(matrix)
    cols = [f"embedding_pc{i + 1}" for i in range(n_components)]
    return pd.DataFrame(pcs, index=df.index, columns=cols)


def load_target_metals(path: Path, ids: list[str]) -> pd.DataFrame:
    """
    Load close prices for the requested target ids, pivoted wide by date.

    Parameters
    ----------
    path : Path
        Location of target_variables.parquet (long format: date, id, open,
        high, low, close).
    ids : list[str]
        Target ids to keep, e.g. ["XAU", "XAG", "XCU"].

    Returns
    -------
    pd.DataFrame
        One column per id (the close price), indexed by date and sorted
        ascending.
    """
    df = pd.read_parquet(path)
    df = df[df["id"].isin(ids)]
    wide = df.pivot(index="date", columns="id", values="close")
    wide.index = pd.to_datetime(wide.index)
    return wide.sort_index()[ids]


def align_and_diff(global_df: pd.DataFrame, target_df: pd.DataFrame,
                    mode: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Restrict both frames to their common date range and difference each.

    Parameters
    ----------
    global_df : pd.DataFrame
        Wide global covariate panel, date-indexed.
    target_df : pd.DataFrame
        Wide target-metal panel (close prices), date-indexed.
    mode : str
        Differencing mode passed through to apply_differencing
        ("no_diff", "diff" or "log_diff").

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (global_diff, target_diff), both restricted to the shared date range,
        forward/back-filled for any remaining gaps before differencing so the
        two frames stay row-aligned.
    """
    common_index = global_df.index.intersection(target_df.index)
    global_df = global_df.loc[common_index].ffill().bfill()
    target_df = target_df.loc[common_index].ffill().bfill()
    global_diff = apply_differencing(global_df, mode)
    target_diff = apply_differencing(target_df, mode)
    return global_diff, target_diff


def zscore(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize every column to zero mean, unit variance.

    Parameters
    ----------
    df : pd.DataFrame
        Frame to standardize, columns treated independently.

    Returns
    -------
    pd.DataFrame
        Same shape, each column (x - mean) / std (std floored at 1e-8 to
        avoid divide-by-zero on constant columns).
    """
    std = df.std().replace(0, 1e-8)
    return (df - df.mean()) / std


def correlation_scores(query_tokens: torch.Tensor, key_tokens: torch.Tensor) -> torch.Tensor:
    """
    Pearson correlation between every query/key feature pair.

    Each row of `query_tokens`/`key_tokens` is one feature's standardized,
    differenced time series (zero mean, unit variance), so dividing the raw
    dot product by the embedding dimension (the number of aligned dates)
    gives the exact Pearson correlation coefficient — not the usual
    1/sqrt(d_model) transformer scaling, which only controls dot-product
    variance for independent unit-variance components and badly under-scales
    here (two identical standardized vectors dot to ~d_model, not
    ~sqrt(d_model)).

    Parameters
    ----------
    query_tokens : torch.Tensor
        (n_queries, d_model) — one row per query feature.
    key_tokens : torch.Tensor
        (n_keys, d_model) — one row per key feature.

    Returns
    -------
    torch.Tensor
        (n_queries, n_keys) correlation matrix, values in [-1, 1].
    """
    d_model = query_tokens.shape[-1]
    return query_tokens @ key_tokens.T / d_model


def cross_attention(scores: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    Turn correlation scores into an attention distribution over keys.

    Parameters
    ----------
    scores : torch.Tensor
        (n_queries, n_keys) correlation matrix, values in [-1, 1].
    temperature : float
        Divides the scores before softmax. Correlations already live in a
        narrow [-1, 1] range, so softmax needs a temperature well below 1 to
        spread weight across keys instead of collapsing onto whichever key
        has the single highest correlation (self, on the diagonal).

    Returns
    -------
    torch.Tensor
        (n_queries, n_keys) attention weight matrix, each row summing to 1.
    """
    return torch.softmax(scores / temperature, dim=-1)


def save_matrix(matrix: np.ndarray, row_labels: list[str], col_labels: list[str],
                 name: str, out_dir: Path, cmap: str, vmin: float, vmax: float,
                 highlight_threshold: float = HIGHLIGHT_THRESHOLD,
                 exclude_diagonal: bool = False) -> None:
    """
    Write a matrix to CSV and render it as a heatmap PNG with crisp cell borders.

    Cells with value > highlight_threshold get a bold black border. Cells
    with value < -highlight_threshold get one too, but only if the matrix
    actually contains negative values — matrices that are non-negative by
    construction (e.g. softmax attention weights) never trigger the negative
    side, so no threshold line is drawn there.

    Parameters
    ----------
    matrix : np.ndarray
        (len(row_labels), len(col_labels)) values to plot.
    row_labels : list[str]
        Names for the matrix rows (queries).
    col_labels : list[str]
        Names for the matrix columns (keys).
    name : str
        Base filename (without extension) for the CSV and PNG outputs.
    out_dir : Path
        Directory the outputs are written to (created if missing).
    cmap : str
        Matplotlib colormap name. Use a diverging map (e.g. "coolwarm") for
        signed data centered at 0, or a sequential map (e.g. "magma") for
        non-negative data like attention weights.
    vmin, vmax : float
        Fixed color-scale bounds, so every plot in a run is comparable and a
        single outlier cell can't wash out the rest of the palette.
    highlight_threshold : float
        Absolute-value cutoff for the bold-border highlight.
    exclude_diagonal : bool
        When True (a square self-correlation matrix), skip highlighting the
        diagonal — a feature is always correlated 1.0 with itself, which
        would otherwise trigger the threshold trivially on every cell.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(matrix, index=row_labels, columns=col_labels).to_csv(out_dir / f"{name}.csv")

    n_rows, n_cols = matrix.shape
    fig_w = max(6, 0.35 * n_cols)
    fig_h = max(5, 0.35 * n_rows)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(matrix, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)

    # Sharp white gridlines between every cell: minor ticks sit on cell
    # boundaries (offset by half a cell) while major ticks carry the labels.
    ax.set_xticks(np.arange(n_cols) - 0.5, minor=True)
    ax.set_yticks(np.arange(n_rows) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)

    # High-contrast marking: a lime border (reads against both the dark-red/
    # blue cmap extremes and the white gridlines) plus a black-outlined white
    # dot in the cell center, so a highlighted cell is unambiguous even when
    # scanning a large grid quickly.
    has_negatives = matrix.min() < 0
    above = matrix > highlight_threshold
    below = (matrix < -highlight_threshold) if has_negatives else np.zeros_like(matrix, dtype=bool)
    mask = above | below
    if exclude_diagonal:
        np.fill_diagonal(mask, False)
    for r, c in zip(*np.where(mask)):
        ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False,
                                edgecolor="#39FF14", linewidth=3.0, zorder=3))
        ax.plot(c, r, marker="o", markersize=4, markerfacecolor="white",
                markeredgecolor="black", markeredgewidth=0.8, zorder=4)

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(col_labels, rotation=90, fontsize=7)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(row_labels, fontsize=7)
    ax.set_title(name)
    fig.colorbar(im, ax=ax, fraction=0.03)
    fig.tight_layout()
    fig.savefig(out_dir / f"{name}.png", dpi=150)
    plt.close(fig)


def stage1_target_survivors(global_target_corr: pd.DataFrame, target_threshold: float) -> pd.DataFrame:
    """
    Stage 1: keep every global covariate relevant to at least one target.

    A feature survives if its correlation with ANY single target column
    exceeds `target_threshold` in absolute value — it does not need to clear
    the threshold against all three targets, one is enough. This is the same
    rule already driving the GLOBAL*TARGET highlight boxes (`max` over the
    row, not `min`).

    Parameters
    ----------
    global_target_corr : pd.DataFrame
        (n_global, n_target) Pearson correlation, global covariates x target ids.
    target_threshold : float
        Minimum |r| against any single target column to survive.

    Returns
    -------
    pd.DataFrame
        One row per surviving feature: the per-target correlations plus
        `max_abs_target_corr`, sorted by that column descending.
    """
    max_abs = global_target_corr.abs().max(axis=1)
    survivors = max_abs[max_abs > target_threshold].sort_values(ascending=False)
    result = global_target_corr.loc[survivors.index].copy()
    result["max_abs_target_corr"] = survivors
    return result


def stage2_redundant_pairs(survivor_corr: pd.DataFrame, redundancy_threshold: float) -> list[tuple[str, str, float]]:
    """
    Stage 2: report which stage-1 survivors are near-duplicates of each other.

    Scans the upper triangle of the survivors-only GLOBAL*GLOBAL submatrix
    (excluding the diagonal) and flags every pair whose |r| exceeds
    `redundancy_threshold` — these are the survivors that made it past target
    relevance but are still redundant with one another, left for a human
    decision rather than auto-dropped.

    Parameters
    ----------
    survivor_corr : pd.DataFrame
        (n_survivors, n_survivors) GLOBAL*GLOBAL correlation submatrix,
        restricted to stage-1 survivors on both axes.
    redundancy_threshold : float
        |r| above which a pair is flagged as redundant.

    Returns
    -------
    list[tuple[str, str, float]]
        (feature_a, feature_b, r) for every flagged pair, most correlated first.
    """
    cols = survivor_corr.columns
    pairs = [
        (cols[i], cols[j], float(survivor_corr.iloc[i, j]))
        for i in range(len(cols)) for j in range(i + 1, len(cols))
        if abs(survivor_corr.iloc[i, j]) > redundancy_threshold
    ]
    return sorted(pairs, key=lambda p: abs(p[2]), reverse=True)


def main() -> None:
    """
    Build and save the GLOBAL*GLOBAL and GLOBAL*TARGET correlation/attention matrices.

    Loads the raw parquet panels, aligns and differences them, standardizes
    every column, then computes Pearson correlation between global covariates
    (GLOBAL*GLOBAL) and between global covariates and the XAU/XAG/XCU
    close-price series (GLOBAL*TARGET). Each pair is written twice: the raw
    correlation matrix (interpretable directly, values in [-1, 1]) and a
    temperature-scaled softmax attention matrix derived from it. All four
    are written to results/ as CSV + heatmap PNG.
    """
    global_df = load_global(GLOBAL_PATH)
    target_df = load_target_metals(TARGET_PATH, TARGET_IDS)

    global_diff, target_diff = align_and_diff(global_df, target_df, DIFF_MODE)
    print(f"Aligned range: {global_diff.index[0].date()} -> {global_diff.index[-1].date()}  "
          f"(T={len(global_diff)})")

    # Embedding PCA correlation: restricted to target_diff's date range (never
    # differenced itself — see load_embedding_pca), reported separately from
    # the GLOBAL*TARGET matrix since PCA components aren't literal
    # GLOBAL_COVARIATES entries you can paste into a config.
    embedding_pca_df = load_embedding_pca(GLOBAL_PATH, EMBEDDING_PCA_COMPONENTS)
    embedding_common = embedding_pca_df.index.intersection(target_diff.index)
    embedding_pca_aligned = embedding_pca_df.loc[embedding_common].ffill().bfill()
    target_diff_for_embedding = target_diff.loc[embedding_common]

    global_z = zscore(global_diff)
    target_z = zscore(target_diff)

    global_tokens = torch.tensor(global_z.values.T, dtype=torch.float32)  # (G, T)
    target_tokens = torch.tensor(target_z.values.T, dtype=torch.float32)  # (K, T)

    global_labels = list(global_z.columns)
    target_labels = list(target_z.columns)

    corr_frames: dict[str, pd.DataFrame] = {}
    for corr_name, attn_name, q_tokens, k_tokens, row_labels, col_labels, threshold, is_self in [
        ("global_global_correlation", "global_global_attention",
         global_tokens, global_tokens, global_labels, global_labels, HIGHLIGHT_THRESHOLD, True),
        ("global_target_correlation", "global_target_attention",
         global_tokens, target_tokens, global_labels, target_labels, TARGET_HIGHLIGHT_THRESHOLD, False),
    ]:
        scores = correlation_scores(q_tokens, k_tokens)
        attn = cross_attention(scores, ATTENTION_TEMPERATURE)
        corr_frames[corr_name] = pd.DataFrame(scores.numpy(), index=row_labels, columns=col_labels)

        save_matrix(scores.numpy(), row_labels, col_labels, corr_name, OUT_DIR,
                    cmap="coolwarm", vmin=-1.0, vmax=1.0, highlight_threshold=threshold,
                    exclude_diagonal=is_self)
        save_matrix(attn.numpy(), row_labels, col_labels, attn_name, OUT_DIR,
                    cmap="magma", vmin=0.0, vmax=float(attn.max()), highlight_threshold=threshold,
                    exclude_diagonal=is_self)

        print(f"{corr_name}: {tuple(scores.shape)}  saved to {OUT_DIR / (corr_name + '.csv')}")
        print(f"{attn_name}: {tuple(attn.shape)}  saved to {OUT_DIR / (attn_name + '.csv')}")

    # Embedding helpfulness: correlate each PCA component (scalar) against
    # the target, same math as GLOBAL*TARGET, reported separately since these
    # rows aren't config-pastable GLOBAL_COVARIATES entries.
    embedding_z = zscore(embedding_pca_aligned)
    target_z_embedding = zscore(target_diff_for_embedding)
    embedding_tokens = torch.tensor(embedding_z.values.T, dtype=torch.float32)
    target_tokens_embedding = torch.tensor(target_z_embedding.values.T, dtype=torch.float32)
    embedding_labels = list(embedding_z.columns)

    embedding_scores = correlation_scores(embedding_tokens, target_tokens_embedding)
    embedding_corr = pd.DataFrame(embedding_scores.numpy(), index=embedding_labels, columns=target_labels)
    save_matrix(embedding_scores.numpy(), embedding_labels, target_labels,
                "embedding_pca_target_correlation", OUT_DIR,
                cmap="coolwarm", vmin=-1.0, vmax=1.0, highlight_threshold=TARGET_HIGHLIGHT_THRESHOLD)
    print(f"\nembedding_pca_target_correlation: {tuple(embedding_scores.shape)}  "
          f"saved to {OUT_DIR / 'embedding_pca_target_correlation.csv'}")
    max_abs_embedding = embedding_corr.abs().max(axis=1).sort_values(ascending=False)
    print(embedding_corr.loc[max_abs_embedding.index].round(3))
    n_relevant = (max_abs_embedding > TARGET_HIGHLIGHT_THRESHOLD).sum()
    print(f"{n_relevant}/{len(embedding_labels)} embedding components exceed "
          f"|r| > {TARGET_HIGHLIGHT_THRESHOLD} against any target — "
          f"{'embedding carries usable signal' if n_relevant else 'no PCA component clears the threshold'}")

    # Stage 1: which globals are relevant to at least one target.
    print(f"\n[stage 1] target survivors (any |r| > {TARGET_HIGHLIGHT_THRESHOLD}):")
    survivors = stage1_target_survivors(corr_frames["global_target_correlation"], TARGET_HIGHLIGHT_THRESHOLD)
    survivors.to_csv(OUT_DIR / "stage1_target_survivors.csv")
    print(survivors.round(3))
    print(f"{len(survivors)} features survived stage 1, saved to {OUT_DIR / 'stage1_target_survivors.csv'}")

    # Stage 2: among stage-1 survivors only, which are near-duplicates of each other.
    survivor_names = survivors.index.tolist()
    survivor_corr = corr_frames["global_global_correlation"].loc[survivor_names, survivor_names]
    save_matrix(survivor_corr.values, survivor_names, survivor_names, "stage2_survivor_redundancy", OUT_DIR,
                cmap="coolwarm", vmin=-1.0, vmax=1.0, highlight_threshold=REDUNDANCY_THRESHOLD,
                exclude_diagonal=True)

    print(f"\n[stage 2] redundant pairs among stage-1 survivors (|r| > {REDUNDANCY_THRESHOLD}):")
    redundant_pairs = stage2_redundant_pairs(survivor_corr, REDUNDANCY_THRESHOLD)
    if redundant_pairs:
        for a, b, r in redundant_pairs:
            print(f"  {a}  <->  {b}   r={r:+.3f}")
    else:
        print("  none")
    print(f"matrix saved to {OUT_DIR / 'stage2_survivor_redundancy.csv'} "
          f"(highlighted cells mark the pairs above)")


if __name__ == "__main__":
    main()
