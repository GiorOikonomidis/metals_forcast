"""Output routing: where one run's artifacts live on disk."""

from datetime import datetime
from pathlib import Path

from model_impl.consts import OUTPUT_ROOT


def make_output_dir(n_covariates: int, no_news: bool, index: str, type_of_diff: str,
                    base_dir: str = "", create: bool = True) -> Path:
    """
    <base_dir or OUTPUT_ROOT>/<INDEX>/<TYPE_OF_DIFF>_c_{covariates}_w_{news|no_news}/YYYYMMDD

    `create=False` (TRACKING.LOCAL.use off) still returns the path so callers
    have a stable name (e.g. for the MLflow run name), but never touches disk.
    """
    news_flag = "no_news" if no_news else "news"
    date_str = datetime.now().strftime("%Y%m%d")
    root = Path(base_dir) if base_dir else Path(OUTPUT_ROOT)
    out = root / index / f"{type_of_diff}_c_{n_covariates}_w_{news_flag}" / date_str
    if create:
        out.mkdir(parents=True, exist_ok=True)
    return out
