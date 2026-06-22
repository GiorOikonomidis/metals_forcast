from typing import Literal
import pandas as pd
from news_creation import *

# parquet files
# from the enrchied files we will get 
"""
Target Value file :
- id
- date
- Close

Dynamic covariates :
- id 
- date
- Al features in sep coluns

Feature covariates :
- date
- date features

There is missaligment problems :
- all covariates dont start from the same date 
- new are every day insead of the other covs
"""
JoinType    = Literal["inner", "left", "right", "outer"]
FillStrategy = Literal["keep_nan", "drop_nan", "ffill_nan"]


def align(
    left: pd.DataFrame,
    right: pd.DataFrame,
    how: JoinType = "inner",
    fill: FillStrategy = "keep_nan",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Aligns two DataFrames on their Date index.

    Join types:
        inner  — keep only dates present in both
        left   — keep all dates from left, NaN where right is missing
        right  — keep all dates from right, NaN where left is missing
        outer  — keep all dates from either, NaN where either is missing

    Fill strategies (applied after join):
        keep_nan  — leave NaN values as-is
        drop_nan  — drop any rows where either side has NaN
        ffill_nan — forward-fill NaN values in right from last known value

    Args:
        left:  DataFrame indexed by Date (e.g. price features)
        right: DataFrame indexed by Date (e.g. news features)
        how:   join type controlling which dates to keep
        fill:  how to handle NaN values introduced by the join
    Returns:
        (left_aligned, right_aligned): both indexed by the same set of dates
    """
    if how == "inner":
        idx = left.index.intersection(right.index)
        left, right = left.loc[idx], right.loc[idx]
    elif how == "left":
        right = right.reindex(left.index)
    elif how == "right":
        left = left.reindex(right.index)
    elif how == "outer":
        idx = left.index.union(right.index)
        left  = left.reindex(idx)
        right = right.reindex(idx)
    else:
        raise ValueError(f"Unknown how: {how!r}. Choose from: inner, left, right, outer")

    if fill == "keep_nan":
        pass
    elif fill == "ffill_nan":
        right = right.ffill().bfill()
    elif fill == "drop_nan":
        combined = pd.concat([left, right], axis=1).dropna()
        left  = combined[left.columns]
        right = combined[right.columns]
    else:
        raise ValueError(f"Unknown fill: {fill!r}. Choose from: keep_nan, drop_nan, ffill_nan")

    return left, right


# den kserw pws ta fetcharoume opote apo presaved
if __name__ == "__main__":
    date_start = "2007-01-03"
    date_end = None