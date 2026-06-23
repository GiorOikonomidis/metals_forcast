"""
Prints the column headers and first row of every parquet file across all
three dataset cases.
"""
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PARENT = Path(__file__).resolve().parent.parent
CASES  = ["case_1_agg_news", "case_2_mask", "case_3_discard"]
FILES  = ["target.parquet", "feature_covariates.parquet", "dynamic_covariates.parquet"]

for case in CASES:
    print(f"\n{'='*70}")
    print(f"  {case}")
    print(f"{'='*70}")
    for fname in FILES:
        path = PARENT / "datasets" / case / fname
        df   = pd.read_parquet(path)
        print(f"\n  -- {fname}  ({len(df)} rows x {len(df.columns)} cols) --")
        print(f"  COLUMNS : {list(df.columns)}")
        print(f"  FIRST ROW:")
        row = df.iloc[0]
        for col, val in row.items():
            if hasattr(val, "__len__") and not isinstance(val, str):
                display = f"[...] len={len(val)}"
            else:
                display = repr(val)
            print(f"    {col:<20} {display}")
