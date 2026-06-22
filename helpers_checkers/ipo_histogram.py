import os
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

data_dir = os.path.join("..","data_enriched", "companies")

ticker_year = {}
for f in os.listdir(data_dir):
    if f.endswith(".csv"):
        ticker = f.replace(".csv", "")
        df = pd.read_csv(os.path.join(data_dir, f), index_col="Date", nrows=1)
        year = pd.to_datetime(df.index[0]).year
        ticker_year[ticker] = year

years      = list(ticker_year.values())
counts     = Counter(years)
all_years  = list(range(min(years), max(years) + 1))
values     = [counts.get(y, 0) for y in all_years]

late_by_year = {}
for ticker, year in ticker_year.items():
    if year > 2007:
        late_by_year.setdefault(year, []).append(ticker)

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={"width_ratios": [2, 1]})

ax.bar(all_years, values, color="#185FA5", width=0.6)
ax.set_xlabel("Year")
ax.set_ylabel("Number of companies")
ax.set_title("Nasdaq-100 company start dates")
ax.set_xticks(all_years)
ax.set_xticklabels([str(y) for y in all_years], rotation=45, ha="right")
ax.yaxis.grid(True, alpha=0.3)
ax.set_axisbelow(True)

ax2.axis("off")
rows = []
for year in sorted(late_by_year):
    rows.append([str(year), ", ".join(sorted(late_by_year[year]))])

table = ax2.table(
    cellText=rows,
    colLabels=["Year", "Tickers"],
    cellLoc="left",
    loc="center",
)
table.auto_set_font_size(False)
table.set_fontsize(9)
table.auto_set_column_width([0, 1])
ax2.set_title("Companies starting after 2007", pad=12)

plt.tight_layout()
plt.savefig("ipo_histogram.png", dpi=150)
print("Saved → ipo_histogram.png")
plt.show()
