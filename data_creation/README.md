κκ# proakt — data creation

Builds the parquet datasets that `model_impl` trains on. It downloads price series
and news, enriches the prices with technical indicators and the news with
transformer sentiment/embeddings, and merges everything into three aligned files.

The package lives in `data_creation/` and installs as `proakt` (`pip install -e .`).

---

## The one idea to understand first

The pipeline builds **one dataset at a time**, and *which* dataset is a single
argument: `--dataset`. Everything else about that dataset — what the prediction
target is, what identifier gets stamped into the output, which covariates get
downloaded, which news feed is used — is looked up from one registry and can
therefore never disagree with itself.

That registry is `DATASETS` in [`constants.py`](constants.py):

```python
DATASETS = {
    "index":  {"target_ticker": "^NDX", "target_id": "^nsdq",
               "covariates": NASDAQ_100_YAHOO,    "news_topic": "stocks"},
    "metals": {"target_ticker": "HG=F", "target_id": "XCU",
               "covariates": METALS_MINING_YAHOO, "news_topic": "metals"},
}
```

**Adding a third dataset is one entry here and nothing else.** No new flags, no
new directories, no edits to any step. If you read nothing else in this file,
read that dict.

Every step takes `--base-dir` and `--dataset`, and derives the rest. The
`--index` / `--target` / `--id` flags this package used to have are gone: they
were three flags naming one concept, each reaching a different subset of the
steps, so a run could download copper and label the output as the Nasdaq.

---

## Requirements

- **Python ≥ 3.10** (the code uses `X | None` union syntax).
- A virtual environment is strongly recommended.
- Runtime dependencies are declared in [`requirements.txt`](requirements.txt) and
  installed by `pip install -e .` (numpy, pandas, torch, transformers, yfinance,
  pyarrow, scikit-learn, scipy, statsmodels, huggingface_hub, tqdm,
  python-dotenv, python-dateutil, …).
- An NYT Developer API key in `data_creation/api_key.env` — needed **only** by
  the `news` step. Copy `api_key.env.example` and paste a real key.

## Installation

From the `data_creation/` directory:

**Windows (PowerShell / cmd)**
```bat
cd data_creation
python -m venv venv
venv\Scripts\activate
pip install -e .
```

**Linux / macOS**
```bash
cd data_creation
python -m venv venv
source venv/bin/activate
pip install -e .
```

The editable install is what makes the top-level `constants` module and the
`scripts` package importable, which the `python -m scripts.*` invocations below
require. Run it once; after that just activate the venv.

> **Invocation note.** The scripts use absolute imports (`from scripts.cli import …`),
> so they must be launched **as modules from the `data_creation/` root**:
> `python -m scripts.main`, `python -m scripts.merge.run`, etc.
> `python scripts/main.py` fails with an import error.

---

## Quick start

```bash
python -m scripts.main --base-dir <path> --dataset index
```

```bash
python -m scripts.main --base-dir <path> --dataset metals --model financialbert
```

Both write into the same `--base-dir` without colliding — see
[Data layout](#data-layout).

**Budget the time before you start.** A cold full run is dominated by two steps:

| Step | Cost | Notes |
|------|------|-------|
| `news` | ~45 min for 2007–2026 | NYT allows ~5 req/min; the code sleeps 12 s per month fetched. Resumable. |
| `news-feat` | tens of minutes to hours | one transformer forward pass per headline, on CPU by default |

Both are cached and shared across datasets (see below), so you pay them once per
*news topic*, not once per dataset.

---

## Data layout

Everything lives under the `--base-dir` you pass. There are two kinds of
directory: **per-dataset trees**, and **one shared news cache**.

```
<base-dir>/
  news/                          shared, keyed by news TOPIC — not by dataset
    stocks/
      news.csv                     raw fetched headlines
      news_data_checkpoints/       per-month JSON, makes the fetch resumable
      news_keywords.csv            daily keyword/tag audit (review aid only)
      news_flat.csv                per-article NLP output
      news_enriched.csv            per-day aggregated NLP output
    metals/
      news_metals.csv              raw fetched headlines (combined)
      news_metals_specific.csv     review-only split: mentions an actual metal
      news_economy.csv             review-only split: macro signal only
      news_metals_checkpoints/
      news_metals_keywords.csv
      news_metals_json/            one subfolder per day, one JSON per article
      news_flat.csv
      news_enriched.csv

  index/                         per-dataset tree
    data/target/                   ^NDX.csv
    data/covariates/               AAPL.csv, MSFT.csv, …
    data_enriched/target/          ^NDX.csv        + technical indicators
    data_enriched/covariates/      AAPL.csv, …     + technical indicators
    datasets/
      target_variables.parquet
      global_covariates.parquet
      feature_covariates.parquet

  metals/                        same shape
    data/target/                   HG=F.csv
    data/covariates/               FCX.csv, BHP.csv, RIO.csv, AA.csv, REMX.csv, MP.csv
    data_enriched/…
    datasets/…
```

Three things about this layout are deliberate and worth knowing:

**News is shared, and keyed by topic rather than by dataset.** The fetch is
rate-limited to roughly 45 minutes and the embedding pass is a transformer run
over every headline — far too expensive to repeat for each dataset. Two datasets
declaring `"news_topic": "stocks"` read one fetch and one set of embeddings.

**Directories are named by role, not by asset type.** `target` / `covariates`,
not `index` / `companies`. The covariates are Nasdaq constituents for one
dataset and mining equities for another; "companies" was index-specific
vocabulary that made the shared tree shape unreadable.

**There is no `case_interp` level under `datasets/`.** It once separated two
output variants (`case_interp` vs `case_mask`); only the interpolated one is
produced now, so it discriminated nothing. The dataset name is the sole
discriminator, and it is the level that actually matters.

[`scripts/paths.py`](scripts/paths.py) is the **only** module that assembles
these paths. If you need to change the layout, change it there and nowhere else.

---

## Pipeline steps

Seven ordered steps. Each consumes what the earlier ones produced.

| # | Key | Module | What it does |
|---|-----|--------|--------------|
| 1 | `target` | `load_symbols` | Download the target's OHLCV from Yahoo Finance |
| 2 | `covariates` | `load_symbols` | Download each covariate's OHLCV |
| 3 | `news` | `load_news` | Fetch and filter NYT archive articles for the dataset's topic |
| 4 | `target-feat` | `symb_feat_gen` | Add technical indicators to the target |
| 5 | `covariates-feat` | `symb_feat_gen` | Add technical indicators to the covariates |
| 6 | `news-feat` | `news_feat_gen` | Transformer pass → `news_flat.csv` → `news_enriched.csv` |
| 7 | `merge` | `merge` | Build the three output parquets |

### Running a subset

`--only` and `--skip` take comma-separated step keys and are mutually exclusive:

```bash
python -m scripts.main --base-dir <path> --dataset metals --only merge
```

```bash
python -m scripts.main --base-dir <path> --dataset metals --skip news,news-feat
```

Two behaviours make this safe to rely on:

- **Selected steps always run in table order**, whatever order you list them in.
  `--only merge,news` runs `news` first, because `merge` consumes its output.
- **Each step checks its inputs before running.** A missing input produces
  `cannot run step 'merge' for dataset 'metals': no enriched target at
  <path> - run step 'target-feat' first`, naming the step that produces it —
  rather than a `FileNotFoundError` from somewhere inside pandas.

An unknown step key exits immediately and lists the valid ones.

---

## Command reference

### `scripts.main` — the orchestrator

| Flag | Type / choices | Default | Applies to |
|------|----------------|---------|------------|
| `--base-dir` | path (**required**) | — | all |
| `--dataset` | `index` \| `metals` (**required**) | — | all |
| `--date-start` | `YYYY-MM-DD` | `2007-01-03` | steps 1–2 |
| `--date-end` | `YYYY-MM-DD` | today | steps 1–2 |
| `--start-year` | int | `2007` | step 3 |
| `--end-year` | int | last completed month | step 3 |
| `--model` | `finbert` \| `financialbert` \| `minilm` | `finbert` | step 6 |
| `--run-flat` / `--no-run-flat` | flag | on | step 6 |
| `--cutoff-date` | `YYYY-MM-DD` | none | step 7 |
| `--min-start` | `YYYY-MM-DD` | none | step 7 |
| `--only` | comma-separated keys | all steps | — |
| `--skip` | comma-separated keys | none | — |

`--dataset` is validated against `DATASETS`, so a typo fails at parse time.

### Individual steps

Every step is runnable on its own with the same `--base-dir` + `--dataset` pair.

```bash
python -m scripts.load_symbols.run   --base-dir .. --dataset metals --mode 0   # 0=target, 1=covariates
python -m scripts.load_news.run      --base-dir .. --dataset metals --start-year 2015
python -m scripts.symb_feat_gen.run  --base-dir .. --dataset metals --mode 1
python -m scripts.news_feat_gen.run  --base-dir .. --dataset metals --model financialbert
python -m scripts.merge.run          --base-dir .. --dataset metals --min-start 2016-01-01
```

Step-specific flags beyond `--base-dir`/`--dataset`:

| Module | Flag | Meaning |
|--------|------|---------|
| `load_symbols.run` | `--mode 0\|1` | `0` = target series, `1` = covariates |
| `load_symbols.run` | `--date-start`, `--date-end` | download range |
| `load_news.run` | `--start-year`, `--end-year` | NYT archive range |
| `symb_feat_gen.run` | `--mode 0\|1` | `0` = target series, `1` = covariates |
| `news_feat_gen.run` | `--model` | which transformer |
| `news_feat_gen.run` | `--no-run-flat` | reuse an existing `news_flat.csv`, redo only the per-day aggregate |
| `merge.run` | `--cutoff-date` | drop output dates on or after it |
| `merge.run` | `--min-start` | exclude covariates whose history starts after it |

Note there is **no `--read-dir` / `--write-dir` split** any more, and
`news_feat_gen` no longer takes three separate directory arguments. Every entry
point takes one `--base-dir`, resolved to an absolute path so a step's output
location never depends on the directory you launched it from.

---

## The news step in detail

`load_news` fetches the NYT Archive API month by month and keeps an article only
if it passes that topic's filter. Each topic (`TARGETS` in
[`scripts/load_news/load_news.py`](scripts/load_news/load_news.py)) declares its
own desk/section gate, subject tags, and keyword list.

The entry point is `fetch_news(target, start_year, end_year, base_dir)`
(wrapped by [`scripts/load_news/run.py`](scripts/load_news/run.py)). `target`
is the topic key (`"stocks"` or `"metals"`), matching a dataset's
`news_topic` in the `DATASETS` registry. Output always lands under the shared
cache at `<base_dir>/news/<target>/` (see [Data layout](#data-layout)) —
never under a per-dataset tree.

### The filter

Filtering happens in two stages, and only the first one is a hard gate.

**Stage 1 — desk/section eligibility gate (`passes_gate`).** `news_desk` and
`section_name` are NYT's own *editorial* placement of the article — which
internal desk produced it, which section it ran in — not a judgment about its
content. A document is only eligible for stage 2 at all if its desk is in
`cfg["news_desk"]` or its section is in `cfg["sections"]` (either is enough;
they're OR'd, since NYT's own desk/section labeling isn't perfectly consistent
across the archive's history). This runs first and unconditionally — an
article that fails it is dropped before any tag or keyword logic ever sees it.
It exists because keyword matching alone is just a regex check against the
abstract text with no idea what kind of article it's looking at: without this
gate, "gold" would just as happily match an Olympics recap or an art review.
`METALS_NEWS_DESK`/`METALS_SECTIONS` are deliberately wider than
`STOCKS_NEWS_DESK`/`STOCKS_SECTIONS` (adding `science`, `climate`, `foreign`,
`international`, `national`) because metals stories legitimately come out of
science/climate desks in a way stock-market stories don't.

**Stage 2 — tag/keyword scoring.** Only articles that passed stage 1 are
scored into three buckets, in priority order:

| Bucket | Rule | Rationale |
|--------|------|-----------|
| `both` | broad tag match **and** ≥ 1 keyword | highest confidence |
| `nyt_only` | tight tag match, no keyword needed | the tag alone is specific enough |
| `kw_only` | ≥ 2 keywords, no tag needed | catches articles NYT tagged loosely |

Here, tags are NYT's own *content* classification (their subject-tag
taxonomy), and keywords are this pipeline's own regex match against the
abstract text — both are signals about what the article is about, checked
only once stage 1 has already established that it's eligible to be about it.

`stocks` predates the tight/broad split and uses one tag set for both, which
reduces it to the original two-tier filter. `metals` uses a genuinely tighter
tag set for the tag-alone bucket, because its broad tag set includes generic
macro tags that would otherwise let too much through unsupported.

Every article that survives the filter becomes a **record** —
`{date, text, category, keywords, tags, id, web_url, headline, byline,
pub_date, news_desk, section_name, document_type, type_of_material,
word_count, source}` (`v2_filter` in `load_news.py`; the metadata fields come
from `extract_metadata`). `category` is `"metal"` / `"economy"` for topics with
`split_metal_economy` set, otherwise `None`. This is the one record shape that
feeds every output below — the CSVs and the per-article JSON both derive from
the same in-memory `{date: [record, ...]}` structure per run.

### Resumability

Every completed month is checkpointed as JSON (the raw record list, unchanged)
under the topic's checkpoint directory (`news_data_checkpoints/` for stocks,
`news_metals_checkpoints/` for metals) and skipped on re-run. A failed request
saves whatever's accumulated so far and stops the run entirely — it does not
retry, so a single flaky NYT response ends the fetch and requires a manual
re-run (which resumes from the last checkpointed month for free). Months that
have not fully ended are never fetched — checkpointing a partial month would
record it as done with too few articles and permanently hide the rest.

### Outputs, per topic

Every topic writes:

- `<out_name>` (`news.csv` / `news_metals.csv`) — the combined, filtered set.
  This is what `news_feat_gen` and `merge` read.
- `<keywords_out_name>` (`news_keywords.csv` / `news_metals_keywords.csv`) —
  per calendar day, the pipe-separated set of keywords and tags that matched
  that day's articles. Review-only, not read by any other step.

**Metals additionally writes** (`TARGETS["metals"]` sets
`split_metal_economy: True` and `json_dirname`):

- `news_metals_specific.csv` / `news_economy.csv` — an exclusive partition of
  `news_metals.csv` by `category`: `news_metals_specific.csv` holds articles
  that matched via an actual metal/mine/critical-mineral signal
  (`METALS_SPECIFIC_KEYWORDS` / `METALS_SPECIFIC_TAGS`), `news_economy.csv`
  holds the rest — articles that matched only through a general macro signal
  such as inflation or tariffs, with no metal mention at all. The two add back
  up to `news_metals.csv` exactly. Review-only, same as the keyword audit — no
  downstream step reads them.
- `news_metals_json/<date>/<article_id>.json` — one JSON file per article, one
  subfolder per calendar day (`save_json_articles`). Each file is the full
  record for that article verbatim — `date`, `text`, `category`, `keywords`,
  `tags`, and every `extract_metadata` field — so it's self-contained without
  needing to cross-reference the CSVs. Metal and economy articles share the
  same day folder undistinguished (their `category`/`keywords`/`tags` fields
  already carry that information for anyone who needs it). The filename is the
  NYT article id (sanitized), or `<date>_<index>` if an id is missing. Each
  day's folder is cleared and rewritten on every run, so it stays consistent
  with whatever `news_metals.csv` currently contains — it is not an
  append-only log. Not read by any other step; it exists for manual
  inspection / ad hoc analysis of individual articles.

Adding this to `stocks` (or a future topic) is a one-line change: give its
`TARGETS` entry a `json_dirname`.

---

## The NLP step in detail

Three models are available:

| `--model` | Hugging Face id | Dim | Sentiment |
|-----------|-----------------|-----|-----------|
| `finbert` | `ProsusAI/finbert` | 768 | yes |
| `financialbert` | `ahmedrachid/FinancialBERT-Sentiment-Analysis` | 768 | yes |
| `minilm` | `sentence-transformers/all-MiniLM-L6-v2` | 384 | **no** — embeddings only |

### Class ordering — read this before adding a fourth model

The two sentiment models **do not agree on class order**:

| Model | `config.id2label` |
|-------|-------------------|
| FinBERT | `{0: positive, 1: negative, 2: neutral}` |
| FinancialBERT | `{0: negative, 1: neutral, 2: positive}` |

Probabilities are therefore mapped into the `prob_*` columns **by label name**,
read from each model's own `config.id2label` — never by index. Assigning them
positionally (as the code originally did, with FinBERT's order baked in) would
write the *negative* probability into `prob_positive` for FinancialBERT, with no
error anywhere: every sentiment feature silently inverted.

If you add a model, you get this for free as long as its three class names are
`positive` / `negative` / `neutral`. `label_index()` validates exactly that and
raises if the names differ, rather than guessing.

A useful invariant when checking a new model: in `news_flat.csv`, the `label`
column must always name the highest-probability `prob_*` column. That holds
under any class ordering *only* if the mapping is by name.

### `--no-run-flat`

The per-article pass is the expensive half. `--no-run-flat` skips it and
recomputes only `news_enriched.csv` (the per-day mean) from an existing
`news_flat.csv` — useful when changing the aggregation, not the model. The
precondition check will tell you if no flat file exists.

---

## Output

Three parquets per dataset, in `<base-dir>/<dataset>/datasets/`:

| File | Shape | Contents |
|------|-------|----------|
| `target_variables.parquet` | long | `date, id, open, high, low, close` — the prediction target |
| `global_covariates.parquet` | wide | `date` + `{TICKER}_{feat}` panel + target tech + news |
| `feature_covariates.parquet` | wide | `date` + six sin/cos calendar encodings |

Full build logic and schema: [`dataset_builder.md`](dataset_builder.md).

### Everything is raw

`data_creation` always writes **raw (undifferenced) levels**. Differencing is a
model-layer transform applied at load time in `model_impl`
(`apply_differencing` in `model_impl/utils/data_loader_utils/transforms.py`,
selected by `type_of_diff` / `TYPE_OF_DIFF`). One output tree therefore serves
every variant, and choosing raw / first-order / log-return is a `model_impl`
concern — not a flag here.

### Enriched OHLCV columns (steps 4–5)

| Column | Description |
|--------|-------------|
| `Movement` | `[Open(t) − Close(t−1)] / Close(t−1)` — overnight gap |
| `Daily_Return` | % change `Close(t−1) → Close(t)` |
| `Volatility` | rolling 5-day std of daily returns |
| `EMA_12`, `EMA_26` | exponential moving averages |
| `MACD` | `EMA_12 − EMA_26` |
| `RSI` | relative strength index, 0–100 |
| `Stoch_K`, `Stoch_D` | stochastic oscillator %K and smoothed %D |
| `Williams_R` | Williams %R, −100 to 0 |
| `ROC` | % price change over 10 days |

### Enriched news columns (step 6)

| Column | Description |
|--------|-------------|
| `embedding` | CLS vector averaged over the day's headlines (768 or 384-dim) |
| `label` | dominant sentiment: `positive`, `negative`, `neutral` |
| `prob_positive`, `prob_negative`, `prob_neutral` | probabilities averaged over the day's headlines |

With `--model minilm` the last four are absent. Downstream code checks for them
rather than assuming — see `has_sentiment()` in the builder.

---

## Config

Directory names, filenames, column lists, ticker pools and the dataset registry
live in [`constants.py`](constants.py). Path *construction* lives in
[`scripts/paths.py`](scripts/paths.py) — constants declare names, paths assemble
them.

### Output-shape toggles

| Constant | Default | Effect |
|----------|---------|--------|
| `GLOBAL_INCLUDE_TECH_FEATURES` | `True` | fold technical indicators into `global_covariates` |
| `WRITE_FEATURE_COVARIATES` | `True` | emit `feature_covariates.parquet` |
| `WRITE_NEWS_TO_GLOBAL` | `True` | fold the news `embedding` into `global_covariates` |
| `NEWS_INCLUDE_SENTIMENT` | `True` | add `prob_*` / `label` beside the embedding |
| `GLOBAL_COV_SEP` | `_` | wide-column separator (`AAPL_close`) |
| `GLOBAL_COVARIATE_PRICE_COLS` | `["Open","High","Low","Close"]` | price columns copied into the wide panel |
| `TARGET_OHLC_COLS` | `["Open","High","Low","Close"]` | OHLC columns in the long target file |

> **`WRITE_NEWS_TO_GLOBAL` is a handshake with `model_impl`.** Turn it off and
> the global parquet has no `embedding` column; `model_impl` must then run with
> `no_news=True` or it fails with
> `ArrowInvalid: No match for FieldRef.Name(embedding)`.

### Covariate ticker pools

`NASDAQ_100_YAHOO` and `METALS_MINING_YAHOO` are the *available pools*. Which
columns a given experiment actually consumes is a `model_impl` choice, made via
`GLOBAL_COVARIATES` in the model yaml.

Every ticker in a pool is used **verbatim** as both the yfinance download symbol
and the wide-panel column prefix (`FCX` → `FCX_close`). That equality is why the
metals pool deliberately omits the energy, FX and STOXX columns present in the
older metals parquet: those are the only ones whose column prefix differs from
their yfinance symbol (`CL1` → `CL=F`, `eur_usd` → `EURUSD=X`), and none of them
cleared the relevance filter in `val_data/correlation/RESULTS.md` anyway.

A leading `^` is stripped from column prefixes (`^NDX` → `NDX_close`) because
`^` is not allowed in MLflow param/tag keys.

---

## Migrating from the old layout

Trees built before the restructure are flat and single-dataset:

```
<base>/data/{index,companies,news}/
<base>/data_enriched/{index,companies,news}/
<base>/datasets/case_interp/
```

The code no longer understands this shape, and there is no migration script.
To reuse an existing tree, move it by hand — the file *contents* are unchanged,
only their locations moved:

| Old | New |
|-----|-----|
| `data/index/^NDX.csv` | `index/data/target/^NDX.csv` |
| `data/companies/*.csv` | `index/data/covariates/*.csv` |
| `data/news/news.csv` | `news/stocks/news.csv` |
| `data/news/news_data_checkpoints/` | `news/stocks/news_data_checkpoints/` |
| `data_enriched/index/^NDX.csv` | `index/data_enriched/target/^NDX.csv` |
| `data_enriched/companies/*.csv` | `index/data_enriched/covariates/*.csv` |
| `data_enriched/news/news_flat.csv` | `news/stocks/news_flat.csv` |
| `data_enriched/news/news_enriched.csv` | `news/stocks/news_enriched.csv` |
| `datasets/case_interp/*.parquet` | `index/datasets/*.parquet` |

Moving the news files is the one that pays for itself — it saves the ~45-minute
fetch and the full embedding pass.

`build_index_dataset.sh` has been deleted. It hardcoded the old flat layout and
the removed `--read-dir`/`--write-dir`/`--id` flags; its only real value was a
preflight check, which the step preconditions now do for every step.

---

## Notes for the incoming maintainer

A few things that are easy to get wrong, and where the sharp edges are:

**`--dataset` is the whole design.** Resist adding a flag that duplicates
something the registry already knows. The bugs this package had were almost all
the same shape: configuration expressed at the CLI but resolved from a hardcoded
constant deeper down, so the pipeline worked on exactly one golden path and
produced wrong output — silently — off it.

**Prefer failing loudly over defaulting.** `dataset_config()` raises on an
unknown key and tells you to add a registry entry. `load_covariates` prints every
ticker it excludes, by name and start date, because a filter that silently
dropped a third of the metals covariate set is how REMX and MP went missing for
months. `label_index()` raises if a model's class names are unexpected rather
than guessing an order.

**Don't hardcode a date.** The cutoff used to be a module constant, which meant
every build silently stopped at a date that quietly went stale. It is
`--cutoff-date` now, defaulting to no cutoff. Same reasoning applies to the news
end-year, which resolves to the last completed month on every call.

**The expensive steps are cached; keep it that way.** If you add a dataset that
shares an existing news topic, it costs nothing extra. If you give it a new
topic, budget the full fetch plus embedding pass.
