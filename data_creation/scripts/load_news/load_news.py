import os
import re
import csv
import json
import time
import datetime
import requests
from collections import defaultdict
from dateutil.relativedelta import relativedelta
from pathlib import Path
from dotenv import load_dotenv

from scripts.paths import news_dir as topic_news_dir

# ── Configuration ──────────────────────────────────────────────────────────────
SLEEP_SECS = 12   # stay safely under NYT 10 req/min limit
# ──────────────────────────────────────────────────────────────────────────────

# ── stocks filter config ───────────────────────────────────────────────────────
STOCKS_NEWS_DESK = {
    "business", "business day", "entrepreneurs", "financial", "technology",
    "markets", "economy", "dealbook", "u.s.", "energy",
    "world business", "investing", "real estate",
}
STOCKS_SECTIONS = {
    "business", "business day", "technology", "u.s.", "world",
    "your money", "economy", "dealbook", "energy", "real estate",
}
STOCKS_TAGS = {
    "Stocks and Bonds", "Standard & Poor's 500-Stock Index", "Nasdaq Composite Index",
    "Dow Jones Stock Index", "Russell 2000 Index", "Initial Public Offerings",
    "Short Sales", "Stock Options and Ownership Plans", "Stock Buybacks",
    "Insider Trading and Violations", "Futures and Options Trading",
    "Shareholder Rights and Activism", "Dividends",
    "Special Purpose Acquisition Companies (SPAC)",
    "United States Economy", "Inflation (Economics)", "Interest Rates",
    "Economic Conditions and Trends", "Recession and Depression",
    "Gross Domestic Product", "Consumer Price Index", "Unemployment",
    "Labor and Jobs", "Wages and Salaries", "Layoffs and Job Reductions",
    "Stimulus (Economic)", "Federal Budget (US)", "Prices (Fares, Fees and Rates)",
    "Customs (Tariff)", "Protectionism (Trade)", "Embargoes and Sanctions",
    "International Trade and World Market", "Foreign Investments", "Supply Chain",
    "Banking and Financial Institutions", "Regulation and Deregulation of Industry",
    "Antitrust Laws and Competition Issues", "Securities and Commodities Violations",
    "Frauds and Swindling", "Boards of Directors",
    "Tax Credits, Deductions and Exemptions", "Income Tax", "Federal Taxes (US)", "Taxation",
    "Mergers, Acquisitions and Divestitures", "Company Reports", "Corporations",
    "Bankruptcies", "Credit and Debt", "Government Bonds", "Currency", "Hedge Funds",
    "Private Equity", "Venture Capital", "Start-ups", "Entrepreneurship", "Small Business",
    "Income Inequality", "Income",
    "Oil (Petroleum) and Gasoline", "Natural Gas", "Alternative and Renewable Energy",
    "Electric and Hybrid Vehicles", "Computer Chips",
    "Virtual Currency", "E-Commerce", "Advertising and Marketing",
    "Inflation Reduction Act of 2022",
}
STOCKS_KEYWORDS = [
    "nasdaq", "s&p 500", "s&p500", "dow jones", "stock market", "stock exchange",
    "wall street", "ipo", "initial public offering", "earnings", "quarterly results",
    "stock", "stocks", "shares", "equities", "equity", "trading", "market cap",
    "short selling", "short sell", "bull market", "bear market", "rally", "selloff", "sell-off",
    "federal reserve", "fed rate", "interest rate", "interest rates", "inflation",
    "recession", "gdp", "unemployment", "jobs report", "consumer price", "cpi",
    "tariff", "trade war", "semiconductor", "antitrust", "rare earth", "banks", "lenders",
    "revenue", "profit", "losses", "merger", "acquisition", "dividend", "buyback",
    "debt", "yield", "hedge fund", "private equity", "venture capital",
    "economy", "economic", "financial", "investors", "investment", "growth", "market",
]
# ──────────────────────────────────────────────────────────────────────────────

# ── metals filter config ───────────────────────────────────────────────────────
# Wider desk/section gate than stocks — includes science, climate, world, foreign
METALS_NEWS_DESK = {
    "business", "business day", "financial", "markets", "economy", "dealbook",
    "technology", "energy", "world business", "investing",
    "u.s.", "world", "foreign", "international",
    "science", "climate", "national",
}
METALS_SECTIONS = {
    "business", "business day", "technology", "economy", "dealbook", "energy",
    "u.s.", "world", "your money", "science", "climate",
}
# Tight tags: specific enough that a tag match alone is reliable signal
METALS_TIGHT_TAGS = {
    "Oil (Petroleum) and Gasoline",
    "Natural Gas",
    "Interest Rates",
    "Gross Domestic Product",
    "Inflation (Economics)",
    "Recession and Depression",
    "Customs (Tariff)",
    "Protectionism (Trade)",
    "Supply Chain",
    "Coronavirus (2019-nCoV)",
    "Russian Invasion of Ukraine (2022)",
    "Subprime Mortgage Crisis",
    "European Sovereign Debt Crisis (2010- )",
    "Japan Earthquake and Tsunami (2011)",
    "Middle East and North Africa Unrest (2010- )",
    "Electric and Hybrid Vehicles",
    "Computer Chips",
}
METALS_TAGS = {
    # ── Macro / trade ────────────────────────────────────────────────────────
    "United States Economy",
    "Economic Conditions and Trends",
    "International Trade and World Market",
    "Customs (Tariff)",
    "Protectionism (Trade)",
    "Foreign Investments",
    "Supply Chain",
    "Prices (Fares, Fees and Rates)",
    "Inflation (Economics)",
    "Interest Rates",
    "Recession and Depression",
    "Gross Domestic Product",
    "Unemployment",
    "Stimulus (Economic)",
    "Federal Budget (US)",

    # ── Financial / corporate ────────────────────────────────────────────────
    "Banking and Financial Institutions",
    "Banks and Banking",
    "Mergers, Acquisitions and Divestitures",
    "Company Reports",
    "Corporations",
    "Bankruptcies",
    "Credit and Debt",
    "Stocks and Bonds",
    "Hedge Funds",
    "Private Equity",
    "Regulation and Deregulation of Industry",
    "Antitrust Laws and Competition Issues",
    "Frauds and Swindling",
    "Layoffs and Job Reductions",
    "Organized Labor",
    "Taxation",
    "Federal Taxes (US)",

    # ── Energy / environment ─────────────────────────────────────────────────
    "Oil (Petroleum) and Gasoline",
    "Natural Gas",
    "Alternative and Renewable Energy",
    "Electric and Hybrid Vehicles",
    "Nuclear Energy",
    "Environment",
    "Factories and Manufacturing",

    # ── Geopolitics / crisis events ──────────────────────────────────────────
    "Espionage and Intelligence Services",
    "War and Armed Conflicts",
    "Russian Invasion of Ukraine (2022)",
    "Subprime Mortgage Crisis",
    "European Sovereign Debt Crisis (2010- )",
    "Japan Earthquake and Tsunami (2011)",
    "Middle East and North Africa Unrest (2010- )",
    "Coronavirus (2019-nCoV)",

    # ── Technology demand side ────────────────────────────────────────────────
    "Automobiles",
    "Science and Technology",
    "Computer Chips",
}

# Subject tags that name (or directly imply) a physical metal/mineral supply
# or demand story, as opposed to a general macro/economy/crisis tag. Used to
# split the "metals" target's output into a metal-specific CSV and an
# economy/other CSV — see split_metal_economy in TARGETS.
METALS_SPECIFIC_TAGS = {
    "Supply Chain",
    "Alternative and Renewable Energy",
    "Electric and Hybrid Vehicles",
    "Automobiles",
    "Computer Chips",
}

# ── Direct metals ────────────────────────────────────────────────────────────
METALS_SPECIFIC_KEYWORDS = [
    "gold", "silver", "copper", "cobalt", "tungsten", "iron ore",
    "coking coal", "metallurgical coal", "rare earth", "rare earths",
    "lithium", "nickel", "zinc", "aluminum", "aluminium", "platinum",
    "palladium", "manganese", "chromium", "molybdenum", "vanadium",
    "neodymium", "dysprosium", "praseodymium", "terbium",
    "indium", "gallium", "germanium",

    # ── Metals market / trading ───────────────────────────────────────────────
    "bullion", "commodity", "commodities", "futures", "spot price",
    "lme", "london metal exchange", "comex", "shanghai metals",
    "metal prices", "base metals", "precious metals", "industrial metals",
    "ore", "smelting", "refining", "alloy",

    # ── Mining / production ───────────────────────────────────────────────────
    "mining", "mine", "miner", "miners", "open pit", "underground mine",
    "drilling", "extraction", "reserves", "ore grade",
    "rio tinto", "bhp", "glencore", "freeport", "vale", "anglo american",
    "barrick", "newmont", "antofagasta",

    # ── Supply chain / geopolitics (metals & critical minerals specific) ─────
    "china supply", "china dominance", "strategic reserve", "stockpile",
    "export ban", "export restriction", "critical mineral", "critical metals",
    "supply disruption", "mine strike", "production cut",

    # ── Demand drivers (physical metal demand) ────────────────────────────────
    "electric vehicle", "ev battery", "battery storage", "energy storage",
    "solar panel", "wind turbine", "green energy", "energy transition",
    "semiconductor", "chip shortage", "defense spending", "arms",
    "construction", "infrastructure spending", "steel", "stainless steel",
]
# ── Oil / energy and broader macro signals — not a metal, but part of what   ──
# ── moves metal prices, so still fetched under the "metals" target          ──
ECONOMY_KEYWORDS = [
    # ── Oil / energy (macro cost driver, not a metal itself) ─────────────────
    "crude oil", "brent", "wti", "opec", "oil price", "oil prices",
    "natural gas price", "energy price", "energy prices", "energy cost",

    # ── Macro signals that move metals ────────────────────────────────────────
    "federal reserve", "interest rate", "interest rates", "inflation",
    "dollar index", "dxy", "commodity prices", "commodity index",
    "recession", "gdp", "trade war", "tariff", "sanctions", "embargo",
    "china economy", "china gdp", "china manufacturing", "china demand",
    "pmi", "purchasing managers", "manufacturing output",
    "industrial production", "capacity utilization",
    "global growth", "economic slowdown", "economic growth",
]
METALS_KEYWORDS = METALS_SPECIFIC_KEYWORDS + ECONOMY_KEYWORDS
METALS_SPECIFIC_KEYWORDS_SET = set(METALS_SPECIFIC_KEYWORDS)
# ──────────────────────────────────────────────────────────────────────────────

# ── targets ────────────────────────────────────────────────────────────────────
# Each target supplies: which news_desk/section values pass the gate, which NYT
# subject tags count as "broad" (needs >= 1 keyword too) vs "tight" (reliable
# alone, no keyword needed), the keyword list, and where output lands.
# Stocks predates the tight/broad split, so it uses one tag set for both.
#
# split_metal_economy (metals only): in addition to the combined out_name CSV
# (everything the metals filter catches — unchanged, still what news_feat_gen/
# merge read), also partition articles into metal_out_name (mentions an actual
# metal/mining/critical-mineral signal) and economy_out_name (matched only via
# a general macro/economy signal, e.g. "inflation" or "interest rates", with no
# metal mention at all) — exclusive, so metal_out_name + economy_out_name
# reconstitute out_name exactly with no overlap.
TARGETS = {
    "stocks": {
        "news_desk":         STOCKS_NEWS_DESK,
        "sections":          STOCKS_SECTIONS,
        "broad_tags":        STOCKS_TAGS,
        "tight_tags":        STOCKS_TAGS,
        "keywords":          STOCKS_KEYWORDS,
        "out_name":          "news.csv",
        "ckpt_dirname":      "news_data_checkpoints",
        "keywords_out_name": "news_keywords.csv",
        "split_metal_economy": False,
    },
    "metals": {
        "news_desk":         METALS_NEWS_DESK,
        "sections":          METALS_SECTIONS,
        "broad_tags":        METALS_TAGS,
        "tight_tags":        METALS_TIGHT_TAGS,
        "keywords":          METALS_KEYWORDS,
        "out_name":          "news_metals.csv",
        "ckpt_dirname":      "news_metals_checkpoints",
        "keywords_out_name": "news_metals_keywords.csv",
        "split_metal_economy": True,
        "metal_out_name":    "news_metals_specific.csv",
        "economy_out_name":  "news_economy.csv",
        "json_dirname":      "news_metals_json",
    },
}
# ──────────────────────────────────────────────────────────────────────────────

# api_key.env lives at the data_creation/ package root regardless of the cwd
# the pipeline is launched from.
load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / "api_key.env")
NYT_API_KEY = os.getenv("NYT_API_KEY")


def fetch_archive(year, month):
    if not NYT_API_KEY or NYT_API_KEY == "Paste_your_key_here":
        raise RuntimeError("NYT_API_KEY is not set — put a real key in data_creation/api_key.env")
    url  = f"https://api.nytimes.com/svc/archive/v1/{year}/{month}.json"
    resp = requests.get(url, params={"api-key": NYT_API_KEY}, timeout=60)
    resp.raise_for_status()
    return resp.json().get("response", {}).get("docs") or []


def get_abstract(doc):
    text = (doc.get("abstract") or "").replace(",", "").strip()
    if not text:
        text = (doc.get("headline", {}).get("main") or "").replace(",", "").strip()
    return text


def extract_metadata(doc):
    """Article-level metadata carried into the per-article JSON dumps."""
    headline = doc.get("headline") or {}
    byline   = doc.get("byline") or {}
    return {
        "id":               doc.get("_id", ""),
        "web_url":          doc.get("web_url", ""),
        "headline":         headline.get("main", ""),
        "byline":           byline.get("original", ""),
        "pub_date":         doc.get("pub_date", ""),
        "news_desk":        doc.get("news_desk", ""),
        "section_name":     doc.get("section_name", ""),
        "document_type":    doc.get("document_type", ""),
        "type_of_material": doc.get("type_of_material", ""),
        "word_count":       doc.get("word_count", ""),
        "source":           doc.get("source", ""),
    }


def passes_gate(doc, cfg):
    desk    = (doc.get("news_desk")    or "").lower()
    section = (doc.get("section_name") or "").lower()
    return desk in cfg["news_desk"] or section in cfg["sections"]


def matched_tags(doc, tags):
    """Set of NYT subject-tag values on doc that are in the given tags set."""
    return {
        kw.get("value", "") for kw in (doc.get("keywords") or [])
        if kw.get("name", "").lower() == "subject" and kw.get("value", "") in tags
    }


def has_tag(doc, tags):
    return bool(matched_tags(doc, tags))


def matched_keywords(text, keywords):
    """Set of keywords (from the given list) found in text."""
    lower = text.lower()
    return {kw for kw in keywords if re.search(r'\b' + re.escape(kw) + r's?\b', lower)}


def kw_count(text, keywords):
    return len(matched_keywords(text, keywords))


def v2_filter(docs, cfg):
    """
    Three-bucket filter, priority-ordered:
      both     — broad_tags match + >= 1 keyword    (highest confidence)
      nyt_only — tight_tags match, no keyword needed (tag specific enough alone)
      kw_only  — >= 2 keywords, no tag needed
    For "stocks", broad_tags == tight_tags, so this reduces to the original
    two-tier stocks filter.

    Returns {date: [record, ...]}, where each record is
    {"date", "text", "category", "keywords", "tags", ...metadata}. "category"
    is "metal" or "economy" when cfg["split_metal_economy"] is set (metal if
    any matched keyword/tag is in METALS_SPECIFIC_KEYWORDS/METALS_SPECIFIC_TAGS,
    else economy), otherwise None. "keywords"/"tags" are the full matched sets
    (sorted lists), kept for the per-day keyword audit CSV regardless of
    target. The remaining fields come from extract_metadata(doc) and are
    kept so the record can be dumped as a self-contained per-article JSON
    (see save_json_articles) regardless of target.
    """
    pool = defaultdict(lambda: {"both": [], "nyt_only": [], "kw_only": []})
    for doc in docs:
        if not passes_gate(doc, cfg):
            continue
        pub  = (doc.get("pub_date") or "")[:10]
        text = get_abstract(doc)
        if not pub or not text:
            continue

        broad_hits = matched_tags(doc, cfg["broad_tags"])
        tight_hits = matched_tags(doc, cfg["tight_tags"])
        kw_hits    = matched_keywords(text, cfg["keywords"])
        all_tags   = broad_hits | tight_hits

        if broad_hits and len(kw_hits) >= 1:
            bucket = "both"
        elif tight_hits:
            bucket = "nyt_only"
        elif len(kw_hits) >= 2:
            bucket = "kw_only"
        else:
            continue

        category = None
        if cfg.get("split_metal_economy"):
            is_metal = bool(kw_hits & METALS_SPECIFIC_KEYWORDS_SET) or bool(all_tags & METALS_SPECIFIC_TAGS)
            category = "metal" if is_metal else "economy"

        pool[pub][bucket].append({
            "date":     pub,
            "text":     text,
            "category": category,
            "keywords": sorted(kw_hits),
            "tags":     sorted(all_tags),
            **extract_metadata(doc),
        })
    return {
        date: b["both"] + b["nyt_only"] + b["kw_only"]
        for date, b in pool.items()
    }


def ckpt_path(ckpt_dir, year, month):
    return ckpt_dir / f"{year}-{month:02d}.json"


def save_checkpoint(ckpt_dir, year, month, articles_by_date):
    """articles_by_date: {date: [record, ...]} — see v2_filter."""
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_path(ckpt_dir, year, month)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(dict(sorted(articles_by_date.items())), f, ensure_ascii=False)
    os.replace(tmp, path)


def load_checkpoint(ckpt_dir, year, month):
    p = ckpt_path(ckpt_dir, year, month)
    if not p.exists():
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def all_months(start_year, end_year):
    d   = datetime.date(start_year, 1, 1)
    end = datetime.date(end_year, 12, 31)
    months = []
    while d <= end:
        months.append((d.year, d.month))
        d += relativedelta(months=1)
    return months


def last_completed_month(today=None):
    """
    (year, month) of the most recent calendar month that has fully ended.
    Resolved fresh on every call — never a stored/hardcoded value.
    e.g. today=2026-07-16 -> (2026, 6).
    """
    today = today or datetime.date.today()
    last_day_of_prev_month = today.replace(day=1) - datetime.timedelta(days=1)
    return last_day_of_prev_month.year, last_day_of_prev_month.month


def month_end(year, month):
    """Last calendar date of the given (year, month)."""
    return datetime.date(year, month, 1) + relativedelta(months=1) - datetime.timedelta(days=1)


def calendar_dates(start_year, end_date):
    start = datetime.date(start_year, 1, 1)
    delta = (end_date - start).days + 1
    return [(start + datetime.timedelta(days=i)).isoformat() for i in range(delta)]


def records_to_texts(articles_by_date, category=None):
    """
    Flattens {date: [record, ...]} down to {date: [text, ...]}. With
    category=None, every record's text is kept (the combined view); with
    category="metal"/"economy", only records classified into that category
    are kept (see v2_filter / split_metal_economy).
    """
    if category is None:
        return {date: [r["text"] for r in records] for date, records in articles_by_date.items()}
    return {
        date: [r["text"] for r in records if r["category"] == category]
        for date, records in articles_by_date.items()
    }


def save_news_csv(all_articles_by_date, start_year, end_date, out_csv):
    """Write a Date + News1..N CSV with dynamic column count."""
    if not all_articles_by_date:
        print(f"No articles for {out_csv.name} — nothing to save.")
        return

    max_articles = max((len(v) for v in all_articles_by_date.values()), default=0)
    header = ["Date"] + [f"News {i+1}" for i in range(max_articles)]
    all_dates = calendar_dates(start_year, end_date)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_csv.with_suffix(".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for d in all_dates:
            arts = all_articles_by_date.get(d, [])
            row  = [d] + arts + [""] * (max_articles - len(arts))
            writer.writerow(row)
    os.replace(tmp, out_csv)

    days_with_news = sum(1 for d in all_dates if all_articles_by_date.get(d))
    total_articles = sum(len(v) for v in all_articles_by_date.values())
    print(f"\nSaved: {out_csv}")
    print(f"  Date range : {start_year}-01-01 to {end_date.isoformat()}  ({len(all_dates)} days)")
    print(f"  Days with news    : {days_with_news}")
    print(f"  Total articles    : {total_articles}")
    print(f"  Max articles/day  : {max_articles}")
    print(f"  Columns           : Date + News 1..{max_articles}")


def save_keyword_audit(all_articles_by_date, start_year, end_date, out_csv):
    """
    Write a Date + Keywords + Tags CSV: for each calendar day, the sorted set
    of unique keywords/subject-tags that matched across that day's articles
    (pipe-separated). Manual-review aid — not consumed by the rest of the
    pipeline — so a human can see exactly what triggered each day's news.
    """
    all_dates = calendar_dates(start_year, end_date)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_csv.with_suffix(".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Date", "Keywords", "Tags"])
        for d in all_dates:
            records  = all_articles_by_date.get(d, [])
            keywords = sorted({kw for r in records for kw in r["keywords"]})
            tags     = sorted({tag for r in records for tag in r["tags"]})
            writer.writerow([d, "|".join(keywords), "|".join(tags)])
    os.replace(tmp, out_csv)
    print(f"Saved: {out_csv}")


def sanitize_filename(name):
    return re.sub(r'[^A-Za-z0-9_.-]', "_", name)


def save_json_articles(all_articles_by_date, out_dir):
    """
    Writes <out_dir>/<date>/<article_id>.json — one JSON file per article,
    one subfolder per calendar day. Each JSON is a self-contained record:
    date, text, keywords, tags, category, plus the extract_metadata(doc)
    fields. Uses the combined (unsplit) record set — metal and economy
    articles land in the same day folder with no distinction, since their
    keywords/tags are already in the JSON for anyone who needs to tell them
    apart later.

    Idempotent/rerunnable: each day folder is cleared before rewriting, so
    stale files from an earlier run (e.g. after a filter change reduces that
    day's article count) don't linger.
    """
    for date, records in all_articles_by_date.items():
        if not records:
            continue
        day_dir = out_dir / date
        day_dir.mkdir(parents=True, exist_ok=True)
        for old in day_dir.glob("*.json"):
            old.unlink()
        for i, record in enumerate(records):
            article_id = sanitize_filename(record.get("id") or f"{date}_{i}")
            path = day_dir / f"{article_id}.json"
            tmp  = path.with_suffix(".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(record, f, ensure_ascii=False, indent=2)
            os.replace(tmp, path)
    print(f"Saved per-article JSONs under: {out_dir}")


def save_outputs(cfg, all_articles, start_year, end_date, out_csv, metal_csv, economy_csv, keywords_csv, json_dir):
    save_news_csv(records_to_texts(all_articles), start_year, end_date, out_csv)
    if cfg.get("split_metal_economy"):
        save_news_csv(records_to_texts(all_articles, category="metal"), start_year, end_date, metal_csv)
        save_news_csv(records_to_texts(all_articles, category="economy"), start_year, end_date, economy_csv)
    save_keyword_audit(all_articles, start_year, end_date, keywords_csv)
    if json_dir is not None:
        save_json_articles(all_articles, json_dir)


def fetch_news(target: str = "stocks", start_year: int = 2007, end_year: int | None = None, base_dir: str = "."):
    """
    Fetch and filter NYT archive articles for one news topic.

    Articles are fetched month by month between ``start_year`` and ``end_year``
    and filtered with that topic's v2 dual-filter (see ``TARGETS``).

    Output lands in the *shared* news cache at ``<base_dir>/news/<target>/``,
    outside any dataset tree: the result depends only on the topic, and both
    the fetch (NYT rate-limiting makes a 2007-2026 span take roughly 45
    minutes) and the downstream embedding pass are far too expensive to repeat
    per dataset. Every dataset whose ``news_topic`` is ``target`` reads this
    one copy.

    For targets with a json_dirname (currently "metals" only), also writes
    ``<base_dir>/news/<target>/<json_dirname>/<date>/<article_id>.json`` — one
    JSON per article, one subfolder per calendar day, combined view (metal and
    economy articles together, undistinguished — see save_json_articles).

    For ``"metals"`` (``split_metal_economy=True``), two further CSVs are
    written: ``metal_out_name`` (articles mentioning an actual metal/mining
    signal) and ``economy_out_name`` (articles matched only via a general
    macro/economy signal). The partition is exclusive, so the two add back up
    to the combined file. Every topic also gets a ``keywords_out_name`` CSV
    listing, per calendar day, the keywords/tags that matched — a manual-review
    aid, not read by any other step.

    Parameters
    ----------
    target : str, optional
        News topic — a key of ``TARGETS`` (``"stocks"`` or ``"metals"``).
        Supplied by the caller from the dataset registry's ``news_topic``.
    start_year : int, optional
        First year of the NYT archive to fetch.
    end_year : int or None, optional
        Last year to fetch. None means "through the last fully completed
        calendar month", resolved fresh on every call — never a hardcoded
        year. However the range is obtained, any month that has not fully
        ended is dropped: fetching one early would checkpoint it as done with
        zero articles, permanently hiding its real articles later.
    base_dir : str, optional
        Root directory holding the shared ``news/`` cache.

    Returns
    -------
    None
        Writes the topic's CSVs into ``<base_dir>/news/<target>/``. Resumable —
        completed months are cached under that directory and skipped on re-run.

    Raises
    ------
    ValueError
        If ``target`` is not a known topic.
    """
    if target not in TARGETS:
        raise ValueError(f"Unknown news topic {target!r} - must be one of {list(TARGETS)}")
    cfg = TARGETS[target]

    news_dir     = Path(topic_news_dir(base_dir, target))
    out_csv      = news_dir / cfg["out_name"]
    ckpt_dir     = news_dir / cfg["ckpt_dirname"]
    keywords_csv = news_dir / cfg["keywords_out_name"]
    metal_csv    = news_dir / cfg["metal_out_name"]   if cfg.get("split_metal_economy") else None
    economy_csv  = news_dir / cfg["economy_out_name"] if cfg.get("split_metal_economy") else None
    json_dir     = news_dir / cfg["json_dirname"]     if cfg.get("json_dirname")        else None

    cap_year, cap_month = last_completed_month()
    cap_date          = month_end(cap_year, cap_month)
    resolved_end_year = end_year if end_year is not None else cap_year
    end_date          = min(datetime.date(resolved_end_year, 12, 31), cap_date)

    months = [(y, m) for (y, m) in all_months(start_year, resolved_end_year)
              if datetime.date(y, m, 1) <= cap_date]
    total  = len(months)

    if not months:
        print(f"Nothing to fetch — {start_year}-01 is after the last completed month ({cap_year}-{cap_month:02d}).")
        return

    print(f"Fetching {total} months for target={target!r}  "
          f"({months[0][0]}-{months[0][1]:02d} to {months[-1][0]}-{months[-1][1]:02d})\n")

    all_articles = {}   # date -> [record, ...]

    for i, (year, month) in enumerate(months):
        label = f"{year}-{month:02d}"

        # Load from checkpoint if available
        cached = load_checkpoint(ckpt_dir, year, month)
        if cached is not None:
            all_articles.update(cached)
            print(f"[{i+1}/{total}] {label}  loaded from checkpoint  ({sum(len(v) for v in cached.values())} articles)")
            continue

        # Fetch from NYT
        print(f"[{i+1}/{total}] {label}  fetching...", end=" ", flush=True)
        try:
            docs = fetch_archive(year, month)
        except Exception as e:
            print(f"ERROR: {e}  — saving progress and stopping.")
            save_outputs(cfg, all_articles, start_year, end_date, out_csv, metal_csv, economy_csv, keywords_csv, json_dir)
            return

        month_articles = v2_filter(docs, cfg)
        n = sum(len(v) for v in month_articles.values())
        print(f"done  ({n} articles)")

        all_articles.update(month_articles)
        save_checkpoint(ckpt_dir, year, month, month_articles)
        time.sleep(SLEEP_SECS)

    save_outputs(cfg, all_articles, start_year, end_date, out_csv, metal_csv, economy_csv, keywords_csv, json_dir)

