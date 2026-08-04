"""
Step 6: run a transformer over the fetched headlines.

Produces two files in the shared per-topic news cache: ``news_flat.csv`` (one
row per article) and ``news_enriched.csv`` (one row per calendar day, the
per-day mean of the articles' embeddings and sentiment probabilities).

Class ordering
--------------
Sentiment models do not agree on class order — FinBERT is
``{0: positive, 1: negative, 2: neutral}`` while FinancialBERT is
``{0: negative, 1: neutral, 2: positive}``. Probabilities are therefore mapped
into the ``prob_*`` columns **by label name**, read from the model's own
``config.id2label``, never by index. Assigning them positionally would silently
write the negative probability into ``prob_positive`` for any model whose order
differs from the one the code was written against.
"""

import ast
import os

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

from constants import PROB_COLS, PROB_LABELS
from scripts.load_news.load_news import TARGETS
from scripts.paths import dataset_config, news_dir, news_enriched_path, news_flat_path

# Headlines per transformer forward pass. Caps peak memory so that a single
# NYT bulk-backfill date (~1,950 articles) cannot take the process down; see
# analyze(). Purely a memory/throughput knob — results do not depend on it.
BATCH_SIZE = 64

# Days between resume-checkpoint flushes in run_flat_analysis. Each flush
# rewrites the whole partial file, so this trades restart cost against write
# cost — at the metals corpus's ~8.9 articles/day, 100 days is a few seconds of
# rewriting to cap a crash at ~100 days of lost inference.
CHECKPOINT_EVERY_DAYS = 100

MODELS = {
    # FinBERT: id2label {0: positive, 1: negative, 2: neutral}, 768-dim
    "finbert": {
        "name": "ProsusAI/finbert",
        "loader": AutoModelForSequenceClassification,
        "has_sentiment": True,
    },
    # FinancialBERT: id2label {0: negative, 1: neutral, 2: positive}, 768-dim.
    # A different order from FinBERT's — handled by name-based mapping below.
    "financialbert": {
        "name": "ahmedrachid/FinancialBERT-Sentiment-Analysis",
        "loader": AutoModelForSequenceClassification,
        "has_sentiment": True,
    },
    # MiniLM: embeddings only, no classification head, 384-dim.
    "minilm": {
        "name": "sentence-transformers/all-MiniLM-L6-v2",
        "loader": AutoModel,
        "has_sentiment": False,
    },
}


def get_model(model_key: str):
    """
    Load the tokenizer/model pair for a model key.

    Parameters
    ----------
    model_key : str
        One of the keys of ``MODELS``.

    Returns
    -------
    tuple
        ``(tokenizer, model, has_sentiment)`` — ``has_sentiment`` is False for
        embedding-only models, whose outputs carry no ``prob_*``/``label``.

    Raises
    ------
    KeyError
        If ``model_key`` is not a known model, listing the valid keys.
    """
    try:
        spec = MODELS[model_key]
    except KeyError:
        raise KeyError(f"unknown model {model_key!r} - valid models are {sorted(MODELS)}") from None
    tokenizer = AutoTokenizer.from_pretrained(spec["name"])
    model = spec["loader"].from_pretrained(spec["name"])
    return tokenizer, model, spec["has_sentiment"]


def label_index(model) -> dict[str, int]:
    """
    Map each sentiment label name to its column index in the model's logits.

    Parameters
    ----------
    model : transformers.PreTrainedModel
        A loaded sequence-classification model.

    Returns
    -------
    dict[str, int]
        e.g. ``{"positive": 0, "negative": 1, "neutral": 2}`` for FinBERT and
        ``{"negative": 0, "neutral": 1, "positive": 2}`` for FinancialBERT.
        Label names are lowercased.

    Raises
    ------
    ValueError
        If the model's classes do not match the three expected labels. The
        downstream schema (``PROB_COLS``) is fixed at
        positive/negative/neutral, so a model with different classes needs an
        explicit decision rather than a silent mismatch.
    """
    mapping = {name.lower(): idx for idx, name in model.config.id2label.items()}
    if set(mapping) != set(PROB_LABELS):
        raise ValueError(
            f"model exposes classes {sorted(mapping)} but the pipeline schema expects "
            f"{sorted(PROB_LABELS)} (PROB_COLS in constants.py) - add an explicit mapping "
            f"before using this model"
        )
    return mapping


def analyze(texts: list, tokenizer, model, has_sentiment: bool,
            batch_size: int = BATCH_SIZE) -> dict:
    """
    Run a tokenizer/model pair over headlines, in fixed-size batches.

    The caller passes one day's headlines, and per-day volume is not bounded by
    anything: the NYT archive carries bulk-backfill dates where thousands of
    articles share a ``pub_date`` (2021-01-27 and 2023-03-28 hold ~1,950 each,
    against a corpus mean of 8.9). Encoding a day as a single batch made peak
    memory a function of that volume — 1,938 sequences at 512 tokens is roughly
    24 GB of attention — and killed the process with SIGSEGV partway through a
    run, losing every day processed up to that point.

    Batching in chunks of `batch_size` makes peak memory constant in the day's
    size, so no day can flood it. Nothing is discarded: every headline is still
    encoded, the chunks are concatenated, and the returned tensors are identical
    to what a single oversized batch would have produced. Padding is per chunk
    rather than per day, so a day holding one long article no longer pads every
    other headline out to that length.

    Parameters
    ----------
    texts : list of str
        Headlines to encode. Processed in slices of `batch_size`.
    tokenizer : transformers.PreTrainedTokenizer
        Tokenizer paired with ``model``.
    model : transformers.PreTrainedModel
        Loaded model.
    has_sentiment : bool
        Whether ``model`` has a classification head.
    batch_size : int, default BATCH_SIZE
        Headlines per forward pass. Bounds peak memory; does not affect results.

    Returns
    -------
    dict
        Always carries ``"embedding"``, the ``(N, D)`` CLS vectors. With a
        classification head it also carries ``"probs"`` ``(N, 3)`` in the
        model's own class order, ``"label"`` (one name per headline) and
        ``"label_index"`` mapping label name to its column in ``probs`` — the
        caller must use that mapping rather than assuming an order.
    """
    if not texts:
        raise ValueError("analyze() received no texts")

    embedding_chunks, prob_chunks = [], []
    for start in range(0, len(texts), batch_size):
        chunk = texts[start:start + batch_size]
        inputs = tokenizer(chunk, return_tensors="pt", padding=True,
                           truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        embedding_chunks.append(outputs.hidden_states[-1][:, 0, :])
        if has_sentiment:
            prob_chunks.append(torch.nn.functional.softmax(outputs.logits, dim=-1))

    embedding = torch.cat(embedding_chunks, dim=0)

    if not has_sentiment:
        return {"embedding": embedding}

    idx_to_name = {idx: name.lower() for idx, name in model.config.id2label.items()}
    probs = torch.cat(prob_chunks, dim=0)        # (N, 3), model's own class order
    labels = [idx_to_name[i] for i in probs.argmax(dim=-1).tolist()]

    return {
        "embedding": embedding,
        "probs": probs,
        "label": labels,
        "label_index": label_index(model),
    }


def _write_articles(rows: list, path: str) -> pd.DataFrame:
    """
    Write per-article rows to CSV, stringifying the embedding column.

    Shared by the resume checkpoint and the final output so both files have
    exactly one format. ``str()`` on an already-stringified embedding is a no-op,
    which is what lets a reloaded checkpoint be written back out unchanged.

    Parameters
    ----------
    rows : list of dict
        Per-article records; ``embedding`` may hold a list or its string form.
    path : str
        Destination CSV. Parent directories are created.

    Returns
    -------
    pd.DataFrame
        The frame that was written.
    """
    out = pd.DataFrame(rows)
    out["embedding"] = out["embedding"].map(str)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    out.to_csv(path, index=False)
    return out


def run_flat_analysis(orig_file: str, dest_file: str, model_key: str) -> pd.DataFrame:
    """
    Encode every article and write one row per article.

    Each day's headlines are encoded in batches of ``BATCH_SIZE``, but every
    article keeps its own independent outputs — no aggregation happens here.

    The pass is resumable. Transformer inference over a full corpus runs for
    tens of minutes to hours, and this used to write only after the loop
    finished, so any failure discarded everything — a crash 4,778 days into a
    6,754-day run cost 2.5 hours and left no output at all. Progress is now
    flushed to ``<dest_file>.partial`` every ``CHECKPOINT_EVERY_DAYS`` days, and
    a rerun reloads it and skips the days already encoded. The partial file is
    removed once the real output is written, so its presence always means "an
    earlier attempt did not finish".

    Parameters
    ----------
    orig_file : str
        Raw news CSV (columns ``Date``, then one column per headline).
    dest_file : str
        Path the flat per-article CSV is written to. The resume checkpoint lives
        alongside it as ``<dest_file>.partial``.
    model_key : str
        A key of ``MODELS``.

    Returns
    -------
    pd.DataFrame
        Columns ``Date``, ``headline``, ``embedding``, plus (for models with a
        classification head) ``label`` and the ``PROB_COLS``. Probability
        columns are filled by label name via the model's ``id2label``, so they
        hold what their names say regardless of the model's class order.
    """
    df_news = pd.read_csv(orig_file)
    date_col = df_news["Date"]
    news_ = df_news.drop("Date", axis=1)

    tokenizer, model, has_sentiment = get_model(model_key)

    checkpoint_file = f"{dest_file}.partial"
    rows, done_days = [], set()
    if os.path.exists(checkpoint_file):
        previous = pd.read_csv(checkpoint_file)
        rows = previous.to_dict("records")
        done_days = set(previous["Date"].astype(str))
        print(f"Resuming from {checkpoint_file}: "
              f"{len(done_days)} days / {len(rows)} articles already encoded")

    encoded_days = 0
    for idx, row in news_.iterrows():
        day = date_col[idx]
        if str(day) in done_days:
            continue
        headlines = row.dropna().tolist()
        if not headlines:
            continue

        day_results = analyze(headlines, tokenizer, model, has_sentiment)
        for i, headline in enumerate(headlines):
            article_row = {"Date": day, "headline": headline}
            article_row["embedding"] = day_results["embedding"][i].tolist()
            if has_sentiment:
                article_row["label"] = day_results["label"][i]
                # By name, not by position: see the module docstring.
                for name, col_idx in day_results["label_index"].items():
                    article_row[f"prob_{name}"] = day_results["probs"][i, col_idx].item()
            rows.append(article_row)

        print(f"Processed {day}  ({len(headlines)} articles)")

        encoded_days += 1
        if encoded_days % CHECKPOINT_EVERY_DAYS == 0:
            _write_articles(rows, checkpoint_file)
            print(f"  checkpoint: {len(rows)} articles written, through {day}")

    out = _write_articles(rows, dest_file)
    # Only now is the real output complete — dropping the checkpoint any earlier
    # would leave a failure with neither file.
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
    return out


def run_day_aggregate(orig_file: str, dest_file: str) -> pd.DataFrame:
    """
    Collapse the flat per-article file to one row per calendar day.

    Embeddings are averaged across the day's articles. When probability columns
    are present they are averaged too, and the day's label is the argmax of the
    averaged distribution. Aggregation is vectorised via ``groupby().mean()``.

    Parameters
    ----------
    orig_file : str
        Flat per-article CSV, as written by :func:`run_flat_analysis`.
    dest_file : str
        Path the per-day CSV is written to.

    Returns
    -------
    pd.DataFrame
        Columns ``Date``, ``embedding``, plus ``label`` and the ``PROB_COLS``
        when the source had them.

    Notes
    -----
    Isolated single-day gaps are **not** filled here — that happens in the
    merge step (``dataset_builder.fill_news_isolated_gaps``), which reindexes
    onto a full calendar first. Filling at this stage is impossible: the frame
    below only contains days that have articles, so there are no gap rows to
    find.
    """
    flat = pd.read_csv(orig_file)
    has_sentiment = "prob_positive" in flat.columns

    # parse all embeddings at once into a (N, D) matrix, then avg per day
    emb_matrix = np.stack(flat["embedding"].map(ast.literal_eval))
    emb_df = pd.DataFrame(emb_matrix, index=flat["Date"])
    emb_avg = emb_df.groupby("Date").mean()                          # (days, D)
    out = pd.DataFrame({"Date": emb_avg.index, "embedding": emb_avg.values.tolist()})

    if has_sentiment:
        prob_avg = flat.groupby("Date")[PROB_COLS].mean().reset_index()
        # argmax runs over PROB_COLS order, so the label list must be in that
        # same order — both come from constants.py and are defined together.
        winner = prob_avg[PROB_COLS].values.argmax(axis=1)
        prob_avg["label"] = [PROB_LABELS[i] for i in winner]
        out = out.merge(prob_avg, on="Date")

    out["embedding"] = out["embedding"].map(str)
    os.makedirs(os.path.dirname(dest_file), exist_ok=True)
    out.to_csv(dest_file, index=False)
    return out


def pipe_line(base_dir: str, dataset: str, run_flat: int = 1, model: str = "finbert") -> None:
    """
    Run the NLP step for a dataset's news topic.

    Reads and writes the shared per-topic news cache, so two datasets on the
    same topic reuse one set of embeddings rather than recomputing them.

    Parameters
    ----------
    base_dir : str
        Root directory holding the shared ``news/`` cache.
    dataset : str
        Dataset key; its ``news_topic`` selects which cache is processed.
    run_flat : int, optional
        When truthy, re-run the per-article pass. When falsy, reuse an existing
        ``news_flat.csv`` and only redo the per-day aggregation.
    model : str, optional
        A key of ``MODELS``.

    Returns
    -------
    None

    Raises
    ------
    FileNotFoundError
        If the raw news for this topic has not been fetched, or if ``run_flat``
        is falsy and no existing flat file is present.
    """
    topic = dataset_config(dataset)["news_topic"]
    news_path = os.path.join(news_dir(base_dir, topic), TARGETS[topic]["out_name"])
    flat_path = news_flat_path(base_dir, dataset)
    enriched_path = news_enriched_path(base_dir, dataset)

    if run_flat:
        if not os.path.isfile(news_path):
            raise FileNotFoundError(
                f"no fetched news for topic {topic!r} at {news_path} - run the news step first"
            )
        run_flat_analysis(news_path, flat_path, model)
    elif not os.path.isfile(flat_path):
        raise FileNotFoundError(
            f"--no-run-flat was given but no existing flat file at {flat_path} - "
            f"run once with the flat pass enabled first"
        )

    run_day_aggregate(flat_path, enriched_path)
