from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
model_setts = model.config

# {0: 'positive', 1: 'negative', 2: 'neutral'}
label_map = model_setts.id2label


def analyze(texts: list) -> dict:
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)  # (N, 3)
    embedding = outputs.hidden_states[-1][:, 0, :]               # (N, 768)

    return {
        "embedding": embedding,
        "probs": probs,
    }


def aggregate_day(result: dict) -> dict:
    """
    Aggregates per-headline outputs into a single representation for the day.

    Args:
        result: output of analyze() with keys "embedding" (N, 768) and "probs" (N, 3)
    Returns:
        dict with:
            "embedding": mean embedding across headlines, shape (768,)
            "probs":     mean sentiment probs across headlines, shape (3,)
            "label":     dominant sentiment label for the day
    """
    day_embedding = result["embedding"].mean(dim=0)   # (768,)
    day_probs     = result["probs"].mean(dim=0)        # (3,)
    day_label     = label_map[day_probs.argmax().item()]
    return {
        "embedding":     day_embedding,
        "label":         day_label,
        "prob_positive": day_probs[0].item(),
        "prob_negative": day_probs[1].item(),
        "prob_neutral":  day_probs[2].item(),
    }


import os
from config import ENRICHED_DATASETS_DIR, NEWS_DIR , ORIGINAL_DATASETS_DIR


def enrich_news_file(orig_file: str, dest_file: str):
    """
    Reads a raw news CSV, runs FinBERT on each day's headlines,
    aggregates per day, and saves the enriched result as a CSV.

    Each output row contains:
        Date, label, prob_positive, prob_negative, prob_neutral,
        emb_0 ... emb_767

    Args:
        orig_file: path to raw news CSV (columns: Date, headline_1, headline_2, ...)
        dest_file: path to save enriched CSV
    """
    df_news = pd.read_csv(orig_file)
    date  = df_news["Date"]
    news_ = df_news.drop("Date", axis=1)

    os.makedirs(os.path.dirname(dest_file), exist_ok=True)
    batch_size = 32
    rows = []

    HEADER = True

    for idx, row in news_.iterrows():
        headlines = row.dropna().tolist()
        if not headlines:
            continue
        result = analyze(headlines)
        day    = aggregate_day(result)

        rows.append({
            "Date":          date[idx],
            "label":         day["label"],
            "prob_positive": day["prob_positive"],
            "prob_negative": day["prob_negative"],
            "prob_neutral":  day["prob_neutral"],
            "embedding":     str(day["embedding"].tolist()),
        })
        print(f"Processed {date[idx]}  →  {day['label']}")

        if len(rows) >= batch_size:
            print(f"writing to disk from {rows[0]['Date']} to {rows[-1]['Date']}")
            pd.DataFrame(rows).set_index("Date").to_csv(
                dest_file, mode="w" if HEADER else "a", header=HEADER, index=True
            )
            rows = []
            HEADER = False

    if rows:
        pd.DataFrame(rows).set_index("Date").to_csv(
            dest_file, mode="w" if HEADER else "a", header=HEADER, index=True
        )


if __name__ == "__main__":
    NEWS_PATH = os.path.join(ORIGINAL_DATASETS_DIR, NEWS_DIR , "news_paper2.csv")
    DEST_PATH = os.path.join(ENRICHED_DATASETS_DIR, NEWS_DIR, "news_paper2.csv")
    enrich_news_file(NEWS_PATH, DEST_PATH)
