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
        "probs": probs
    }


import os

if __name__ == "__main__":

    NEWS_PATH = os.path.join(os.getcwd(), "news_", "news_paper2.csv")

    df_news = pd.read_csv(NEWS_PATH)

    date = df_news["Date"]
    news_ = df_news.drop("Date", axis=1)

    for _, row in news_.iterrows():
        headlines = row.dropna().tolist()
        result = analyze(headlines)

        probs_dict = [
            {label_map[idx]: prob.item() for idx, prob in enumerate(row_probs)}
            for row_probs in result["probs"]
        ]
        
        print("Date:     ", date[_])
        print("Probs:    ", probs_dict)
        print("Embedding:", result["embedding"].shape)
