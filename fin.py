from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
model_setts = model.config

# {0: 'positive', 1: 'negative', 2: 'neutral'}
label_map = model_setts.id2label


def analyze(text: str) -> dict:
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    probs = torch.nn.functional.softmax(outputs.logits, dim=-1).squeeze()

    #label_id = probs.argmax().item()
    #print(label_id)

    embedding = outputs.hidden_states[-1][:, 0, :].squeeze()  # CLS token, shape (768,)

    return {
        "embedding": embedding,
        "probs": probs
    }


if __name__ == "__main__":
    result = analyze("The company reported big debt.")
    probs_dict = {label_map[idx]: prob.item() for idx, prob in enumerate(result["probs"].numpy())}
    print("Probs:    ", probs_dict)
    print("Embedding:", result["embedding"].shape)
