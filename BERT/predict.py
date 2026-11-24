import sys
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_DIR = Path(__file__).resolve().parent / "results" / "imdb_roberta" / "final"

# Use the saved tokenizer if it exists, else the same one from the hub
tok_src = MODEL_DIR if (MODEL_DIR / "tokenizer.json").exists() else "roberta-base"
tok = AutoTokenizer.from_pretrained(tok_src)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(device).eval()

# Review from the command line, or from a prompt
review = " ".join(sys.argv[1:]) or input("Paste your review:\n> ")

inputs = tok(review, truncation=True, max_length=512, return_tensors="pt").to(device)
with torch.no_grad():
    logits = model(**inputs).logits
probs = logits.softmax(-1)[0].cpu()

label = "positive" if probs[1] > probs[0] else "negative"
n_tokens = len(tok(review)["input_ids"])

print(f"\nPrediction: {label}")
print(f"  negative {probs[0]:.3f} | positive {probs[1]:.3f}")
if n_tokens > 512:
    print(f"  Note: your review has {n_tokens} tokens, so only the first 512 were used.")