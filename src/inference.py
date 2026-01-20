"""Inference script for a text classification model."""

import argparse
import torch
from transformers import AutoTokenizer
from src.model import TransformerClassifier

LABELS = ["World", "Sports", "Business", "Sci/Tech"]

def load_checkpoint(path, device):
    ckpt = torch.load(path, map_location=device)
    model = TransformerClassifier(ckpt["model_name"], num_labels=4).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    tok = AutoTokenizer.from_pretrained(ckpt["tokenizer_name"])
    return model, tok, ckpt

@torch.no_grad()
def predict(text, model, tokenizer, max_len, device):
    enc = tokenizer(text, truncation=True, padding="max_length", max_length=max_len, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}
    logits = model(**enc)
    probs = torch.softmax(logits, dim=1).squeeze(0)
    idx = int(torch.argmax(probs).item())
    return LABELS[idx], float(probs[idx].item())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="models/checkpoints/best.pt")
    ap.add_argument("--text", type=str, default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok, ckpt = load_checkpoint(args.ckpt, device)

    text = args.text or "Apple unveils a new AI chip for laptops."
    label, p = predict(text, model, tok, ckpt["max_len"], device)
    print(f"Text: {text}\nPred: {label} (p={p:.4f})")

if __name__ == "__main__":
    main()
