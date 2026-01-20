"""Engine for training a text classification model using transformers."""

import os, yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from src.data import load_ag_news, NewsDataset
from src.model import TransformerClassifier
from src.utils import set_seed, ensure_dir, compute_metrics, plot_history

def evaluate(model, loader, device, loss_fn):
    model.eval()
    total_loss = 0.0
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop("labels")
            logits = model(**batch)
            loss = loss_fn(logits, labels)
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1).cpu().tolist()
            y_pred.extend(preds)
            y_true.extend(labels.cpu().tolist())
    avg_loss = total_loss / max(1, len(loader))
    metrics = compute_metrics(y_true, y_pred)
    return avg_loss, metrics

def main(cfg_path="configs/default.yaml"):
    cfg = yaml.safe_load(open(cfg_path, "r", encoding="utf-8"))

    cfg["seed"] = int(cfg["seed"])
    cfg["max_len"] = int(cfg["max_len"])
    cfg["batch_size"] = int(cfg["batch_size"])
    cfg["epochs"] = int(cfg["epochs"])
    cfg["num_workers"] = int(cfg["num_workers"])

    cfg["lr"] = float(str(cfg["lr"]).split()[0])
    cfg["weight_decay"] = float(cfg["weight_decay"])
    cfg["warmup_ratio"] = float(cfg["warmup_ratio"])

    set_seed(cfg["seed"])

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cpu":
        torch.set_num_threads(8)

    out_dir = cfg["paths"]["out_dir"]
    ensure_dir(out_dir)

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])

    train_df = load_ag_news(cfg["paths"]["train_csv"])
    test_df  = load_ag_news(cfg["paths"]["test_csv"])

    train_df = train_df.sample(10000, random_state=42).reset_index(drop=True)
    test_df  = test_df.sample(3000, random_state=42).reset_index(drop=True)

    train_ds = NewsDataset(train_df, tokenizer, cfg["max_len"], cfg["use_title_and_desc"])
    val_ds   = NewsDataset(test_df,  tokenizer, cfg["max_len"], cfg["use_title_and_desc"])

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=cfg["num_workers"])
    val_loader   = DataLoader(val_ds,   batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"])

    model = TransformerClassifier(cfg["model_name"], num_labels=4).to(device)
    loss_fn = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])

    total_steps = cfg["epochs"] * len(train_loader)
    warmup_steps = int(cfg["warmup_ratio"] * total_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_f1": []}
    best_f1 = -1.0
    best_path = os.path.join(out_dir, "best.pt")

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        running = 0.0

        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{cfg['epochs']}")
        for batch in pbar:
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop("labels")

            optimizer.zero_grad(set_to_none=True)
            logits = model(**batch)
            loss = loss_fn(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            running += loss.item()
            pbar.set_postfix(loss=loss.item())

        train_loss = running / max(1, len(train_loader))
        val_loss, metrics = evaluate(model, val_loader, device, loss_fn)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(metrics["acc"])
        history["val_f1"].append(metrics["f1_macro"])

        print(f"\nEpoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
              f"acc={metrics['acc']:.4f} f1_macro={metrics['f1_macro']:.4f}")
        print(metrics["report"])

        if metrics["f1_macro"] > best_f1:
            best_f1 = metrics["f1_macro"]
            torch.save({
                "model_name": cfg["model_name"],
                "state_dict": model.state_dict(),
                "tokenizer_name": cfg["model_name"],
                "max_len": cfg["max_len"],
                "use_title_and_desc": cfg["use_title_and_desc"],
            }, best_path)
            print(f"Saved best checkpoint to {best_path}")

    plot_history(history, os.path.join(out_dir, "loss_curve.png"))
    print("Done.")

if __name__ == "__main__":
    main()
