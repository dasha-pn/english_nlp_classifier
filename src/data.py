"""Viualization of training metrics and model predictions."""

import pandas as pd
import torch
from torch.utils.data import Dataset

LABEL_COL = 0
TITLE_COL = 1
DESC_COL = 2

def load_ag_news(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, header=None)
    if str(df.iloc[0, 0]).lower().strip() == "class index":
        df = df.iloc[1:].reset_index(drop=True)
    df[LABEL_COL] = df[LABEL_COL].astype(int) - 1
    df[TITLE_COL] = df[TITLE_COL].astype(str)
    df[DESC_COL] = df[DESC_COL].astype(str)
    return df

class NewsDataset(Dataset):
    def __init__(self, df, tokenizer, max_len: int, use_title_and_desc: bool = True):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.use_title_and_desc = use_title_and_desc

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        label = int(row[LABEL_COL])
        title = row[TITLE_COL]
        desc = row[DESC_COL]
        text = f"{title} {desc}" if self.use_title_and_desc else title

        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(label, dtype=torch.long)
        return item
