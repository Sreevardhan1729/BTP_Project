from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Dict
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed,
)


from ...utils.io import get_logger as get_logger_pkg  # fallback if above path differs

# Ensure logger resolves regardless of import path
try:
    logger = get_logger("fdof.llm")
except Exception:
    logger = get_logger_pkg("fdof.llm")

class TxtDataset(Dataset):
    def __init__(self, texts: List[str], tokenizer, max_length: int):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        # For causal LM SFT we train on next-token prediction over whole string
        item["labels"] = item["input_ids"].clone()
        return item

def save_lines(path: str | Path, lines: List[str]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line.replace("\r\n", "\n").replace("\r", "\n") + "\n")

@dataclass
class TrainCfg:
    model_name_or_path: str
    max_length: int
    seed: int
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    warmup_ratio: float
    gradient_accumulation_steps: int
    logging_steps: int
    save_total_limit: int
    out_model_dir: str

def load_model_tokenizer(name_or_path: str):
    tok = AutoTokenizer.from_pretrained(name_or_path, use_fast=True)
    # GPT2-like models may not have pad_token
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(name_or_path)
    model.config.pad_token_id = tok.pad_token_id
    return model, tok

def train_causal_lm(
    train_texts: List[str],
    val_texts: List[str],
    cfg: TrainCfg,
):
    set_seed(cfg.seed)
    model, tok = load_model_tokenizer(cfg.model_name_or_path)

    ds_tr = TxtDataset(train_texts, tok, cfg.max_length)
    ds_va = TxtDataset(val_texts, tok, cfg.max_length)

    collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=False)

    out_dir = Path(cfg.out_model_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    args = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        logging_steps=cfg.logging_steps,
        eval_steps=1709,
        save_steps=1709,
        save_total_limit=cfg.save_total_limit,
        report_to=[],  # no wandb required
        fp16=False,
        bf16=False,
        dataloader_pin_memory=False,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds_tr,
        eval_dataset=ds_va,
        data_collator=collator,
        tokenizer=tok,
    )

    trainer.train()
    # Save final model
    trainer.save_model(str(out_dir))
    tok.save_pretrained(str(out_dir))
    return trainer.model, tok

def predict_labels_greedy(
    model,
    tokenizer,
    prompts: List[str],
    max_new_tokens: int = 1,
) -> List[int]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    res = []
    with torch.no_grad():
        for p in prompts:
            enc = tokenizer(
                p, return_tensors="pt", truncation=True, max_length=tokenizer.model_max_length
            )
            input_ids = enc["input_ids"].to(device)
            attn = enc["attention_mask"].to(device)
            gen = model.generate(
                input_ids=input_ids,
                attention_mask=attn,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            new_tokens = gen[0, input_ids.shape[1]:]
            text = tokenizer.decode(new_tokens, skip_special_tokens=True)
            # Parse first digit 0/1
            lab = None
            for ch in text.strip():
                if ch in ("0", "1"):
                    lab = int(ch)
                    break
            if lab is None:
                # Fallback: compare logits of '0' vs '1' at next position
                logits = model(input_ids=input_ids, attention_mask=attn).logits
                next_logits = logits[0, -1, :]
                tok_0 = tokenizer.encode("0", add_special_tokens=False)[0]
                tok_1 = tokenizer.encode("1", add_special_tokens=False)[0]
                lab = int(next_logits[tok_1] > next_logits[tok_0])
            res.append(lab)
    return res
