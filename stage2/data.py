# stage2/data.py
import json
import logging
from typing import Dict, List

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

class InstructionDataset(Dataset):
    def __init__(self, file_path: str, tokenizer: AutoTokenizer, max_length: int = 1200):
        """Initialize the dataset by loading .jsonl file and tokenizer."""
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data: List[Dict] = []

        logger.info(f"Loading instructions from {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():  # Skip empty lines
                    self.data.append(json.loads(line))
        logger.info(f"Loaded {len(self.data)} instructions")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Return tokenized input, attention mask, labels, and metadata for a single item."""
        item = self.data[idx]
        input_text = item["input"]
        output_text = item["output"]

        # Tokenize input and output separately
        input_encoding = self.tokenizer(
            input_text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        output_encoding = self.tokenizer(
            output_text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        # Prepare labels: Use output input_ids, mask padding with -100
        labels = output_encoding["input_ids"].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_encoding["input_ids"].squeeze(0),
            "attention_mask": input_encoding["attention_mask"].squeeze(0),
            "labels": labels.squeeze(0),
            "task": item["task"],
            "label": item["label"],
        }