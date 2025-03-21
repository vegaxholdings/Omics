# stage2/data.py
import json
import logging
from typing import Dict, List
import re

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
        
        # Format as instruction-tuning data for Llama 3.1
        formatted_text = f"<s>[INST] {input_text} [/INST] {output_text}</s>"
        
        # Tokenize the formatted text
        encodings = self.tokenizer(
            formatted_text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        
        # 입력 ID와 어텐션 마스크 가져오기
        input_ids = encodings["input_ids"].squeeze(0)
        attention_mask = encodings["attention_mask"].squeeze(0)
        
        # 손실 계산을 위해 레이블 준비 (응답 부분만 계산)
        labels = input_ids.clone()
        
        # [/INST] 토큰 위치 찾기
        inst_token_pos = (input_ids == self.tokenizer.encode("[/INST]", add_special_tokens=False)[-1]).nonzero()
        if len(inst_token_pos) > 0:
            # [/INST] 토큰까지 모든 토큰을 -100으로 마스킹 (손실 계산에서 제외)
            labels[:inst_token_pos[-1] + 1] = -100
            
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "task": item["task"],
            "label": item["label"],
        }

# 평가 관련 유틸리티 함수
def extract_value_from_response(response):
    """모델 응답에서 수치 값 추출"""
    # "percentage of XX.XX"와 같은 패턴이나 소수점이 있는 숫자 찾기
    pattern = r'(\d+\.\d+)'
    matches = re.findall(pattern, response)
    
    if matches:
        try:
            return float(matches[0])
        except ValueError:
            pass
    
    # 대안: 아무 숫자나 찾기
    pattern = r'(\d+)'
    matches = re.findall(pattern, response)
    
    if matches:
        try:
            return float(matches[0])
        except ValueError:
            pass
    
    return None

def calculate_mae(predictions, ground_truth):
    """평균 절대 오차(MAE) 계산"""
    if len(predictions) != len(ground_truth):
        raise ValueError(f"길이 불일치: predictions ({len(predictions)}) != ground_truth ({len(ground_truth)})")
    
    # None 값 처리
    cleaned_predictions = []
    cleaned_ground_truth = []
    
    for p, t in zip(predictions, ground_truth):
        if p is not None:
            cleaned_predictions.append(p)
            cleaned_ground_truth.append(t)
    
    if len(cleaned_predictions) == 0:
        return float('inf')  # 유효한 예측이 없으면 무한대 반환
    
    # 평균 절대 오차 계산
    mae = sum(abs(p - t) for p, t in zip(cleaned_predictions, cleaned_ground_truth)) / len(cleaned_predictions)
    return mae