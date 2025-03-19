import random
import logging
from typing import Dict, List

import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

class SequenceDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=1200, use_packing=True, tokens_per_batch=2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.sequences = []
        self.use_packing = use_packing
        self.tokens_per_batch = tokens_per_batch
        
        logger.info(f"Loading sequences from {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                sequence = line.strip()
                if sequence:  # Skip empty lines
                    self.sequences.append(sequence)
        
        logger.info(f"Loaded {len(self.sequences)} sequences")
        
        if use_packing:
            logger.info("Using sequence packing strategy")
            self.packed_examples = self._create_packed_examples()
            logger.info(f"Created {len(self.packed_examples)} packed examples")
    
    def _create_packed_examples(self) -> List[Dict[str, torch.Tensor]]:
        """
        패킹 전략을 사용하여 시퀀스를 전처리합니다.
        여러 시퀀스를 단일 예제로 패킹하여 패딩을 최소화합니다.
        """
        all_tokenized = []
        current_tokens = 0
        current_batch = []
        
        # 모든 시퀀스를 무작위로 섞습니다
        random.shuffle(self.sequences)
        
        for sequence in self.sequences:
            # 시퀀스 토큰화
            tokenized = self.tokenizer(
                sequence,
                truncation=True,
                max_length=self.max_length,
                return_tensors=None,  # 텐서가 아닌 목록 반환
            )
            
            # EOS 토큰 추가
            tokenized["input_ids"].append(self.tokenizer.eos_token_id)
            tokenized["attention_mask"].append(1)
            
            seq_length = len(tokenized["input_ids"])
            
            # 현재 배치에 추가했을 때 토큰 수가 한계를 초과하는지 확인
            if current_tokens + seq_length > self.tokens_per_batch and current_batch:
                # 현재 배치를 패킹하고 저장
                all_tokenized.append(self._pack_batch(current_batch))
                current_batch = [tokenized]
                current_tokens = seq_length
            else:
                # 현재 배치에 시퀀스 추가
                current_batch.append(tokenized)
                current_tokens += seq_length
        
        # 마지막 배치 처리
        if current_batch:
            all_tokenized.append(self._pack_batch(current_batch))
        
        return all_tokenized
    
    def _pack_batch(self, batch_sequences):
        """
        여러 시퀀스를 하나의 예제로 패킹합니다.
        """
        # 모든 입력 ID와 어텐션 마스크 연결
        input_ids = []
        attention_mask = []
        
        for sequence in batch_sequences:
            input_ids.extend(sequence["input_ids"])
            attention_mask.extend(sequence["attention_mask"])
        
        # 최대 길이에 맞춰 패딩
        if len(input_ids) < self.max_length:
            padding_length = self.max_length - len(input_ids)
            input_ids.extend([self.tokenizer.pad_token_id] * padding_length)
            attention_mask.extend([0] * padding_length)
        else:
            # 최대 길이에 맞게 잘라내기
            input_ids = input_ids[:self.max_length]
            attention_mask = attention_mask[:self.max_length]
        
        # 텐서로 변환
        result = {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask),
        }
        
        # 레이블 추가 (언어 모델링용)
        result["labels"] = result["input_ids"].clone()
        
        return result
    
    def __len__(self):
        if self.use_packing:
            return len(self.packed_examples)
        return len(self.sequences)
    
    def __getitem__(self, idx):
        if self.use_packing:
            return self.packed_examples[idx]
        
        sequence = self.sequences[idx]
        
        # 시퀀스 토큰화
        encoding = self.tokenizer(
            sequence,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        
        # 배치 차원 제거
        encoding = {k: v.squeeze(0) for k, v in encoding.items()}
        
        # 언어 모델링을 위한 레이블 생성
        encoding["labels"] = encoding["input_ids"].clone()
        
        return encoding