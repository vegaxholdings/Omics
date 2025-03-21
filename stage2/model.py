import os

import torch
from transformers import TrainingArguments, Trainer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

class SiRNAModel:
    def __init__(self, model_name_or_path="meta-llama/Meta-Llama-3.1-8B"):
        self.model_name_or_path = model_name_or_path
        self.tokenizer = None
        self.model = None
    
    def load_model(self, use_4bit=True, device_map="auto"):
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"  # 왼쪽 패딩 설정 (디코더 모델에 적합)
        
        # Quantization config
        quant_config = None
        if use_4bit:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=False,
            )
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            quantization_config=quant_config,
            device_map=device_map,
            trust_remote_code=True,
            use_cache=False
        )
        
        if use_4bit:
            self.model = prepare_model_for_kbit_training(self.model)
        
        return self.model, self.tokenizer
    
    def get_lora_config(self, r=16, lora_alpha=32, lora_dropout=0.05):
        # LoRA configuration for Llama models
        # 논문에 맞춰 Wq, Wv, W1, W2, W3에 해당하는 모듈만 타겟으로 설정
        target_modules = [
            "q_proj", "v_proj",  # Wq, Wv
            "gate_proj", "up_proj", "down_proj"  # W1, W2, W3
        ]
        
        lora_config = LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        return lora_config
    
    def prepare_for_training(self, r=16, lora_alpha=32, lora_dropout=0.05):
        # Apply LoRA
        lora_config = self.get_lora_config(r, lora_alpha, lora_dropout)
        self.model = get_peft_model(self.model, lora_config)
        
        # RMSNorm 파라미터를 학습 가능하게 설정 (논문에서 언급한 "Mixed" 방식 구현)
        for name, param in self.model.named_parameters():
            if "norm" in name:
                param.requires_grad = True
        
        # LoRA+ 구현: 논문에서 언급한 "trainable parameters" (LoRA, RMSNorm, wq, wv) 설정
        # 학습률 스케일링을 위한 파라미터 그룹 설정
        params_a = []  # 기본 파라미터 (기본 학습률)
        params_b = []  # LoRA B 파라미터 (16배 학습률)
        params_norm = []  # RMSNorm 파라미터 (기본 학습률)
        params_wq_wv = []  # Wq, Wv 파라미터 (기본 학습률)
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if ".lora_B." in name:
                    params_b.append(param)  # LoRA B 파라미터
                elif "norm" in name.lower():
                    params_norm.append(param)  # RMSNorm 파라미터
                elif any(key in name for key in ["q_proj", "v_proj"]):
                    params_wq_wv.append(param)  # Wq, Wv 관련 파라미터
                else:
                    params_a.append(param)  # 기타 파라미터
        
        # 모델에 파라미터 그룹 저장 (train 메서드에서 사용)
        self.param_groups = [
            {"params": params_a, "lr": 1.0},  # 기본 스케일 (1.0)
            {"params": params_b, "lr": 16.0},  # LoRA+ 스케일 (16.0) - 논문에 맞춤
            {"params": params_norm, "lr": 1.0},  # RMSNorm 스케일 (1.0)
            {"params": params_wq_wv, "lr": 1.0}  # Wq, Wv 파라미터 스케일 (1.0)
        ]
        
        return self.model
    
    def train(self, train_dataset, output_dir, batch_size=4, gradient_accumulation_steps=8, 
              num_train_epochs=3, learning_rate=2e-4, max_steps=-1):
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            lr_scheduler_type="cosine",
            warmup_ratio=0.1,
            max_steps=max_steps,
            num_train_epochs=num_train_epochs,
            save_strategy="epoch",
            logging_steps=10,
            report_to="tensorboard",
            gradient_checkpointing=True,
            bf16=True,
            tf32=True,
            remove_unused_columns=False,
        )
        
        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            tokenizer=self.tokenizer,
            data_collator=lambda data: {
                'input_ids': torch.stack([f['input_ids'] for f in data]),
                'attention_mask': torch.stack([f['attention_mask'] for f in data]),
                'labels': torch.stack([f['labels'] for f in data]),
            }
        )
        
        # 커스텀 옵티마이저 설정 (LoRA+ 스케일링 구현)
        if hasattr(self, 'param_groups'):
            # LoRA+ 스케일링 적용
            optimizer = torch.optim.AdamW(
                self.param_groups,
                lr=learning_rate,
                betas=(0.9, 0.999),
                eps=1e-8,
                weight_decay=0.01
            )
            trainer.optimizer = optimizer
        
        # Train
        trainer.train()
        
        # Save model
        self.save_model(output_dir)
        
    def save_model(self, output_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
    
    def load_trained_model(self, model_path, device_map="auto"):
        # Load tokenizer and model from saved path
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.tokenizer.padding_side = "left"
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device_map,
            trust_remote_code=True
        )
        
        return self.model, self.tokenizer 