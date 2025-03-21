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
        self.tokenizer.padding_side = "left" # 왼쪽 패딩 설정 추가 (디코더 모델에 적합)
        
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
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
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