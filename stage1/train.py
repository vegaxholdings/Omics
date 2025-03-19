import os
import torch
import logging
import time
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    Trainer, 
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.tensorboard import SummaryWriter
from .data import SequenceDataset

# 메모리 단편화 방지 환경 변수 설정
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

logger = logging.getLogger(__name__)

def train(args):
    """Stage 1: Biological sequences continued pre-training"""
    # 기본 로깅 설정
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    
    # 출력 디렉토리 설정
    output_dir = args.output_dir
    metrics_dir = output_dir.replace("models", "metrics")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    
    # TensorBoard 설정
    writer = SummaryWriter(log_dir=metrics_dir)
    
    # 시드 설정
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # 메모리 캐시 초기화
    torch.cuda.empty_cache()
    
    # GPU 정보 로깅
    if torch.cuda.is_available():
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        logger.warning("No GPU available, using CPU instead")
    
    # 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    tokenizer.pad_token = tokenizer.eos_token
    
    # 모델 로드 - 간단한 설정
    logger.info(f"Loading model from {args.model_dir}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        # Flash Attention 2 - 올바른 파라미터 사용
        attn_implementation="flash_attention_2" if torch.cuda.is_available() else None,
    )
    
    # FlashAttention-2 활성화
    if torch.cuda.is_available():
        torch.backends.cuda.enable_flash_sdp(True)
    
    # 모델 준비
    model = prepare_model_for_kbit_training(model)
    
    # LoRA 설정 - 논문 기반 (r=128, alpha=32)
    logger.info("Configuring LoRA")
    lora_config = LoraConfig(
        r=args.lora_r,  # 논문에서는 128 사용
        lora_alpha=args.lora_alpha,  # 논문에서는 32 사용
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    
    # LoRA 모델 가져오기
    model = get_peft_model(model, lora_config)
    
    # 학습 가능한 파라미터 수 계산 및 로깅
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    logger.info(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}%"
    )
    
    # LoRA+ - 논문에 따라 B 가중치에 4배 학습률 적용
    params_a = []
    params_b = []
    params_norm = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            if ".lora_B." in name:
                params_b.append(param)
            elif "norm" in name.lower():
                params_norm.append(param)
            else:
                params_a.append(param)
    
    optimizer_grouped_parameters = [
        {"params": params_a, "lr": args.learning_rate},
        {"params": params_b, "lr": args.learning_rate * 4},  # 논문: LoRA+ scaler = 4
        {"params": params_norm, "lr": args.learning_rate},
    ]
    
    # 데이터셋 로드 - 논문 기반 (max_length=1200)
    max_seq_length = min(args.max_seq_length, 1200)  # 논문에서 최대 1200 토큰 사용
    
    logger.info(f"Loading dataset from {args.train_file}")
    train_dataset = SequenceDataset(
        file_path=args.train_file,
        tokenizer=tokenizer,
        max_length=max_seq_length,
        use_packing=True,
        tokens_per_batch=args.tokens_per_batch if hasattr(args, 'tokens_per_batch') else 2048
    )
    
    logger.info(f"Loading validation dataset from {args.val_file}")
    val_dataset = SequenceDataset(
        file_path=args.val_file,
        tokenizer=tokenizer,
        max_length=max_seq_length,
        use_packing=True,
        tokens_per_batch=args.tokens_per_batch if hasattr(args, 'tokens_per_batch') else 2048
    )
    
    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Validation dataset size: {len(val_dataset)}")
    
    # 데이터 콜레이터 설정
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # 훈련 설정 - 단순화
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=1,  # 메모리 문제 방지를 위해 1로 설정
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps * 2,  # 효과적인 배치 크기 유지
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_dir=metrics_dir,
        logging_steps=args.logging_steps,
        eval_strategy="epoch",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        fp16=False,
        bf16=True,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="tensorboard",
        optim="adamw_torch",
        # 기본적인 메모리 관리만 설정
        torch_empty_cache_steps=100,
        save_total_limit=1,
    )
    
    # 트레이너 초기화
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )
    
    # 옵티마이저 설정
    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters,
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01
    )
    
    trainer.optimizer = optimizer
    
    # 시작 시간 기록
    start_time = time.time()
    
    try:
        # 모델 훈련
        logger.info("Starting training")
        trainer.train()
        
        # 종료 시간 기록
        end_time = time.time()
        training_time = end_time - start_time
        
        # 훈련 시간 기록
        with open(os.path.join(metrics_dir, "training_time.txt"), "w") as f:
            f.write(f"Training time: {training_time} seconds\n")
        
        # 최종 모델 저장
        logger.info(f"Saving final model to {output_dir}")
        trainer.save_model()
        
    except Exception as e:
        logger.error(f"Error during training: {e}")
        # 에러 발생 시 메모리 상태 기록
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0) / 1e9
            reserved = torch.cuda.memory_reserved(0) / 1e9
            logger.error(f"GPU memory at error: Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")
        raise
    finally:
        # 메모리 정리
        torch.cuda.empty_cache()
    
    # 메트릭 저장
    with open(os.path.join(metrics_dir, "training_args.txt"), "w") as f:
        for arg, value in vars(args).items():
            f.write(f"{arg}: {value}\n")
    
    logger.info("Training completed!")
    
    return model