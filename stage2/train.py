# stage2/train.py
import logging
import time
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter

from stage2.model import SiRNAModel
from stage2.data import InstructionDataset

logger = logging.getLogger(__name__)

def train(args):
    """Stage 2: Massive Instruction Tuning"""
    # Logging setup
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    # Directory setup
    run_id = datetime.now().isoformat().split(".")[0]
    output_dir = Path(args.output_dir) / run_id
    metrics_dir = Path(str(args.output_dir).replace("models", "metrics")) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Models will be saved to: {output_dir}")
    logger.info(f"Metrics will be saved to: {metrics_dir}")

    writer = SummaryWriter(log_dir=str(metrics_dir))

    # Seed and memory setup
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.empty_cache()

    # Load base model path from Stage1
    stage1_model_dir = Path(args.stage1_model_dir)
    base_model_path_file = stage1_model_dir / "base_model_path.txt"
    if not base_model_path_file.exists():
        raise FileNotFoundError(f"base_model_path.txt not found in {stage1_model_dir}")
    base_model_path = base_model_path_file.read_text().strip()

    # Initialize SiRNAModel 
    logger.info(f"Initializing model with base model from: {base_model_path}")
    sirna_model = SiRNAModel(model_name_or_path=stage1_model_dir)
    
    # QLoRA 양자화와 함께 모델 로드
    logger.info("Loading model with QLoRA quantization...")
    model, tokenizer = sirna_model.load_model(use_4bit=True)
    
    # LoRA 파인튜닝 준비
    logger.info(f"Preparing model for LoRA fine-tuning with r={args.lora_r}, alpha={args.lora_alpha}")
    sirna_model.prepare_for_training(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout
    )
    
    # 데이터셋 로드
    logger.info(f"Loading train dataset from {args.train_file}")
    train_dataset = InstructionDataset(args.train_file, tokenizer, args.max_seq_length)
    
    logger.info(f"Loading validation dataset from {args.val_file}")
    val_dataset = InstructionDataset(args.val_file, tokenizer, args.max_seq_length)

    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Validation dataset size: {len(val_dataset)}")

    # 모델 학습
    start_time = time.time()
    logger.info("Starting training")
    
    # max_steps 파라미터 추가
    max_steps = getattr(args, 'max_steps', -1)
    logger.info(f"Max steps: {max_steps if max_steps > 0 else '자동 계산됨 (에포크 기반)'}")
    
    sirna_model.train(
        train_dataset=train_dataset,
        output_dir=str(output_dir),
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        max_steps=max_steps
    )

    end_time = time.time()
    training_time = end_time - start_time
    (metrics_dir / "training_time.txt").write_text(f"Training time: {training_time} seconds\n")

    # 모델 저장 (이미 train 메서드에서 저장됨)
    logger.info(f"Model saved to {output_dir}")
    
    # 기존 모델 경로 저장
    (output_dir / "base_model_path.txt").write_text(base_model_path)

    # 학습 인자 저장
    with (metrics_dir / "training_args.txt").open("w") as f:
        for arg, value in vars(args).items():
            f.write(f"{arg}: {value}\n")

    logger.info(f"Training completed! Run ID: {run_id}")
    torch.cuda.empty_cache()

    return sirna_model.model