# stage2/train.py
import logging
import time
from datetime import datetime
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from torch.utils.tensorboard import SummaryWriter

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

    # Load tokenizer from base model path
    logger.info(f"Loading tokenizer from base model: {base_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    # Load Stage1 model
    logger.info(f"Loading Stage1 model from {stage1_model_dir}")
    model = AutoModelForCausalLM.from_pretrained(
        stage1_model_dir,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
        attn_implementation="flash_attention_2",
    )
    model.gradient_checkpointing_enable()

    # Prepare model for training
    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)

    # Load datasets
    train_dataset = InstructionDataset(args.train_file, tokenizer, args.max_seq_length)
    val_dataset = InstructionDataset(args.val_file, tokenizer, args.max_seq_length)

    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Validation dataset size: {len(val_dataset)}")

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_dir=str(metrics_dir),
        logging_steps=args.logging_steps,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        bf16=True,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="tensorboard",
        optim="adamw_torch",
        save_total_limit=1,
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    # Train model
    start_time = time.time()
    logger.info("Starting training")
    trainer.train()

    end_time = time.time()
    training_time = end_time - start_time
    (metrics_dir / "training_time.txt").write_text(f"Training time: {training_time} seconds\n")

    # Save model and tokenizer
    trainer.save_model()
    tokenizer.save_pretrained(str(output_dir))
    model.peft_config[model.active_adapter].save_pretrained(str(output_dir))
    (output_dir / "base_model_path.txt").write_text(base_model_path)

    # Save training args
    with (metrics_dir / "training_args.txt").open("w") as f:
        for arg, value in vars(args).items():
            f.write(f"{arg}: {value}\n")

    logger.info(f"Training completed! Run ID: {run_id}")
    torch.cuda.empty_cache()

    return model