# stage2/eval.py (수정된 코드)
import re
import json
import logging
from pathlib import Path

import torch
import numpy as np
from torch.utils.data import DataLoader
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import mean_absolute_error, f1_score

from stage2.data import InstructionDataset

logger = logging.getLogger(__name__)

def compute_mixed_score(predictions, labels, tasks):
    """Compute Mixed Score based on task type."""
    metrics = {}
    task_types = set(tasks)
    
    for task in task_types:
        task_preds = [p for p, t in zip(predictions, tasks) if t == task]
        task_labels = [l for l, t in zip(labels, tasks) if t == task]
        
        # 텐서를 CPU로 옮기고 NumPy 배열로 변환
        if isinstance(task_preds[0], torch.Tensor):
            task_preds = [p.cpu().numpy() for p in task_preds]
        else:
            task_preds = np.array(task_preds)
        
        if isinstance(task_labels[0], torch.Tensor):
            task_labels = [l.cpu().numpy() for l in task_labels]
        else:
            task_labels = np.array(task_labels)
        
        # MAE 계산
        mae = mean_absolute_error(task_labels, task_preds)
        range_mae = mae / (max(task_labels) - min(task_labels)) if max(task_labels) != min(task_labels) else mae
        
        # 회귀 태스크인지 분류 태스크인지에 따라 점수 계산
        if is_regression_task(task):
            mixed_score = 50 * (1 - mae / 100) + 50 * (1 - range_mae / 100)
            metrics[task] = {"MAE": mae, "Mixed Score": mixed_score}
        else:
            f1 = f1_score(task_labels, task_preds, average="macro")
            mixed_score = 50 * (1 - mae / 100) + 50 * f1 * (1 - range_mae / 100)
            metrics[task] = {"F1 Score": f1, "MAE": mae, "Mixed Score": mixed_score}
    
    return metrics

def is_regression_task(task: str) -> bool:
    """Determine if the task is regression based on task name."""
    regression_tasks = {"sirnaEfficiency-sirnaEfficiency"}  # Add more as needed
    return task in regression_tasks

def evaluate(args):
    """Evaluate the Stage2 model."""
    # Logging setup
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    # Directory setup
    model_dir = Path(args.model_dir)
    metrics_dir = Path(args.output_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Evaluating model from: {model_dir}")
    logger.info(f"Evaluation results will be saved to: {metrics_dir}")

    torch.cuda.empty_cache()

    # Load base model path from Stage2 model directory
    base_model_path_file = model_dir / "base_model_path.txt"
    if not base_model_path_file.exists():
        raise FileNotFoundError(f"base_model_path.txt not found in {model_dir}")
    base_model_path = base_model_path_file.read_text().strip()

    # Load tokenizer from base model path
    logger.info(f"Loading tokenizer from base model: {base_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    # Load Stage2 model
    logger.info(f"Loading model from {model_dir}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base_model, model_dir)
    if torch.cuda.is_available():
        model = model.to("cuda")

    # Load dataset
    val_dataset = InstructionDataset(args.val_file, tokenizer, args.max_seq_length)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Evaluation
    model.eval()
    predictions = []
    labels = []
    tasks = []

    logger.info("Starting evaluation")
    with torch.no_grad():
        for batch in val_dataloader:
            batch = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            # 모델이 순차적으로 응답을 생성하도록 generate() 함수 사용
            generated_ids = model.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"]            
            )
            decoded_preds = [tokenizer.decode(ids, skip_special_tokens=True) for ids in generated_ids]
            # 숫자 추출 로직 유지
            numeric_preds = []
            for pred in decoded_preds:
                match = re.search(r'\d+\.?\d*', pred)  # 첫 번째 정수 또는 소수를 찾음
                numeric_preds.append(float(match.group()) if match else 0.0)
            
            predictions.extend(numeric_preds)
            labels.extend(batch["label"])
            tasks.extend(batch["task"])

    # Compute metrics
    metrics = compute_mixed_score(predictions, labels, tasks)

    # Save metrics
    metrics_file = metrics_dir / "eval_results.json"
    with metrics_file.open("w") as f:
        json.dump(metrics, f, indent=2)

    summary_file = metrics_dir / "evaluation_summary.txt"
    with summary_file.open("w") as f:
        f.write(f"Model evaluated: {model_dir}\n")
        for task, metric in metrics.items():
            f.write(f"Task: {task}\n")
            for key, value in metric.items():
                f.write(f"{key}: {value}\n")
        f.write(f"Validation file: {args.val_file}\n")
        f.write(f"Batch size: {args.batch_size}\n")

    logger.info(f"Evaluation results saved to {metrics_dir}")
    torch.cuda.empty_cache()

    return metrics