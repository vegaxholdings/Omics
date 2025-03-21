# stage2/eval.py
import logging
import json
from pathlib import Path
from collections import defaultdict

import torch
import numpy as np
from sklearn.metrics import mean_absolute_error, f1_score
from transformers import AutoModelForCausalLM, AutoTokenizer

from stage2.model import SiRNAModel
from stage2.data import InstructionDataset, extract_value_from_response, calculate_mae

logger = logging.getLogger(__name__)

def compute_mixed_score(predictions, labels, tasks):
    """
    논문에 정의된 Mixed Score 계산
    Mixed Score = 50% * (1 - MAE/100) + 50% * (1 - Range-MAE/100) (회귀)
    Mixed Score = 50% * (1 - MAE/100) + 50% * F1 * (1 - Range-MAE/100) (분류)
    """
    metrics = {}
    task_metrics = defaultdict(list)
    
    # 태스크별로 예측과 레이블 그룹화
    for pred, label, task in zip(predictions, labels, tasks):
        task_metrics[task].append((pred, label))
    
    # 각 태스크별 메트릭 계산
    for task, values in task_metrics.items():
        task_preds = [v[0] for v in values]
        task_labels = [v[1] for v in values]
        
        # 숫자형으로 변환 (텐서 등의 경우)
        if isinstance(task_preds[0], torch.Tensor):
            task_preds = [p.item() if hasattr(p, 'item') else p for p in task_preds]
        
        if isinstance(task_labels[0], torch.Tensor):
            task_labels = [l.item() if hasattr(l, 'item') else l for l in task_labels]
        
        # 작업 유형 확인 (회귀 vs 분류)
        is_regression = True
        if all(isinstance(label, str) for label in task_labels) or all(label in ["positive", "negative"] for label in task_labels):
            is_regression = False
            # 분류 작업의 경우 이진 레이블로 변환
            if all(label in ["positive", "negative"] for label in task_labels):
                task_labels = [1 if label == "positive" else 0 for label in task_labels]
                task_preds = [1 if pred == "positive" else 0 for pred in task_preds]
        
        if is_regression:
            # MAE 계산
            mae = mean_absolute_error(task_labels, task_preds)
            range_mae = mae / (max(task_labels) - min(task_labels)) if max(task_labels) != min(task_labels) else mae
            
            # 회귀를 위한 Mixed Score 계산
            mixed_score = 50 * (1 - mae / 100) + 50 * (1 - range_mae / 100)
            metrics[task] = {"MAE": mae, "Mixed Score": mixed_score}
        else:
            # 분류를 위한 F1 Score 계산
            f1 = f1_score(task_labels, task_preds, average="macro")
            mixed_score = 50 * (1 - mae / 100) + 50 * f1 * (1 - range_mae / 100)
            metrics[task] = {"F1 Score": f1, "MAE": mae, "Mixed Score": mixed_score}
    
    # 전체 데이터셋에 대한 평균 메트릭
    all_metrics = {
        "overall_mixed_score": sum(m["Mixed Score"] for m in metrics.values()) / len(metrics)
    }
    
    return {**metrics, **all_metrics}

def is_regression_task(task: str) -> bool:
    """Determine if the task is regression based on task name."""
    regression_tasks = {"sirnaEfficiency-sirnaEfficiency"}  # Add more as needed
    return task in regression_tasks

def evaluate(args):
    """Stage 2 모델 평가"""
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    
    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Evaluating model: {model_dir}")
    
    # SiRNAModel을 사용하여 모델 로드
    sirna_model = SiRNAModel()
    model, tokenizer = sirna_model.load_trained_model(model_dir)
    
    # 검증 데이터셋 로드
    logger.info(f"Loading validation dataset from {args.val_file}")
    val_dataset = InstructionDataset(args.val_file, tokenizer, args.max_seq_length)
    
    # 예측 실행
    logger.info("Starting evaluation...")
    model.eval()
    
    predictions = []
    labels = []
    tasks = []
    
    with torch.no_grad():
        for i in range(len(val_dataset)):
            if i % 10 == 0:
                logger.info(f"Processing example {i}/{len(val_dataset)}")
            
            batch = val_dataset[i]
            batch = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # 입력 ID와 어텐션 마스크 추출
            input_ids = batch["input_ids"].unsqueeze(0)
            attention_mask = batch["attention_mask"].unsqueeze(0)
            
            # 모델 생성
            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=256,
                temperature=0.7,
                top_p=0.9,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
            
            # 텍스트 디코딩
            generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            
            # [/INST] 이후의 응답 부분만 추출
            response = generated_text.split("[/INST]")[-1].strip()
            
            # 레이블과 예측 비교 준비
            true_label = batch["label"]
            
            # 레이블 유형에 따라 처리
            if isinstance(true_label, (int, float)) or (isinstance(true_label, str) and true_label.replace('.', '', 1).isdigit()):
                # 회귀 작업: 응답에서 숫자 추출
                pred_value = extract_value_from_response(response)
                if pred_value is not None:
                    predictions.append(pred_value)
                    labels.append(float(true_label) if isinstance(true_label, str) else true_label)
                    tasks.append(batch["task"])
            else:
                # 분류 작업: 텍스트 응답 그대로 사용
                predictions.append(response)
                labels.append(true_label)
                tasks.append(batch["task"])
    
    # 메트릭 계산
    metrics = compute_mixed_score(predictions, labels, tasks)
    
    # 결과 저장
    logger.info("Evaluation metrics:")
    for task, task_metrics in metrics.items():
        if task != "overall_mixed_score":
            logger.info(f"{task}:")
            for metric_name, value in task_metrics.items():
                logger.info(f"  {metric_name}: {value:.4f}")
    
    logger.info(f"Overall Mixed Score: {metrics['overall_mixed_score']:.4f}")
    
    # 메트릭을 JSON으로 저장
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    return metrics