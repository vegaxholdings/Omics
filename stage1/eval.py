import os
import torch
import logging
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader
from peft import PeftModel, PeftConfig
from .data import SequenceDataset

logger = logging.getLogger(__name__)

def evaluate(args):
    # 설정 및 로깅
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    
    # 출력 디렉토리 설정
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 메모리 캐시 정리
    torch.cuda.empty_cache()

    # PEFT 구성 로드
    try:
        peft_config = PeftConfig.from_pretrained(args.model_dir)
        base_model_path = peft_config.base_model_name_or_path
        logger.info(f"Found PEFT config. Base model path: {base_model_path}")
    except:
        # PEFT 구성을 찾을 수 없는 경우 기본 가정
        base_model_path = "./Meta-Llama-3.1-8B"
        logger.warning(f"Could not find PEFT config. Using default base model path: {base_model_path}")
    
    # 토크나이저 로드 - 원래 기본 모델에서
    logger.info(f"Loading tokenizer from {base_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    tokenizer.pad_token = tokenizer.eos_token
    
    # 기본 모델 로드
    logger.info(f"Loading base model from {base_model_path}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    
    # 어댑터 로드 및 적용
    logger.info(f"Loading adapter from {args.model_dir}")
    model = PeftModel.from_pretrained(base_model, args.model_dir)
    
    # 모델을 GPU로 이동
    if torch.cuda.is_available():
        model = model.to("cuda")
    
    # 데이터셋 로드
    logger.info(f"Loading dataset from {args.val_file}")
    val_dataset = SequenceDataset(
        file_path=args.val_file,
        tokenizer=tokenizer,
        max_length=args.max_seq_length if hasattr(args, 'max_seq_length') else 1200,
        use_packing=False
    )
    
    # 데이터로더 초기화
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )
    
    # 모델을 평가 모드로 설정
    model.eval()
    
    # 평가 로직 구현
    total_loss = 0.0
    total_samples = 0
    
    logger.info("Starting evaluation")
    with torch.no_grad():
        for batch in val_dataloader:
            # 데이터를 디바이스로 이동
            batch = {k: v.to(model.device) for k, v in batch.items()}
            
            # 모델 출력
            outputs = model(**batch)
            
            # 손실 계산
            loss = outputs.loss
            
            # 통계 업데이트
            total_loss += loss.item() * batch["input_ids"].size(0)
            total_samples += batch["input_ids"].size(0)
    
    # 최종 퍼플렉시티 계산
    avg_loss = total_loss / total_samples
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    
    # 결과 출력
    logger.info(f"Evaluation completed. Loss: {avg_loss}, Perplexity: {perplexity}")
    
    # 메트릭 저장
    metrics = {
        "loss": avg_loss,
        "perplexity": perplexity,
    }
    
    with open(os.path.join(args.output_dir, "eval_results.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    
    # 메모리 정리
    torch.cuda.empty_cache()
    
    return metrics