import argparse
import os
import sys
import logging
from pathlib import Path

# 로깅 설정
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Eigen Omics Project")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Stage1 train parser
    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument("--stage", type=int, required=True, choices=[1, 2, 3], help="Training stage")
    train_parser.add_argument("--train_file", type=str, required=True, help="Path to training file")
    train_parser.add_argument("--val_file", type=str, required=True, help="Path to validation file")
    train_parser.add_argument("--model_dir", type=str, default="./Meta-Llama-3.1-8B", help="Path to pre-trained model")
    train_parser.add_argument("--output_dir", type=str, default="./archive/models/stage1", help="Output directory for model checkpoints")
    train_parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    train_parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    train_parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    train_parser.add_argument("--lora_r", type=int, default=128, help="LoRA r dimension")
    train_parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    train_parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout")
    train_parser.add_argument("--max_seq_length", type=int, default=1200, help="Maximum sequence length")
    train_parser.add_argument("--num_epochs", type=int, default=1, help="Number of training epochs")
    train_parser.add_argument("--save_steps", type=int, default=2000, help="Save checkpoint every X steps")
    train_parser.add_argument("--logging_steps", type=int, default=100, help="Log every X steps")
    train_parser.add_argument("--eval_steps", type=int, default=1000, help="Evaluate every X steps")
    train_parser.add_argument("--seed", type=int, default=42, help="Random seed")
    train_parser.add_argument("--tokens_per_batch", type=int, default=2048, help="Maximum tokens per batch for packing")
    train_parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID to use for training")
    
    # Stage1 evaluate parser
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate the model")
    eval_parser.add_argument("--stage", type=int, required=True, choices=[1, 2, 3], help="Evaluation stage")
    eval_parser.add_argument("--model_dir", type=str, required=True, help="Path to model directory")
    eval_parser.add_argument("--val_file", type=str, required=True, help="Path to validation file")
    eval_parser.add_argument("--output_dir", type=str, default="./archive/metrics/stage1", help="Output directory for evaluation results")
    eval_parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    eval_parser.add_argument("--max_seq_length", type=int, default=1200, help="Maximum sequence length")
    eval_parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID to use for evaluation")
    
    args = parser.parse_args()
    
    # 디렉토리 구조 확인 및 생성
    for directory in ["./archive/models/stage1", "./archive/metrics/stage1", 
                     "./archive/models/stage2", "./archive/metrics/stage2",
                     "./archive/models/stage3", "./archive/metrics/stage3"]:
        os.makedirs(directory, exist_ok=True)
    
    # 명령어가 지정되지 않은 경우 도움말 출력
    if args.command is None:
        parser.print_help()
        return
    
    # 지정된 GPU 설정
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    
    # 스테이지별 모듈 동적 로드
    if args.stage == 1:
        sys.path.append(".")
        from stage1.train import train as train_stage1
        from stage1.eval import evaluate as evaluate_stage1
        
        if args.command == "train":
            logger.info(f"Starting Stage 1 training with data: {args.train_file}")
            train_stage1(args)
        
        elif args.command == "evaluate":
            logger.info(f"Starting Stage 1 evaluation with model: {args.model_dir}")
            evaluate_stage1(args)
    
    # 향후 Stage2, Stage3 추가 예정
    else:
        logger.error(f"Stage {args.stage} is not implemented yet")
        sys.exit(1)

if __name__ == "__main__":
    main()