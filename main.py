# main.py
import os
import sys
import logging
import argparse
from pathlib import Path

logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Eigen Omics Project")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Stage1 and Stage2 train parser
    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument("--stage", type=int, required=True, choices=[1, 2], help="Training stage")
    train_parser.add_argument("--train_file", type=str, required=True, help="Path to training file")
    train_parser.add_argument("--val_file", type=str, required=True, help="Path to validation file")
    train_parser.add_argument("--model_dir", type=str, default="./Meta-Llama-3.1-8B", help="Path to pre-trained model (Stage1)")
    train_parser.add_argument("--stage1_model_dir", type=str, help="Path to Stage1 model (Stage2)")
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

    # Stage1 and Stage2 evaluate parser
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate the model")
    eval_parser.add_argument("--stage", type=int, required=True, choices=[1, 2], help="Evaluation stage")
    eval_parser.add_argument("--target_model", type=str, required=True, help="Name of the trained model directory to evaluate")
    eval_parser.add_argument("--val_file", type=str, required=True, help="Path to validation file")
    eval_parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    eval_parser.add_argument("--max_seq_length", type=int, default=1200, help="Maximum sequence length")
    eval_parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID to use for evaluation")

    args = parser.parse_args()

    # Directory setup
    for stage in [1, 2]:
        for directory in [f"./archive/models/stage{stage}", f"./archive/metrics/stage{stage}"]:
            Path(directory).mkdir(parents=True, exist_ok=True)

    if args.command is None:
        parser.print_help()
        return

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    # Dynamic module loading
    if args.stage == 1:
        sys.path.append(".")
        from stage1.train import train as train_stage1
        from stage1.eval import evaluate as evaluate_stage1
        if args.command == "train":
            logger.info(f"Starting Stage 1 training with data: {args.train_file}")
            train_stage1(args)
        elif args.command == "evaluate":
            model_dir = Path(f"./archive/models/stage1/{args.target_model}")
            args.model_dir = str(model_dir)
            args.output_dir = str(Path(f"./archive/metrics/stage1/{args.target_model}"))
            if not model_dir.exists():
                logger.error(f"Model directory {model_dir} does not exist.")
                sys.exit(1)
            logger.info(f"Starting Stage 1 evaluation with model: {args.model_dir}")
            evaluate_stage1(args)
    elif args.stage == 2:
        sys.path.append(".")
        from stage2.train import train as train_stage2
        from stage2.eval import evaluate as evaluate_stage2
        if args.command == "train":
            if not args.stage1_model_dir:
                logger.error("Stage1 model directory must be provided for Stage2 training")
                sys.exit(1)
            args.output_dir = "./archive/models/stage2"
            logger.info(f"Starting Stage 2 training with data: {args.train_file}")
            train_stage2(args)
        elif args.command == "evaluate":
            model_dir = Path(f"./archive/models/stage2/{args.target_model}")
            args.model_dir = str(model_dir)
            args.output_dir = str(Path(f"./archive/metrics/stage2/{args.target_model}"))
            if not model_dir.exists():
                logger.error(f"Model directory {model_dir} does not exist.")
                sys.exit(1)
            logger.info(f"Starting Stage 2 evaluation with model: {args.model_dir}")
            evaluate_stage2(args)
    else:
        logger.error(f"Stage {args.stage} is not implemented yet")
        sys.exit(1)

if __name__ == "__main__":
    main()