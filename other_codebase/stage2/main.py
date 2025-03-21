import argparse
import json

from train import train_model
from evaluate import evaluate_model
from chat import chat_with_model
from src.utils import set_seed
from tune import run_hyperparameter_tuning, train_with_best_params

def main():
    parser = argparse.ArgumentParser(description="ChatsiRNA - siRNA 효율성 예측 모델")
    subparsers = parser.add_subparsers(dest="command", help="실행할 명령어")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="모델 학습")
    train_parser.add_argument("--train_data", type=str, default="data/test/train.jsonl", 
                             help="학습 데이터 경로")
    train_parser.add_argument("--output_dir", type=str, default="./trained_model", 
                             help="모델 저장 디렉토리")
    train_parser.add_argument("--batch_size", type=int, default=4, 
                             help="배치 크기")
    train_parser.add_argument("--grad_accum_steps", type=int, default=8, 
                             help="그래디언트 누적 단계")
    train_parser.add_argument("--num_epochs", type=int, default=3, 
                             help="학습 에폭 수")
    train_parser.add_argument("--lr", type=float, default=2e-4, 
                             help="학습률")
    train_parser.add_argument("--seed", type=int, default=42, 
                             help="랜덤 시드")
    
    # subparsers에 다음 명령어 추가
    tune_parser = subparsers.add_parser("tune", help="하이퍼파라미터 튜닝")
    tune_parser.add_argument("--n_trials", type=int, default=12, 
                            help="하이퍼파라미터 검색 최대 시도 횟수")
    tune_parser.add_argument("--max_hours", type=float, default=15, 
                            help="튜닝 최대 실행 시간(시간)")
    tune_parser.add_argument("--study_name", type=str, default="sirna_hp_tuning", 
                            help="최적화 스터디 이름")
    tune_parser.add_argument("--seed", type=int, default=42, 
                            help="랜덤 시드")

    # train_best 명령어도 추가
    train_best_parser = subparsers.add_parser("train_best", help="최적 하이퍼파라미터로 학습")
    train_best_parser.add_argument("--params_path", type=str, default="./hp_tuning_results/tuning_results.json", 
                                help="최적 하이퍼파라미터 파일 경로")
    train_best_parser.add_argument("--train_data", type=str, default="data/train.jsonl", 
                                help="학습 데이터 경로")
    train_best_parser.add_argument("--output_dir", type=str, default="./final_model", 
                                help="최종 모델 저장 디렉토리")
    train_best_parser.add_argument("--seed", type=int, default=42, 
                                help="랜덤 시드")

    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="모델 평가")
    eval_parser.add_argument("--model_path", type=str, required=True, 
                            help="학습된 모델 경로")
    eval_parser.add_argument("--data_path", type=str, default="data/test/validation.jsonl", 
                            help="평가 데이터 경로")
    eval_parser.add_argument("--seed", type=int, default=42, 
                            help="랜덤 시드")
    
    # Chat command
    chat_parser = subparsers.add_parser("chat", help="모델과 대화")
    chat_parser.add_argument("--model_path", type=str, required=True, 
                           help="학습된 모델 경로")
    chat_parser.add_argument("--seed", type=int, default=42, 
                           help="랜덤 시드")
    
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    if args.command == "train":
        print("학습을 시작합니다...")
        train_model(
            args.train_data,
            args.output_dir,
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum_steps,
            num_epochs=args.num_epochs,
            learning_rate=args.lr
        )
    elif args.command == "evaluate":
        print("평가를 시작합니다...")
        mae = evaluate_model(
            args.model_path,
            args.data_path
        )
        print(f"최종 MAE: {mae:.4f} (낮을수록 좋음)")
    elif args.command == "tune":
        print("시간 제약 하이퍼파라미터 튜닝을 시작합니다...")
        print(f"최대 {args.n_trials}회 시도 또는 {args.max_hours}시간 중 먼저 도달하는 조건에서 종료됩니다.")
        best_params = run_hyperparameter_tuning(
            n_trials=args.n_trials,
            max_hours=args.max_hours,
            study_name=args.study_name
        )
        print(f"최적 하이퍼파라미터 검색 완료!")

    elif args.command == "train_best":
        print("최적 하이퍼파라미터로 학습을 시작합니다...")
        with open(args.params_path, "r") as f:
            params_data = json.load(f)
            best_params = params_data["best_params"]
        
        train_with_best_params(
            best_params=best_params,
            train_data_path=args.train_data,
            output_dir=args.output_dir
        )
        print(f"최종 모델 학습 완료! 모델이 {args.output_dir}에 저장되었습니다.")
    elif args.command == "chat":
        print("모델과 대화를 시작합니다...")
        chat_with_model(args.model_path)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()