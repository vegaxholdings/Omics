import os
import json
import argparse
import time
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from src.utils import set_seed
from train import train_model
from evaluate import evaluate_model

def objective(trial):
    """Objective function for hyperparameter optimization"""
    # 시작 시간 기록
    start_time = time.time()
    
    # 핵심 하이퍼파라미터만 선별하여 튜닝 (시간 절약)
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True),
        # 작은 배치 크기는 학습 속도 저하의 원인이 될 수 있으므로 제외
        "batch_size": 4,  # 고정
        "grad_accum_steps": trial.suggest_categorical("grad_accum_steps", [4, 8, 16]),
        # 에포크 수는 줄여서 시간 절약
        "num_epochs": trial.suggest_int("num_epochs", 1, 3),
        # LoRA 파라미터 중 가장 중요한 r만 튜닝
        "r": trial.suggest_categorical("r", [8, 16, 32]),
        "lora_alpha": 32,  # 고정
        "lora_dropout": 0.05,  # 고정
    }
    
    # 임시 출력 디렉토리 생성
    output_dir = f"./hp_tuning/trial_{trial.number}"
    os.makedirs(output_dir, exist_ok=True)
    
    # 시드 고정
    set_seed(42)
    
    # 미니 데이터셋으로 모델 학습
    train_model(
        train_data_path="data/test/train.jsonl",
        output_dir=output_dir,
        batch_size=params["batch_size"],
        gradient_accumulation_steps=params["grad_accum_steps"],
        num_epochs=params["num_epochs"],
        learning_rate=params["learning_rate"],
        r=params["r"],
        lora_alpha=params["lora_alpha"],
        lora_dropout=params["lora_dropout"]
    )
    
    # 학습된 모델 평가
    mae = evaluate_model(
        model_path=output_dir,
        data_path="data/test/validation.jsonl"
    )
    
    # 하이퍼파라미터와 성능, 소요 시간 저장
    elapsed_time = time.time() - start_time
    with open(os.path.join(output_dir, "params.json"), "w") as f:
        json.dump({
            "params": params,
            "mae": mae,
            "time_minutes": elapsed_time / 60
        }, f, indent=2)
    
    # 진행 상황 로깅
    print(f"Trial {trial.number} completed. MAE: {mae:.4f}, Time: {elapsed_time/60:.2f} minutes")
    
    # Pruning 구현: 첫 5회 시도 후, 중앙값보다 20% 이상 나쁜 결과는 조기 중단
    if trial.number >= 5:
        completed_trials = [t for t in trial.study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if len(completed_trials) >= 5:
            completed_values = [t.value for t in completed_trials if t.value is not None]
            median_value = sorted(completed_values)[len(completed_values) // 2]
            if mae > median_value * 1.2:  # 중앙값보다 20% 이상 나쁜 경우
                print(f"Trial {trial.number} pruned. MAE: {mae:.4f} > {median_value * 1.2:.4f}")
                raise optuna.exceptions.TrialPruned()
    
    return mae

def run_hyperparameter_tuning(n_trials=12, max_hours=15, study_name="sirna_hp_tuning"):
    """시간 제약이 있는 하이퍼파라미터 튜닝 실행"""
    # 시작 시간 기록
    start_time = time.time()
    total_seconds = max_hours * 3600
    
    # Optuna 스터디 생성
    study = optuna.create_study(
        study_name=study_name,
        direction="minimize",  # MAE 최소화
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(n_startup_trials=3, n_warmup_steps=3)
    )
    
    # 튜닝 실행 (시간 제약 추가)
    completed_trials = 0
    for _ in range(n_trials):
        # 남은 시간 확인
        elapsed_seconds = time.time() - start_time
        if elapsed_seconds > total_seconds * 0.9:  # 90% 시간 초과 시 종료
            print(f"Time limit approaching. Stopping after {completed_trials} trials.")
            break
        
        # 다음 시도 실행
        try:
            study.optimize(objective, n_trials=1)
            completed_trials += 1
        except Exception as e:
            print(f"Error in trial: {e}")
        
        # 현재까지 최고 성능 출력
        print(f"Best MAE so far: {study.best_value:.4f}")
        print(f"Completed {completed_trials}/{n_trials} trials")
        
        # 남은 시간 예측
        if completed_trials > 0:
            avg_time_per_trial = elapsed_seconds / completed_trials
            remaining_trials = min(n_trials - completed_trials, 
                                  int((total_seconds - elapsed_seconds) / avg_time_per_trial))
            print(f"Estimated remaining trials possible: {remaining_trials}")
            print(f"Estimated completion in: {avg_time_per_trial * remaining_trials / 3600:.2f} hours")
    
    # 결과 출력
    print("\n" + "="*50)
    print("Hyperparameter Tuning Results:")
    print("="*50)
    print(f"Completed trials: {completed_trials}")
    print(f"Total time: {(time.time() - start_time) / 3600:.2f} hours")
    print("\nBest trial:")
    best_trial = study.best_trial
    print(f"  Trial number: {best_trial.number}")
    print(f"  MAE: {best_trial.value:.4f}")
    print("  Params:")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")
    
    # 모든 시도 정보 저장
    os.makedirs("./hp_tuning_results", exist_ok=True)
    results = {
        "best_params": best_trial.params,
        "best_mae": best_trial.value,
        "completed_trials": completed_trials,
        "total_time_hours": (time.time() - start_time) / 3600,
        "all_trials": []
    }
    
    for trial in study.trials:
        if trial.state == optuna.trial.TrialState.COMPLETE:
            results["all_trials"].append({
                "number": trial.number,
                "params": trial.params,
                "mae": trial.value
            })
    
    # 최적 하이퍼파라미터와 모든 시도 결과 저장
    with open("./hp_tuning_results/tuning_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    return best_trial.params

def train_with_best_params(best_params, train_data_path, output_dir):
    """최적 하이퍼파라미터로 최종 모델 학습"""
    print(f"Training final model with best hyperparameters...")
    
    # 기본값 설정
    params = {
        "batch_size": 4,
        "grad_accum_steps": 8,
        "num_epochs": 3,
        "learning_rate": 2e-4,
        "r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05
    }
    
    # 제공된 최적 파라미터로 업데이트
    for key, value in best_params.items():
        params[key] = value
    
    train_model(
        train_data_path=train_data_path,
        output_dir=output_dir,
        batch_size=params["batch_size"],
        gradient_accumulation_steps=params["grad_accum_steps"],
        num_epochs=params["num_epochs"],
        learning_rate=params["learning_rate"],
        r=params["r"],
        lora_alpha=params["lora_alpha"],
        lora_dropout=params["lora_dropout"]
    )
    
    print(f"Final model trained and saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Time-constrained hyperparameter tuning for ChatsiRNA")
    parser.add_argument("--n_trials", type=int, default=12, 
                        help="Maximum number of trials for hyperparameter search")
    parser.add_argument("--max_hours", type=float, default=15, 
                        help="Maximum hours to run tuning")
    parser.add_argument("--study_name", type=str, default="sirna_hp_tuning", 
                        help="Name of the optimization study")
    parser.add_argument("--full_train", action="store_true", 
                        help="Train on full dataset after tuning")
    parser.add_argument("--train_data", type=str, default="data/train.jsonl", 
                        help="Path to full training data")
    parser.add_argument("--output_dir", type=str, default="./final_model", 
                        help="Output directory for final model")
    parser.add_argument("--seed", type=int, default=42, 
                        help="Random seed")
    
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    # 하이퍼파라미터 튜닝 실행
    best_params = run_hyperparameter_tuning(
        n_trials=args.n_trials, 
        max_hours=args.max_hours,
        study_name=args.study_name
    )
    
    # 최적 하이퍼파라미터로 전체 데이터셋에 대해 학습 (선택사항)
    if args.full_train:
        train_with_best_params(
            best_params=best_params,
            train_data_path=args.train_data,
            output_dir=args.output_dir
        )