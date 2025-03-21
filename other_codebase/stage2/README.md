# ChatsiRNA README

## 핵심 사용 방법

### 1. 모델 학습

- 테스트용 미니 데이터셋으로 모델을 학습합니다
```
python main.py train --train_data data/test/train.jsonl --output_dir ./trained_model  
```
- 추가 옵션
```
python main.py train --train_data data/test/train.jsonl --output_dir ./trained_model --batch_size 4 --grad_accum_steps 8 --num_epochs 3 --lr 2e-4  
```

### 2. 모델 평가
- 학습된 모델의 성능을 평가합니다
```
python main.py evaluate --model_path ./trained_model --data_path data/test/validation.jsonl  
```
- 사용자 정의 정확도 임계값 설정
```
python main.py evaluate --model_path ./trained_model --data_path data/test/validation.jsonl --max_diff 10  
```

### 3. 대화형 인터페이스 사용
- 모델과 직접 대화하여 siRNA 효율성을 예측합니다
```
python main.py chat --model_path ./trained_model  
```

### 4. 하이퍼파라미터 튜닝
- 최적의 하이퍼파라미터를 찾습니다 (시간 제약 설정)
```
python main.py tune --n_trials 12 --max_hours 15
```

### 5. 최적 하이퍼파라미터로 모델 학습
- 찾은 최적 하이퍼파라미터로 전체 데이터셋을 학습합니다
```
python main.py train_best --train_data data/train.jsonl --output_dir ./final_model
```