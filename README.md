```
python main.py train --stage 1 \
    --train_file data/stage1/dna_sequences_human_partial/train.txt \
    --val_file data/stage1/dna_sequences_human_partial/validation.txt \
    --model_dir ./Meta-Llama-3.1-8B \
    --output_dir ./archive/models/stage1 \
    --batch_size 1 \
    --gradient_accumulation_steps 48 \
    --learning_rate 1e-4 \
    --lora_r 128 \
    --lora_alpha 32 \
    --max_seq_length 1200 \
    --num_epochs 1 \
    --gpu_id 3
```

```
python main.py evaluate --stage 1 \
    --target_model 2025-03-19T07:52:37 \
    --val_file data/stage1/validation.txt \
    --batch_size 2 \
    --gpu_id 0
```

```
tensorboard --logdir ./archive/metrics/stage1 --host 0.0.0.0
```

```
python3 -m main train \
    --stage 2 \
    --train_file data/stage2/only_sirna/train.jsonl \
    --val_file data/stage2/only_sirna/validation.jsonl \
    --stage1_model_dir archive/models/stage1/2025-03-19T07:52:37 \
    --batch_size 2 \
    --gradient_accumulation_steps 20 \
    --lora_r 64 \
    --lora_alpha 32 \
    --learning_rate 1e-4 \
    --max_seq_length 1024 \
    --max_steps 3500 \
    --logging_steps 100 \
    --gpu_id 1
```

```
python main.py evaluate --stage 2 \
    --target_model 2025-03-19T09:03:30 \
    --val_file data/stage2/mini/validation.jsonl \
    --batch_size 2 \
    --gpu_id 0
```

```
tensorboard --logdir ./archive/metrics/stage2 --host 0.0.0.0
```