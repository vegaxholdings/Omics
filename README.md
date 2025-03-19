```
python main.py train --stage 1 \
    --train_file data/stage1/train.txt \
    --val_file data/stage1/validation.txt \
    --model_dir ./Meta-Llama-3.1-8B \
    --output_dir ./archive/models/stage1 \
    --batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-4 \
    --lora_r 128 \
    --lora_alpha 32 \
    --max_seq_length 1200 \
    --num_epochs 1 \
    --gpu_id 0
```


```
python main.py evaluate --stage 1 \
    --target_model 2025-03-19T07:52:37 \
    --val_file data/stage1/validation.txt \
    --batch_size 2 \
    --gpu_id 0
```