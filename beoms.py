import os
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments


#os.environ["CUDA_VISIBLE_DEVICES"] = "3"


# Stage1 모델 경로 및 기본 모델 경로 설정
stage2_model_dir = Path("/home/ec2-user/eigen-omics/archive/models/stage2/2025-03-19T09:03:30")
base_model_path_file = stage2_model_dir / "base_model_path.txt"
assert base_model_path_file.exists()
base_model_path = base_model_path_file.read_text().strip()
print(f"Loading base model path from file: {base_model_path}")

model1 = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    #"./Meta-Llama-3.1-8B",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map={"": "cuda:2"},
    attn_implementation="flash_attention_2",
)
# Prepare model for training
model1_kbit = prepare_model_for_kbit_training(model1)
model1_peft = PeftModel.from_pretrained(model1_kbit, stage2_model_dir)

######

# Stage1 모델 경로 및 기본 모델 경로 설정
stage2_model_dir = Path("/home/ec2-user/eigen-omics/archive/models/stage2/2025-03-19T09:59:34")

model2 = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map={"": "cuda:3"},
    attn_implementation="flash_attention_2",
)
# Prepare model for training
model2_kbit = prepare_model_for_kbit_training(model2)
model2_peft = PeftModel.from_pretrained(model2_kbit, stage2_model_dir)

####
for idx, (layer1, layer2) in enumerate(zip(model1_peft.base_model.model.model.layers, model2_peft.base_model.model.model.layers)):
    # Move both layers to CPU
    layer1_cpu = layer1.to("cpu")
    layer2_cpu = layer2.to("cpu")
    for param1, param2 in zip(layer1_cpu.parameters(), layer2_cpu.parameters()):
    # Optionally, you can compare weights
      if torch.allclose(param1, param2, atol=3e-4):
        print(f"Layer {idx} q_proj weights are identical.")
      else:
        diff = (param1 - param2).abs() > 3e-4
        num_diff = diff.sum().item()
        total_params = param1.numel()
        percent_diff = 100.0 * num_diff / total_params
        print(f"Layer {idx} weights differ in ~{percent_diff:.4f}% of elements.")
