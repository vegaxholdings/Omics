import re
import random

import torch
import numpy as np
from tqdm import tqdm

def set_seed(seed):
    """Set seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def extract_number_from_text(text):
    """Extract numerical value from text"""
    # Try to find a decimal number
    pattern = r'(\d+\.\d+)'
    matches = re.findall(pattern, text)
    if matches:
        return float(matches[0])
    
    # If not found, try to find an integer
    pattern = r'(\d+)'
    matches = re.findall(pattern, text)
    if matches:
        return float(matches[0])
    
    return None

def generate_predictions(model, tokenizer, dataset, max_new_tokens=64, batch_size=4):
    """Generate predictions using the model"""
    model.eval()
    predictions = []
    
    for i in tqdm(range(0, len(dataset), batch_size)):
        batch = dataset.input_texts[i:i+batch_size]
        
        # Format inputs with instruction format
        formatted_batch = [f"[INST] {text} [/INST]" for text in batch]
        
        # Tokenize inputs
        inputs = tokenizer(formatted_batch, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Generate outputs
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                # 샘플링 관련 설정 수정
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                num_beams=1,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
        
        # Decode outputs
        generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        
        # Extract responses (part after [/INST])
        for j, full_text in enumerate(generated_texts):
            # Extract the response part (after [/INST])
            response = full_text.split("[/INST]")[-1].strip()
            
            # Extract numerical value
            value = extract_number_from_text(response)
            predictions.append(value if value is not None else None)
    
    return predictions