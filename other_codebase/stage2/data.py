import re
import json

from torch.utils.data import Dataset

class SiRNADataset(Dataset):
    def __init__(self, file_path, tokenizer=None):
        self.samples = []
        self.labels = []
        self.input_texts = []
        self.output_texts = []
        
        # Load data from jsonl file
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                self.samples.append(data)
                self.labels.append(float(data['label']))
                self.input_texts.append(data['input'])
                self.output_texts.append(data['output'])
        
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        if self.tokenizer:
            # Get input and output text
            input_text = sample['input']
            output_text = sample['output']
            
            # Format as instruction-tuning data for Llama 3.1
            formatted_text = f"<s>[INST] {input_text} [/INST] {output_text}</s>"
            
            # Tokenize the formatted text
            encodings = self.tokenizer(formatted_text, truncation=True, 
                                       padding="max_length", max_length=2048, 
                                       return_tensors="pt")
            
            # Create labels (shift input_ids right by one)
            input_ids = encodings["input_ids"].squeeze()
            attention_mask = encodings["attention_mask"].squeeze()
            
            # For loss calculation, we only want to compute loss on the response
            # Mark input tokens with -100 to ignore them in loss computation
            labels = input_ids.clone()
            
            # Find the position of [/INST] token to mask out input
            inst_token_pos = (input_ids == self.tokenizer.encode("[/INST]", add_special_tokens=False)[-1]).nonzero()
            if len(inst_token_pos) > 0:
                # Mask out everything before and including [/INST]
                labels[:inst_token_pos[-1] + 1] = -100
            
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels
            }
        else:
            return sample
    
    def get_labels(self):
        """Return the list of labels for evaluation"""
        return self.labels


def extract_value_from_response(response):
    """Extract numerical value from model's response"""
    # Look for patterns like "percentage of XX.XX" or just a number with decimal
    pattern = r'(\d+\.\d+)'
    matches = re.findall(pattern, response)
    
    if matches:
        try:
            return float(matches[0])
        except ValueError:
            pass
    
    # Fallback: look for any number
    pattern = r'(\d+)'
    matches = re.findall(pattern, response)
    
    if matches:
        try:
            return float(matches[0])
        except ValueError:
            pass
    
    return None


def bin_label(label, bin_size=10):
    """Bin labels into categories of size bin_size"""
    return int(label // bin_size)


def calculate_mae(predictions, ground_truth):
    """Calculate Mean Absolute Error (MAE)
    
    MAE = (1/n) * sum(|prediction - ground_truth|)
    """
    if len(predictions) != len(ground_truth):
        raise ValueError(f"Length mismatch: predictions ({len(predictions)}) != ground_truth ({len(ground_truth)})")
    
    # Handle None values in predictions
    cleaned_predictions = []
    cleaned_ground_truth = []
    
    for p, t in zip(predictions, ground_truth):
        if p is not None:
            cleaned_predictions.append(p)
            cleaned_ground_truth.append(t)
    
    if len(cleaned_predictions) == 0:
        return float('inf')  # Return infinity if no valid predictions
    
    # Calculate mean absolute error
    mae = sum(abs(p - t) for p, t in zip(cleaned_predictions, cleaned_ground_truth)) / len(cleaned_predictions)
    return mae