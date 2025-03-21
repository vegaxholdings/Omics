import os
import json
import argparse

from model import SiRNAModel
from data import SiRNADataset
from src.utils import set_seed, generate_predictions

def evaluate_model(model_path, data_path):
    """Evaluate the fine-tuned model on validation data using MAE"""
    print(f"Loading model from {model_path}...")
    # Initialize model
    sirna_model = SiRNAModel()
    model, tokenizer = sirna_model.load_trained_model(model_path)
    
    print(f"Loading validation data from {data_path}...")
    # Load validation dataset
    val_dataset = SiRNADataset(data_path, tokenizer=None)
    
    print("Generating predictions...")
    # Generate predictions
    predictions = generate_predictions(model, tokenizer, val_dataset)
    
    # Calculate metrics
    true_labels = val_dataset.get_labels()
    
    # Calculate MAE (primary metric)
    from data import calculate_mae
    mae = calculate_mae(predictions, true_labels)
    
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    
    # Save detailed results for analysis
    results = {
        "mae": mae,
        "predictions": [float(p) if p is not None else None for p in predictions],
        "true_labels": [float(l) for l in true_labels],
        "absolute_errors": [abs(float(p) - float(l)) if p is not None else None for p, l in zip(predictions, true_labels)]
    }
    
    results_path = os.path.join(os.path.dirname(model_path), "evaluation_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Detailed results saved to {results_path}")
    
    return mae

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the siRNA model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model")
    parser.add_argument("--data_path", type=str, default="data/test/validation.jsonl", 
                        help="Path to the validation data")
    parser.add_argument("--max_diff", type=float, default=10, 
                        help="Maximum difference threshold for correct predictions")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    set_seed(args.seed)
    evaluate_model(args.model_path, args.data_path, args.max_diff)