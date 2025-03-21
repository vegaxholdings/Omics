import argparse

from model import SiRNAModel
from data import SiRNADataset
from src.utils import set_seed

def train_model(train_data_path, output_dir, batch_size=4, gradient_accumulation_steps=8,
                num_epochs=3, learning_rate=2e-4, r=16, lora_alpha=32, lora_dropout=0.05,
                max_steps=-1):
    """Train the siRNA model"""
    print(f"Loading model...")
    # Initialize model
    sirna_model = SiRNAModel()
    model, tokenizer = sirna_model.load_model(use_4bit=True)
    sirna_model.prepare_for_training(r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
    
    print(f"Loading training data from {train_data_path}...")
    # Load dataset
    train_dataset = SiRNADataset(train_data_path, tokenizer=tokenizer)
    
    print(f"Starting training with {len(train_dataset)} samples for {num_epochs} epochs...")
    # Train model
    sirna_model.train(
        train_dataset=train_dataset,
        output_dir=output_dir,
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        max_steps=max_steps
    )
    
    print(f"Training completed. Model saved to {output_dir}")
    
    return output_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the siRNA model")
    parser.add_argument("--train_data", type=str, default="data/test/train.jsonl", 
                        help="Path to the training data")
    parser.add_argument("--output_dir", type=str, default="./trained_model", 
                        help="Directory to save the model")
    parser.add_argument("--batch_size", type=int, default=4, 
                        help="Training batch size")
    parser.add_argument("--grad_accum_steps", type=int, default=8, 
                        help="Gradient accumulation steps")
    parser.add_argument("--num_epochs", type=int, default=3, 
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=2e-4, 
                        help="Learning rate")
    parser.add_argument("--r", type=int, default=16, 
                        help="LoRA attention dimension")
    parser.add_argument("--lora_alpha", type=int, default=32, 
                        help="LoRA alpha parameter")
    parser.add_argument("--lora_dropout", type=float, default=0.05, 
                        help="LoRA dropout rate")
    parser.add_argument("--max_steps", type=int, default=None, 
                        help="Maximum number of training steps (overrides num_epochs)")
    parser.add_argument("--seed", type=int, default=42, 
                        help="Random seed")
    
    args = parser.parse_args()
    
    set_seed(args.seed)
    train_model(
        args.train_data,
        args.output_dir,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        num_epochs=args.num_epochs,
        learning_rate=args.lr,
        r=args.r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        max_steps=args.max_steps
    )