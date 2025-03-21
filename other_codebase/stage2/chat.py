import argparse

import torch

from model import SiRNAModel
from src.utils import set_seed, extract_number_from_text

def format_query(query):
    """Format user query for model input"""
    # Check if the query already contains RNA and DNA tags
    if "<rna>" in query.lower() and "<dna>" in query.lower():
        return query
    
    # If not, guide the user to provide properly formatted input
    if "<rna>" not in query.lower() or "<dna>" not in query.lower():
        print("\nNote: Your query should include RNA and DNA sequences in the format:")
        print("<rna>RNA SEQUENCE<rna> <dna>DNA SEQUENCE<dna> Your question\n")
        return query

    return query

def chat_with_model(model_path):
    """Interactive CLI to chat with the trained model"""
    print(f"Loading model from {model_path}...")
    # Initialize model
    sirna_model = SiRNAModel()
    model, tokenizer = sirna_model.load_trained_model(model_path)
    
    # 토크나이저 패딩 설정 확인 및 수정
    if tokenizer.padding_side != "left":
        print("토크나이저 패딩 설정을 left로 변경합니다.")
        tokenizer.padding_side = "left"
    
    print("\n===== ChatsiRNA 대화형 인터페이스 =====")
    print("모델과 대화를 시작합니다. 종료하려면 'exit' 또는 'quit'를 입력하세요.")
    print("입력 형식: <rna>RNA 시퀀스<rna> <dna>DNA 시퀀스<dna> 질문")
    print("예시: <rna>a Uf a g dC c g u a<rna> <dna>CAGTGACAG<dna> What outcome is expected in terms of mRNA remaining after siRNA treatment?")
    print("=" * 40 + "\n")
    
    while True:
        # Get user input
        user_input = input("\n사용자: ")
        
        # Check for exit command
        if user_input.lower() in ["exit", "quit", "종료"]:
            print("대화를 종료합니다.")
            break
        
        # Format user input
        user_input = format_query(user_input)
        
        # Format input with instruction format
        formatted_input = f"[INST] {user_input} [/INST]"
        
        # Tokenize input
        inputs = tokenizer(formatted_input, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Generate response
        print("\n생성 중...", end="\r")
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=True,   
                temperature=0.7,  
                top_p=0.9,       
                num_beams=1,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
        
        # Decode response
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        
        # Extract response part (after [/INST])
        response = generated_text.split("[/INST]")[-1].strip()
        
        # Extract numerical prediction
        predicted_value = extract_number_from_text(response)
        
        # Print response
        print(f"ChatsiRNA: {response}")
        if predicted_value is not None:
            print(f"\n추출된 예측값: {predicted_value:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chat with the trained siRNA model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    set_seed(args.seed)
    chat_with_model(args.model_path)