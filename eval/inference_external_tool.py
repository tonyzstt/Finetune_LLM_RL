import sys
import os
import argparse

from utils.countdown_utils import compute_score
from utils.generate_countdown_gt import generate_proposal
from eval_ultrafeedback import generate_response, get_model
from dataset.countdown import CountdownDataset

def inference_countdown():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, required=True)
    args = parser.parse_args()
    
    dataset = CountdownDataset(args.dataset_path)
    llm = get_model(args.model_path)
    scores = []
    
    for item in dataset:
        target = item["target"]
        numbers = item["numbers"]
        
        # use external tool to generate answer
        expr = generate_proposal(numbers, target)
        answer = expr if expr else ""
        
        # prompt with the answer
        prompt = f"<|im_start|>user\nUsing the numbers {numbers}, create an equation that equals {target}. Here is the answer: {answer}. You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer> <|im_end|>\n<|im_start|>assistant\n"
        
        if False:
            print(f"Prompt: {prompt}")
        
        # get the score
        response = generate_response(llm, prompt)
        score = compute_score(
            solution_str=response,
            ground_truth={"target": target, "numbers": numbers},
            method="strict",
            format_score=0.1,
            score=1.0
        )
        scores.append(score)
        
    print(f"Average score: {sum(scores) / len(scores)}")
    

if __name__ == "__main__":
    inference_countdown()