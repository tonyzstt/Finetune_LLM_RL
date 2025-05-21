from vllm import LLM, SamplingParams
from openai import OpenAI
from config import API_KEY
import os
import sys
import argparse
from tqdm import tqdm
from dataset.ultrafeedback import UltraFeedbackDataset

# Example code using vllm to run peft models
# First generate the model checkpoint using 'python zero_to_fp32.py . .'
# Then copy the original model config into the check point dir 

def get_model(model_dir):
    """
    Get the model from the model directory
    """
    llm = LLM(
        model=model_dir,
        dtype="float16",          
        tensor_parallel_size=1,   
        trust_remote_code=True,   
    )

    return llm

def get_client():
    """
    Get the client from the API key
    """
    return OpenAI(
        base_url = "https://integrate.api.nvidia.com/v1",
        api_key = API_KEY
    )

def generate_response(llm, prompt):
    """
    Generate the response from the model
    """
    sampler = SamplingParams(max_tokens=1280)
    outputs = llm.generate(prompt, sampler)
    return outputs[0].outputs[0].text.strip()

def get_reward(client, prompt, response):
    """
    Get the reward from the client
    """
    completion = client.chat.completions.create(
        model ="nvidia/llama-3.1-nemotron-70b-reward",
        messages =[
            {"role": "user", "content": prompt} ,
            {"role": "assistant", "content": response}
        ],
    )

    return completion.choices[0].message.content

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, required=True)
    args = parser.parse_args()

    dataset = UltraFeedbackDataset(args.dataset_path)

    client = get_client()
    llm = get_model(args.model_path)
    rewards = []

    for item in tqdm(dataset):

        prompt = item["prompt"]
        response = generate_response(llm, prompt)
        reward = get_reward(client, prompt, response)

        reward = float(reward.split(":")[-1])
        rewards.append(reward)

    print(f"Average reward: {sum(rewards) / len(rewards)}")


    
