import json
import openai
from typing import Dict, List, Tuple
import os
from dotenv import load_dotenv

load_dotenv()

def get_client():
    """
    Get the client from the API key
    """
    return OpenAI(
        base_url = "https://integrate.api.nvidia.com/v1",
        api_key = API_KEY
    )

gpt_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
reward_client = openai.OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("NVIDIA_API_KEY")
)

def read_prompts(file_path: str) -> List[str]:
    """Read prompts from a JSON file."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data['prompts']

def generate_responses(prompt: str) -> Tuple[str, str]:
    """Generate good and bad responses using GPT-4."""
    good_response = gpt_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Generate a high-quality, informative response."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=1024
    ).choices[0].message.content

    bad_response = gpt_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Generate a low-quality, unhelpful response."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=1024
    ).choices[0].message.content

    return good_response, bad_response

def evaluate_response(prompt: str, response: str) -> float:

    completion = reward_client.chat.completions.create(
        model="nvidia/llama-3.1-nemotron-70b-reward",
        messages=[
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response}
        ]
    )
    
    return completion.choices[0].message.content.strip()

def main():

    prompts = read_prompts("prompts.json")
    
    results = []
    for prompt in prompts:

        good_response, bad_response = generate_responses(prompt)
        good_score = evaluate_response(prompt, good_response)
        bad_score = evaluate_response(prompt, bad_response)
        good_score = float(good_score.split(":")[1].strip())
        bad_score = float(bad_score.split(":")[1].strip())
        
        if good_score <= bad_score or good_score < -10:
            continue
        
        results.append({
            "prompt": prompt,
            "good_response": good_response,
            "bad_response": bad_response,
            "good_score": good_score,
            "bad_score": bad_score
        })
    
    with open("generated_responses.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
