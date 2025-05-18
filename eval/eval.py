from vllm import LLM, SamplingParams
from openai import OpenAI
from config import API_KEY

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
    sampler = SamplingParams(max_tokens=64, temperature=0.7)
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

    model_dir = "/home/tonyzst/Desktop/CS224R-Project/DPO/ckpts/dpo_peft/last" 
    prompts = [
        "Explain why the sky is blue in one short paragraph."
    ]     

    client = get_client()
    llm = get_model(model_dir)
    responses = [generate_response(llm, prompt) for prompt in prompts]
    rewards = [get_reward(client, prompt, response) for prompt, response in zip(prompts, responses)]
