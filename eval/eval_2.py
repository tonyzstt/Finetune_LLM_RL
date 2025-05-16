from vllm import LLM, SamplingParams

# Example code using vllm to run peft models
# First generate the model checkpoint using 'python zero_to_fp32.py . .'
# Then copy the original model config into the check point dir 
# TODO: add dataloader and reward model

model_dir = "/home/tonyzst/Desktop/CS224R-Project/DPO/ckpts/dpo_peft/last"      

llm = LLM(
    model=model_dir,
    dtype="float16",          
    tensor_parallel_size=1,   
    trust_remote_code=True,   
)

prompts = [
    "Explain why the sky is blue in one short paragraph."
]

sampler = SamplingParams(max_tokens=64, temperature=0.7)
outputs = llm.generate(prompts, sampler)

for request_output in outputs:
    print("Prompt:", request_output.prompt)
    print("Generated:", request_output.outputs[0].text.strip())
    print("-" * 40)
