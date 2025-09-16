from vllm import LLM, SamplingParams, sampling_params

# 1. initiate the vLLM inference engine 
llm = LLM(
    model="Qwen/Qwen3-0.6B",      # Model name from hf / local path 
    tensor_parallel_size= 1,      # We are using one gpu
    max_seq_len_to_capture= 32768, # Qwen's Context window
)

# 2. Define parameters for controlled generation
sampling_params = SamplingParams(
    temperature=0.7,        # Controls randomness: 0.0 = deterministic, > 0.7 = creative
    top_p=0.95,             # Nuclues sampling: cumulative probability threshold
    top_k=40,               # Top-k sampling: consider only top k tokens
    max_tokens=128,         # Max new tokens to generate
    presence_penalty=0.0,   # Penalizes new tokens based on whether they appear in generated text so far
    repetition_penalty=1.0, # Penalized new tokens based on whether they appear in prompt and generated txt so far
    stop=["\n", "#"]        # Stops generation when these strings are generated
)

# 3. Define your input prompts
prompts = [
    "Explain the concept of Relativity in simple terms",
    "Write a short poem about the ocean",
    "What is a Neural network ?"
]

# 4. Generate outputs
outputs = llm.generate(prompts=prompts, sampling_params=sampling_params)

# Inspect the outputs
for i, output in enumerate(outputs):
    print(f"prompt {i + 1}: {output.prompt}")
    print(f"Generated text: {output.outputs[0].text}")
    print(f"Finish reason: {output.outputs[0].finish_reason}")
    print("-" * 50)

# Accessing detailed properties
first_output = outputs[0]
completion = first_output.outputs[0] # Get the first  completion for the first prompt

print("Generated text: ", completion.text)
print("Token IDs: ", completion.token_ids)
print("Cumulative log probability: ",  completion.cumulative_logprob)
print("Finish reason: ", completion.finish_reason) # e.g "stop", "length"
