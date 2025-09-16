from vllm import LLM, SamplingParams

# 1. initialize vllm engine
llm = LLM(
          model="Qwen/Qwen3-4B-Instruct-2507", 
          tensor_parallel_size=1, 
          max_model_len=2048,
          max_seq_len_to_capture=2048,
          gpu_memory_utilization=0.9,
        )

# 2. set sampling params
sampling_params = SamplingParams(
                   temperature=0.7,
                   top_p=0.95,
                   top_k=40,
                   max_tokens=128,
                   presence_penalty=0.0,
                   repetition_penalty=1.0,
                   stop=["\n", "#"]
                )
# 3. set prompts
prompts = [
    "what is the captial of france ?",
    "Explain transformer architecture in simple way",
    "What is 30 + 21"
]

# 3. generate sampling
outputs = llm.generate(sampling_params=sampling_params, prompts=prompts)

# 4. outputs 
for i, output in enumerate(outputs):
    print("-"*40)
    print(output.prompt)
    print(output.outputs[0].text)
    print(output.outputs[0].stop_reason)
    print("-"*40)

# 5. release model weights 
llm.llm_engine.engine_core.shutdown()