import torch 
from vllm import LLM

# check how many GPUs vLLM can detect
if torch.cuda.is_available():
    num_gpus = torch.cuda.device_count()
    print(f"Number of available GPUs {num_gpus}")
else:
    print("CUDA is not available. Running on CPU.")
# try to initialize a tiny model (this test CUDA compatibility)
# we use a small model for this test to be quick and use little VRAM.
try:
    llm = LLM(model="microsoft/phi-2") 
    print("SUCCESS: GPU and CUDA are compatible with vLLM!")
except Exception as e:
    print(f"Error: {e}")
    