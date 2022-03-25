from transformers import AutoTokenizer, GPTJForCausalLM
from transformers.deepspeed import HfDeepSpeedConfig
import deepspeed
import os
import torch

local_rank = int(os.getenv("LOCAL_RANK", "0"))
world_size = int(os.getenv("WORLD_SIZE", "1"))
torch.cuda.set_device(local_rank)

model = GPTJForCausalLM.from_pretrained("../hivemind/gpt-j-6B-8bit", low_cpu_mem_usage=True)

ds_engine = deepspeed.initialize(model=model)[0]
ds_engine.module.eval()

tokenizer = AutoTokenizer.from_pretrained("../EleutherAI/gpt-j-6B")
inputs = tokenizer.encode("Esto es una prueba", return_tensors="pt").to(device=local_rank)

outputs = ds_engine.module.generate(inputs)
text_out = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(text_out)