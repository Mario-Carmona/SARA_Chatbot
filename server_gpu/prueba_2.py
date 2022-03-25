from transformers import AutoTokenizer, AutoConfig, GPTJForCausalLM
from transformers.deepspeed import HfDeepSpeedConfig
import deepspeed
import os
import torch

os.environ["TORCH_EXTENSIONS_DIR"] = "/mnt/homeGPU/mcarmona/torch_extensions"

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # To avoid warnings about parallelism in tokenizers

# distributed setup
local_rank = int(os.getenv("LOCAL_RANK", "0"))
world_size = int(os.getenv("WORLD_SIZE", "1"))


torch.cuda.set_device(local_rank)
deepspeed.init_distributed()

model_name = "../EleutherAI/gpt-j-6B"

model = GPTJForCausalLM.from_pretrained(model_name, revision=torch.float16, low_cpu_mem_usage=True)

with torch.no_grad():
    ds_engine = deepspeed.init_inference(model,
                                        mp_size=world_size,
                                        dtype=torch.float16,
                                        replace_method='auto',
					                    replace_with_kernel_inject=True)


tokenizer = AutoTokenizer.from_pretrained(model_name)

import time

inicio = time.time()
inputs = tokenizer.encode("DeepSpeed is", return_tensors="pt").to(device=local_rank)

with torch.no_grad():
    outputs = ds_engine.module.generate(inputs, do_sample=True, max_length = 100, min_length=100, synced_gpus=True)

text_out = tokenizer.decode(outputs[0], skip_special_tokens=True)

fin = time.time()

if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
    print(text_out)
print(fin-inicio)