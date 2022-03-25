from transformers import AutoTokenizer, AutoConfig, GPTJForCausalLM, pipeline
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

model_name = "/mnt/homeGPU/mcarmona/EleutherAI/gpt-j-6B"

generator = pipeline('text-generation', model=model_name, revision=torch.float16,
                     tokenizer=model_name, device=local_rank)

with torch.no_grad():
    generator.model = deepspeed.init_inference(generator.model,
                                        mp_size=world_size,
                                        dtype=torch.float16,
                                        replace_method='auto',
					                    replace_with_kernel_inject=True)


import time

inicio = time.time()


with torch.no_grad():
    string = generator("DeepSpeed is", do_sample=True, max_length = 100, min_length=100)

fin = time.time()

if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
    print(string)
print(fin-inicio)