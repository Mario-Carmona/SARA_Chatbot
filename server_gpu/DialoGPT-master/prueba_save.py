
from lsp_model import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
from gpt2_training.train_utils import load_model
import torch
from os.path import join
from transformers import AutoTokenizer, AutoModelForCausalLM


model_path = "/mnt/homeGPU/mcarmona/tosin/dialogpt_mwoz"

model = AutoModelForCausalLM.from_pretrained(model_path)



torch.save(
            {k: (v.cpu() if v is not None else None)  # save to cpu tensors
                for k, v in model.state_dict().items()},
            "/mnt/homeGPU/mcarmona/tosin/prueba/pytorch_model.bin")



