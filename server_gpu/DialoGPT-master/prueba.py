
from lsp_model import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
from gpt2_training.train_utils import load_model
import torch
from transformers import AutoConfig


tokenizer = GPT2Tokenizer.from_pretrained("/mnt/homeGPU/mcarmona/server_gpu/DialoGPT-master/models/small")

config = AutoConfig.from_pretrained("/mnt/homeGPU/mcarmona/server_gpu/DialoGPT-master/models/small/config.json")


def fix_state_dict_namespace(model_state_dict):
    old_keys = []
    new_keys = []
    for t in model_state_dict:
        new_key = t
        if t.startswith('module.'):
            new_key = t.replace('module.', '')
        old_keys.append(t)
        new_keys.append(new_key)

    for old_key, new_key in zip(old_keys, new_keys):
        model_state_dict[new_key] = model_state_dict.pop(old_key)

    return model_state_dict

model_state_dict = torch.load("/mnt/homeGPU/mcarmona/server_gpu/DialoGPT-master/models/output_model/GPT2.1e-05.64.1gpu.2022-04-20010252/GP2-pretrain-step-10000.pkl")
model_state_dict = fix_state_dict_namespace(model_state_dict)

start_model = GPT2LMHeadModel(config)
start_model.load_state_dict(model_state_dict)

start_model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))


new_user_input_ids = tokenizer.encode(input(">> User:") + tokenizer.eos_token, return_tensors='pt')
# append the new user input tokens to the chat history
bot_input_ids = new_user_input_ids
# generated a response while limiting the total chat history to 1000 tokens, 
chat_history_ids = start_model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
# pretty print last ouput tokens from bot
print("DialoGPT_MWOZ_Bot: {}".format(tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)))















