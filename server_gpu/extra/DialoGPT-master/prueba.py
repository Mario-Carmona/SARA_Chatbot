
from lsp_model import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
from gpt2_training.train_utils import load_model
import torch
from os.path import join
from transformers import AutoTokenizer, AutoModelForCausalLM


#model_path = "/mnt/homeGPU/mcarmona/server_gpu/DialoGPT-master/models/medium"
model_path = "/mnt/homeGPU/mcarmona/tosin/prueba"

"""
tokenizer = GPT2Tokenizer.from_pretrained(
    model_path,
    vocab_file = join(model_path, 'vocab.json'),
    merges_file = join(model_path, 'merges.txt'),
    unk_token = "<|endoftext|>",
    bos_token = "<|endoftext|>",
    eos_token = "<|endoftext|>"
)
"""

tokenizer = GPT2Tokenizer.from_pretrained(model_path)

config = GPT2Config.from_json_file(
    join(model_path, 'config.json')
)

args = {
    "n_gpu": 1,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "fp16": True
}


model = GPT2LMHeadModel.from_pretrained(model_path)


"""
model = load_model(GPT2LMHeadModel(config), join(model_path, 'pytorch_model.bin'),
                   args, verbose=True)
"""





new_user_input_ids = tokenizer.encode(input(">> User:") + "<|endoftext|>")
# append the new user input tokens to the chat history
bot_input_ids = new_user_input_ids

print(bot_input_ids)

# generated a response while limiting the total chat history to 1000 tokens, 
chat_history_ids = model(**bot_input_ids, max_length=1000, pad_token_id="<|endoftext|>")

print(chat_history_ids)
"""
# pretty print last ouput tokens from bot
print("DialoGPT_MWOZ_Bot: {}".format(tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)))
"""








