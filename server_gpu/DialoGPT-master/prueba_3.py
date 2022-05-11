
from lsp_model import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
from gpt2_training.train_utils import load_model
import torch
from os.path import join
from transformers import AutoTokenizer, AutoModelForCausalLM


from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration

mname = "/mnt/homeGPU/mcarmona/facebook/blenderbot-400M-distill"
model = BlenderbotForConditionalGeneration.from_pretrained(mname)
tokenizer = BlenderbotTokenizer.from_pretrained(mname)


"""
UTTERANCE = "My friends are cool but they eat too many carbs."
inputs = tokenizer([UTTERANCE], return_tensors="pt")
reply_ids = model.generate(**inputs)
print(tokenizer.batch_decode(reply_ids))
"""

for step in range(1):
    entry = input(">> User: ") + tokenizer.eos_token
    print(entry)
    new_user_input_ids = tokenizer.encode(entry, return_tensors='pt')
    print(tokenizer.decode(new_user_input_ids, skip_special_tokens=True))
    if step > 0:
      bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], axis=-1)  
    else:
      bot_input_ids = new_user_input_ids

    chat_history_ids = model.generate(bot_input_ids, max_length=1000, max_time=3.0, pad_token_id=tokenizer.eos_token_id)

    output = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    print("Blendetbot: {}".format(output))

