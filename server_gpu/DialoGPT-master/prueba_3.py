
from lsp_model import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
from gpt2_training.train_utils import load_model
import torch
from os.path import join
from transformers import AutoTokenizer, AutoModelForCausalLM

import math

from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration

mname = "/mnt/homeGPU/mcarmona/facebook/blenderbot-1B-distill"
model = BlenderbotForConditionalGeneration.from_pretrained(mname)
tokenizer = BlenderbotTokenizer.from_pretrained(mname)


"""
UTTERANCE = "My friends are cool but they eat too many carbs."
inputs = tokenizer([UTTERANCE], return_tensors="pt")
reply_ids = model.generate(**inputs)
print(tokenizer.batch_decode(reply_ids))
"""

context = []

for step in range(20):
    entry = tokenizer.cls_token + input(">> User: ") + tokenizer.eos_token
    print(entry)
    new_user_input_ids = tokenizer.encode(entry, return_tensors='pt')
    print(tokenizer.decode(new_user_input_ids[0]))
    
    context.append(new_user_input_ids)
    
    pos = -1
    num = 0
    while num <= 500 and math.abs(pos) <= len(context):
        num += len(context[pos][0])

        if num <= 500:
            pos -= 1

    context = context[pos+1:]


    bot_input_ids = torch.cat(context, axis=-1)  

    response = model.generate(bot_input_ids, max_length=1000, max_time=3.0, pad_token_id=tokenizer.eos_token_id)

    context.append(response)

    print("Blendetbot: {}".format(tokenizer.decode(response[0], skip_special_tokens=True)))

