

import torch
import numpy as np


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

context = []

for step in range(20):
    entry = tokenizer.cls_token + input(">> User: ")
    new_user_input_ids = tokenizer.encode(entry, return_tensors='pt')
    

    context.append(new_user_input_ids)

    aux = [np.array(i) for i in context]

    print(aux)

    context = [torch.as_tensor(i) for i in aux]

    print(context)
    
    
    pos = len(context) - 1
    num = 0
    while num <= 500 and pos >= 0:
        num += len(context[pos][0])

        if num <= 500:
            pos -= 1

    context = context[pos+1:]

    print("----")
    for i in context:
        print(tokenizer.decode(i[0], skip_special_tokens=False))
    print("----")

    bot_input_ids = torch.cat(context, axis=-1)  

    print(tokenizer.decode(bot_input_ids[0], skip_special_tokens=False))

    response = model.generate(bot_input_ids, max_length=128, pad_token_id=tokenizer.eos_token_id)

    context.append(response)

    print("Blendetbot: {}".format(tokenizer.decode(response[0], skip_special_tokens=True)))

