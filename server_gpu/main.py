#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import requests
from pathlib import Path

from project_arguments import ProyectArguments
from model_arguments import ModelArguments
from generate_arguments import GenerateArguments
from transformers import HfArgumentParser

from pydantic import BaseModel

from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
import uvicorn

from pyngrok import ngrok, conf

import torch

from transformers import set_seed
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer, AutoModelForSeq2SeqLM, TranslationPipeline, TextGenerationPipeline, Conversation






class bcolors:
    OK = '\033[92m' #GREEN
    WARNING = '\033[93m' #YELLOW
    FAIL = '\033[91m' #RED
    RESET = '\033[0m' #RESET COLOR

class Entry(BaseModel):
    entry: str





BASE_PATH = Path(__file__).resolve().parent
CONFIG_FILE = "config_server.json"


parser = HfArgumentParser(
    (
        ProyectArguments, 
        ModelArguments, 
        GenerateArguments
    )
)


project_args, model_args, generate_args = parser.parse_json_file(json_file=str(BASE_PATH/CONFIG_FILE))


WORKDIR = project_args.workdir



# Ruta donde instalar las extensiones de Pytorch
os.environ["TORCH_EXTENSIONS_DIR"] = WORKDIR + "torch_extensions"

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # To avoid warnings about parallelism in tokenizers



# distributed setup
local_rank = int(os.getenv("LOCAL_RANK", "0"))
world_size = int(os.getenv("WORLD_SIZE", "1"))

torch.cuda.set_device(local_rank)


# Set seed before initializing model.

set_seed(0)



# Load pretrained model and tokenizer
#
# Distributed training:
# The .from_pretrained methods guarantee that only one local process can concurrently
# download model & vocab.

configConver = AutoConfig.from_pretrained(
    WORKDIR + model_args.model_conver_config
)


tokenizerConver = AutoTokenizer.from_pretrained(
    WORKDIR + model_args.model_conver_tokenizer,
    use_fast=True
)


modelConver = AutoModelForCausalLM.from_pretrained(
    WORKDIR + model_args.model_conver,
    from_tf=bool(".ckpt" in model_args.model_conver),
    config=configConver
)

# ----------------------------------------------

configTrans_ES_EN = AutoConfig.from_pretrained(
    WORKDIR + model_args.model_trans_ES_EN_config
)

tokenizerTrans_ES_EN = AutoTokenizer.from_pretrained(
    WORKDIR + model_args.model_trans_ES_EN_tokenizer,
    config=WORKDIR + model_args.model_trans_ES_EN_tokenizer_config,
    use_fast=True
)

modelTrans_ES_EN = AutoModelForSeq2SeqLM.from_pretrained(
    WORKDIR + model_args.model_trans_ES_EN,
    from_tf=bool(".ckpt" in model_args.model_trans_ES_EN),
    config=configTrans_ES_EN,
    torch_dtype=torch.float16
)

# ----------------------------------------------

configTrans_EN_ES = AutoConfig.from_pretrained(
    WORKDIR + model_args.model_trans_EN_ES_config
)

tokenizerTrans_EN_ES = AutoTokenizer.from_pretrained(
    WORKDIR + model_args.model_trans_EN_ES_tokenizer,
    config=WORKDIR + model_args.model_trans_EN_ES_tokenizer_config,
    use_fast=True
)

modelTrans_EN_ES = AutoModelForSeq2SeqLM.from_pretrained(
    WORKDIR + model_args.model_trans_EN_ES,
    from_tf=bool(".ckpt" in model_args.model_trans_EN_ES),
    config=configTrans_EN_ES,
    torch_dtype=torch.float16
)

# ----------------------------------------------

es_en_translator = TranslationPipeline(
    model=modelTrans_ES_EN,
    tokenizer=tokenizerTrans_ES_EN,
    framework="pt",
    device=local_rank
)

# ----------------------------------------------

en_es_translator = TranslationPipeline(
    model=modelTrans_EN_ES,
    tokenizer=tokenizerTrans_EN_ES,
    framework="pt",
    device=local_rank
)

# ----------------------------------------------

pipelineConversation = TextGenerationPipeline(
    model=modelConver,
    tokenizer=tokenizerConver,
    framework="pt",
    device=local_rank
)

conversation = Conversation()

# ----------------------------------------------

os.system("nvidia-smi")






def make_response_Adulto(entry: str):

    #entry_EN = es_en_translator(entry)[0]["translation_text"]

    #print(entry_EN)

    #conversation.add_user_input(entry)

    response = pipelineConversation(
        f"Conversación entre [A] y [B]\n[A]: {entry}\n[B]: ",
        do_sample=generate_args.do_sample,
        temperature=generate_args.temperature,
        top_p=generate_args.top_p,
        max_time=generate_args.max_time,
        max_length=generate_args.max_length,
        use_cache=generate_args.use_cache
    )["generated_text"]

    #conversation.mark_processed()

    #print(response_EN)

    #response = en_es_translator(response_EN)[0]["translation_text"]

    print(response)

    return response





app = FastAPI(version="1.0.0")

@app.get("/", response_class=PlainTextResponse)
def home():
    return "Server GPU ON"

@app.post("/Adulto", response_class=PlainTextResponse)
def adulto(request: Entry):

    response = make_response_Adulto(request.entry)

    return response



if __name__ == "__main__":

    port = eval(os.environ.get("PORT", project_args.port))

    pyngrok_config = conf.PyngrokConfig(
        ngrok_path=WORKDIR + project_args.ngrok_path,
        config_path=WORKDIR + project_args.ngrok_config_path
    )

    public_url = ngrok.connect(
        port, 
        pyngrok_config=pyngrok_config
    ).public_url

    print(bcolors.OK + "Public URL" + bcolors.RESET + ": " + public_url)

    print(bcolors.WARNING + "Enviando URL al controlador..." + bcolors.RESET)
    url = project_args.controller_url
    headers = {'content-type': 'application/json'}
    response = requests.post(url + "/setURL", json={"url": public_url}, headers=headers)
    print(bcolors.OK + "INFO" + bcolors.RESET + ": " + str(response.content.decode('utf-8')))

    # killall ngrok  → Para eliminar todas las sessiones de ngrok

    uvicorn.run(app, port=port)
