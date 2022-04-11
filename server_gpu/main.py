#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Script para iniciar el servidor GPU """

# General
import os
from pathlib import Path
from typing import List
from color import bcolors

# Configuración
import sys
import argparse
from dataclass.project_arguments import ProyectArguments
from dataclass.model_arguments import ModelArguments
from dataclass.generate_arguments import GenerateArguments
from transformers import HfArgumentParser

# Despliegue servidor
from fastapi import FastAPI
import uvicorn
from pyngrok import ngrok, conf

# Gestión peticiones
import requests
from pydantic import BaseModel
from fastapi.responses import PlainTextResponse

# Modelos
import torch

from transformers import set_seed
from transformers import (
    AutoConfig, 
    AutoTokenizer, 
    AutoModelForSeq2SeqLM, 
    MarianMTModel, 
    BlenderbotForConditionalGeneration, 
    TranslationPipeline, 
    ConversationalPipeline, 
    Conversation
)

import deepl

# -------------------------------------------------------------------------#

class Entry(BaseModel):
    entry: str
    past_user_inputs: List[str]
    generated_responses: List[str]



parser = argparse.ArgumentParser()

parser.add_argument(
    "config_file", 
    type = str,
    help = "El formato del archivo debe ser \'config.json\'"
)

try:
    args = parser.parse_args()
    assert args.config_file.split('.')[-1] == "json"
except:
    parser.print_help()
    sys.exit(0)

BASE_PATH = Path(__file__).resolve().parent
CONFIG_FILE = args.config_file


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
    config=WORKDIR + model_args.model_conver_tokenizer_config,
    use_fast=True
)


modelConver = BlenderbotForConditionalGeneration.from_pretrained(
    WORKDIR + model_args.model_conver,
    from_tf=bool(".ckpt" in model_args.model_conver),
    config=configConver,
    torch_dtype=torch.float16
)

# ----------------------------------------------

"""

configTrans_ES_EN = AutoConfig.from_pretrained(
    WORKDIR + model_args.model_trans_ES_EN_config
)

tokenizerTrans_ES_EN = AutoTokenizer.from_pretrained(
    WORKDIR + model_args.model_trans_ES_EN_tokenizer,
    config=WORKDIR + model_args.model_trans_ES_EN_tokenizer_config,
    use_fast=True
)

modelTrans_ES_EN = MarianMTModel.from_pretrained(
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

modelTrans_EN_ES = MarianMTModel.from_pretrained(
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

"""

# ----------------------------------------------

pipelineConversation = ConversationalPipeline(
    model=modelConver,
    tokenizer=tokenizerConver,
    framework="pt",
    device=local_rank
)

# ----------------------------------------------

os.system("nvidia-smi")

auth_key = "663c05c5-179a-a54d-9dd4-85bc4edcd925:fx"
translator = deepl.Translator(auth_key)




def make_response_Adulto(entry: str, past_user_inputs: List[str], generated_responses: List[str]):

    entry_EN = translator.translate_text(entry, target_lang="EN-US").text

    print(entry_EN)

    conversation = Conversation(
        entry_EN,
        past_user_inputs=past_user_inputs,
        generated_responses=generated_responses
    )

    pipelineConversation(
        conversation,
        do_sample=generate_args.do_sample,
        temperature=generate_args.temperature,
        top_p=generate_args.top_p,
        max_time=generate_args.max_time,
        max_length=generate_args.max_length,
        min_length=generate_args.min_length,
        use_cache=generate_args.use_cache
    )

    answer_EN = conversation.generated_responses[-1]

    print(answer_EN)

    answer = translator.translate_text(answer_EN, target_lang="ES").text

    print(answer)

    response = {
        "entry": {
            "ES": entry, 
            "EN": entry_EN
        },
        "answer": {
            "ES": answer, 
            "EN": answer_EN
        }
    }

    return response



def send_public_URL():
    print(bcolors.WARNING + "Enviando URL al controlador..." + bcolors.RESET)
    url = project_args.controller_url
    headers = {'content-type': 'application/json'}
    response = requests.post(url + "/setURL", json={"url": public_url}, headers=headers)
    print(bcolors.OK + "INFO" + bcolors.RESET + ": " + str(response.content.decode('utf-8')))




app = FastAPI(version="1.0.0")

@app.get("/", response_class=PlainTextResponse)
def home():
    return "Server GPU ON"

@app.post("/Adulto")
def adulto(request: Entry):

    response = make_response_Adulto(
        request.entry, 
        request.past_user_inputs,
        request.generated_responses
    )

    return response


@app.get("/Reconnect", response_class=PlainTextResponse)
def adulto():

    send_public_URL()

    return "Reenviada URL"


if __name__ == "__main__":

    port = eval(os.environ.get("PORT", project_args.port))

    pyngrok_config = conf.PyngrokConfig(
        ngrok_path=WORKDIR + project_args.ngrok_path,
        config_path=WORKDIR + project_args.ngrok_config_path
    )

    global public_url
    public_url = ngrok.connect(
        port, 
        pyngrok_config=pyngrok_config
    ).public_url

    print(bcolors.OK + "Public URL" + bcolors.RESET + ": " + public_url)

    send_public_URL()

    # killall ngrok  → Para eliminar todas las sessiones de ngrok

    uvicorn.run(app, port=port)
