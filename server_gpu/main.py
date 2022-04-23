#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Script para iniciar el servidor GPU """

# General
import os
from pathlib import Path
from typing import Dict, List
from color import bcolors

# Configuración
import sys
import argparse
from dataclass.server_arguments import ServerArguments
from transformers import HfArgumentParser

# Despliegue servidor
from fastapi import FastAPI
import uvicorn
from pyngrok import ngrok, conf

# Gestión peticiones
import requests
from pydantic import BaseModel
from fastapi.responses import PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware

# Modelos
import torch

from transformers import set_seed
from transformers import (
    AutoConfig, 
    AutoTokenizer,
    BlenderbotForConditionalGeneration,
    ConversationalPipeline, 
    Conversation
)

import deepl

import base64 
from PIL import Image
import io

# -------------------------------------------------------------------------#

class Entry(BaseModel):
    entry: str
    past_user_inputs: List[str]
    generated_responses: List[str]

class EntryDeduct(BaseModel):
    imagen: str


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
        ServerArguments
    )
)

server_args, = parser.parse_json_file(json_file=str(BASE_PATH/CONFIG_FILE))

WORKDIR = server_args.workdir



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
    server_args.model_conver_config
)


tokenizerConver = AutoTokenizer.from_pretrained(
    server_args.model_conver_tokenizer,
    config=server_args.model_conver_tokenizer_config,
    use_fast=True
)


modelConver = BlenderbotForConditionalGeneration.from_pretrained(
    server_args.model_conver,
    from_tf=bool(".ckpt" in server_args.model_conver),
    config=configConver,
    torch_dtype=torch.float16
)

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
        do_sample=server_args.do_sample,
        temperature=server_args.temperature,
        top_p=server_args.top_p,
        max_time=server_args.max_time,
        max_length=server_args.max_length,
        min_length=server_args.min_length,
        use_cache=server_args.use_cache
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



def deduct_age(image):
    pass


def send_public_URL():
    print(bcolors.WARNING + "Enviando URL al controlador..." + bcolors.RESET)
    url = server_args.controller_url
    headers = {'content-type': 'application/json'}
    response = requests.post(url + "/setURL", json={"url": public_url}, headers=headers)
    print(bcolors.OK + "INFO" + bcolors.RESET + ": " + str(response.content.decode('utf-8')))




app = FastAPI(version="1.0.0")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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

@app.get("/deduct", response_class=PlainTextResponse)
def deduct(imagen: str):

    #print(imagen)
    #age = deduct_age(data["image"])


    # convert it into bytes  
    img_bytes = base64.b64decode(imagen.encode('utf-8'))

    # convert bytes data to PIL Image object
    img = Image.open(io.BytesIO(img_bytes))

    imgAux = img.save("./prueba.jpeg")

    return "Nada"

@app.get("/Reconnect", response_class=PlainTextResponse)
def adulto():

    send_public_URL()

    return "Reenviada URL"


if __name__ == "__main__":

    port = eval(os.environ.get("PORT", server_args.port))

    pyngrok_config = conf.PyngrokConfig(
        ngrok_path=server_args.ngrok_path,
        config_path=server_args.ngrok_config_path
    )

    global public_url
    public_url = ngrok.connect(
        port, 
        pyngrok_config=pyngrok_config
    ).public_url
    # Convertir URL HTTP en HTTPS
    if public_url.split(':')[0] == "http":
        public_url = public_url.split(':')[0] + "s:" + public_url.split(':')[1]

    print(bcolors.OK + "Public URL" + bcolors.RESET + ": " + public_url)

    send_public_URL()

    # killall ngrok  → Para eliminar todas las sessiones de ngrok

    uvicorn.run(app, port=port)
