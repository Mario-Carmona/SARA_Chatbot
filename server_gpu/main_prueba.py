#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Script para iniciar el servidor GPU """

# General
import os
from pathlib import Path
from typing import Dict, List
from color import bcolors
import numpy as np

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
    Conversation,
    ViTFeatureExtractor,
    ViTForImageClassification
)

import deepl

import base64 
from PIL import Image
import io

import deepspeed


from uuid import UUID

# -------------------------------------------------------------------------#

class Entry(BaseModel):
    entry: str
    conver_id: UUID
    last_response: bool

class EntryDeduct(BaseModel):
    imagen: str


parser = argparse.ArgumentParser()

parser.add_argument(
    "--config_file", 
    type = str,
    help = "El formato del archivo debe ser \'config.json\'"
)

parser.add_argument(
    "--local_rank", 
    type = int
)

args = parser.parse_args()

"""
try:
    args = parser.parse_args()
    assert args.config_file.split('.')[-1] == "json"
except:
    parser.print_help()
    sys.exit(0)
"""

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

deepspeed.init_distributed()


# Set seed before initializing model.

set_seed(0)



# Load pretrained model and tokenizer
#
# Distributed training:
# The .from_pretrained methods guarantee that only one local process can concurrently
# download model & vocab.

configConverAdult = AutoConfig.from_pretrained(
    server_args.model_conver_adult_config
)

configConverChild = AutoConfig.from_pretrained(
    server_args.model_conver_child_config
)


tokenizerConverAdult = AutoTokenizer.from_pretrained(
    server_args.model_conver_adult_tokenizer,
    config=server_args.model_conver_adult_tokenizer_config,
    use_fast=True
)

tokenizerConverChild = AutoTokenizer.from_pretrained(
    server_args.model_conver_child_tokenizer,
    config=server_args.model_conver_child_tokenizer_config,
    use_fast=True
)


modelConverAdult = BlenderbotForConditionalGeneration.from_pretrained(
    server_args.model_conver_adult,
    from_tf=bool(".ckpt" in server_args.model_conver_adult),
    config=configConverAdult
)

modelConverChild = BlenderbotForConditionalGeneration.from_pretrained(
    server_args.model_conver_child,
    from_tf=bool(".ckpt" in server_args.model_conver_child),
    config=configConverChild
)



pipelineConverAdult = ConversationalPipeline(
    model = modelConverAdult,
    tokenizer = tokenizerConverAdult,
    framework = "pt",
    device = local_rank
)

pipelineConverChild = ConversationalPipeline(
    model = modelConverChild,
    tokenizer = tokenizerConverChild,
    framework = "pt",
    device = local_rank
)


# ----------------------------------------------



auth_key = "663c05c5-179a-a54d-9dd4-85bc4edcd925:fx"
translator = deepl.Translator(auth_key)




pipelineConverAdult.model = deepspeed.init_inference(
    pipelineConverAdult.model,
    mp_size=world_size,
    replace_method='auto',
    replace_with_kernel_inject=True
)

pipelineConverChild.model = deepspeed.init_inference(
    pipelineConverChild.model,
    mp_size=world_size,
    replace_method='auto',
    replace_with_kernel_inject=True
)




os.system("nvidia-smi")

dicc_conversation = {}




def make_response_adult(entry: str, conver_id: str, last_response: bool):

    entry_EN = translator.translate_text(entry, source_lang="ES", target_lang="EN-US").text

    print(entry_EN)


    global dicc_conversation

    try:
        conversation = dicc_conversation[conver_id]
    except KeyError:
        conversation = Conversation()


    conversation.add_user_input(entry_EN)

    print(conversation)

    output = pipelineConverAdult(
        conversation,
        do_sample=server_args.do_sample,
        temperature=server_args.temperature,
        top_p=server_args.top_p,
        max_time=server_args.max_time,
        max_length=server_args.max_length,
        min_length=server_args.min_length,
        use_cache=server_args.use_cache,
        pad_token_id=tokenizerConverAdult.eos_token_id,
        synced_gpus=True
    )

    print(output)

    print(output.uuid)
    print(type(output.uuid))

    if last_response:
        del dicc_conversation[output.uuid]
    else:
        dicc_conversation[output.uuid] = output


    answer_EN = output.generated_responses[-1]

    print(answer_EN)


    answer = translator.translate_text(answer_EN, source_lang="EN", target_lang="ES").text

    print(answer)

    response = {
        "entry": {
            "ES": entry, 
            "EN": entry_EN
        },
        "answer": {
            "ES": answer, 
            "EN": answer_EN
        },
        "conver_id": output.uuid
    }

    return response


# deepspeed --num_gpus 1 main_prueba.py --config_file configs/config_server_prueba.json



def make_response_child(entry: str, conver_id: str, last_response: bool):

    entry_EN = translator.translate_text(entry, source_lang="ES", target_lang="EN-US").text

    print(entry_EN)


    global dicc_conversation

    try:
        conversation = dicc_conversation[conver_id]
    except KeyError:
        conversation = Conversation()


    conversation.add_user_input(entry_EN)

    print(conversation)

    output = pipelineConverChild(
        conversation,
        do_sample=server_args.do_sample,
        temperature=server_args.temperature,
        top_p=server_args.top_p,
        max_time=server_args.max_time,
        max_length=server_args.max_length,
        min_length=server_args.min_length,
        use_cache=server_args.use_cache,
        pad_token_id=tokenizerConverAdult.eos_token_id,
        synced_gpus=True
    )

    print(output)

    if last_response:
        del dicc_conversation[output.uuid]
    else:
        dicc_conversation[output.uuid] = output


    answer_EN = output.generated_responses[-1]

    print(answer_EN)


    answer = translator.translate_text(answer_EN, source_lang="EN", target_lang="ES").text

    print(answer)

    response = {
        "entry": {
            "ES": entry, 
            "EN": entry_EN
        },
        "answer": {
            "ES": answer, 
            "EN": answer_EN
        },
        "conver_id": output.uuid
    }

    return response






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

@app.post("/adult")
def adult(request: Entry):


    response = make_response_adult(
        request.entry, 
        request.conver_id,
        request.last_response
    )

    return response

@app.post("/child")
def child(request: Entry):

    response = make_response_child(
        request.entry, 
        request.conver_id,
        request.last_response
    )

    return response

@app.post("/deduct", response_class=PlainTextResponse)
def deduct(request: EntryDeduct):


    base64_data = request.imagen.split(',')[1]

    # convert it into bytes  
    img_bytes = base64.b64decode(base64_data)

    # convert bytes data to PIL Image object
    img = Image.open(io.BytesIO(img_bytes))


    # Init model, transforms
    model = ViTForImageClassification.from_pretrained('/mnt/homeGPU/mcarmona/nateraw/vit-age-classifier')
    transforms = ViTFeatureExtractor.from_pretrained('/mnt/homeGPU/mcarmona/nateraw/vit-age-classifier')

    # Transform our image and pass it through the model
    inputs = transforms(img, return_tensors='pt')
    output = model(**inputs)

    # Predicted Class probabilities
    proba = output.logits.softmax(1)

    # Predicted Classes
    preds = proba.argmax(1)

    """
    Los índices empiezan en 0
    0-2
    3-9
    10-19
    20-29
    30-39
    40-49
    50-59
    60-69
    more than 70
    """

    if int(preds) <= 1:
        return "child"
    else:
        return "adult"


@app.get("/Reconnect", response_class=PlainTextResponse)
def reconnect():

    send_public_URL()

    return "Reenviada URL"


if __name__ == "__main__":

    rank = torch.distributed.get_rank()
    if rank == 0:
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
