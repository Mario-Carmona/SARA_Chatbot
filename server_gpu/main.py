#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import json
import uvicorn
import os
import requests
from pathlib import Path
from pyngrok import ngrok, conf
from pydantic import BaseModel





import sys
import time
import argparse
import logging

import torch

from project_arguments import ProyectArguments
from model_arguments import ModelArguments
from inference_arguments import InferenceArguments
from transformers import TrainingArguments, HfArgumentParser



from datasets import load_dataset, load_metric

from transformers import set_seed
from transformers.trainer_utils import is_main_process
from transformers import AutoTokenizer, GPTJConfig, GPTJForCausalLM, pipeline, Conversation

import deepspeed

import transformers





class bcolors:
    OK = '\033[92m' #GREEN
    WARNING = '\033[93m' #YELLOW
    FAIL = '\033[91m' #RED
    RESET = '\033[0m' #RESET COLOR

class Entry(BaseModel):
    entry: str





logger = logging.getLogger(__name__)




parser = HfArgumentParser(
    (
        ProyectArguments, 
        ModelArguments, 
        InferenceArguments, 
        TrainingArguments
    )
)


project_args, model_args, infer_args, training_args = parser.parse_json_file(json_file="config_inference.json")


WORKDIR = project_args.workdir


with open(WORKDIR + model_args.generate_args_path) as file:
    generate_args = json.load(file)


# Ruta donde instalar las extensiones de Pytorch
os.environ["TORCH_EXTENSIONS_DIR"] = WORKDIR + "torch_extensions"

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # To avoid warnings about parallelism in tokenizers



# distributed setup
local_rank = int(os.getenv("LOCAL_RANK", "0"))
world_size = int(os.getenv("WORLD_SIZE", "1"))


torch.cuda.set_device(local_rank)
deepspeed.init_distributed()


# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)
logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)



# Log on each process the small summary:
logger.warning(
    f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
    + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
)
# Set the verbosity to info of the Transformers logger (on main process only):
if is_main_process(training_args.local_rank):
    transformers.utils.logging.set_verbosity_info()
logger.info("Training/evaluation parameters %s", training_args)



 # Set seed before initializing model.

set_seed(training_args.seed)



# Load pretrained model and tokenizer
#
# Distributed training:
# The .from_pretrained methods guarantee that only one local process can concurrently
# download model & vocab.

"""
config = GPTJConfig.from_pretrained(
    WORKDIR + model_args.model_config_name if model_args.model_config_name else WORKDIR + model_args.model_name_or_path,
    task_specific_params=generate_args
)


tokenizer = AutoTokenizer.from_pretrained(
    WORKDIR + model_args.tokenizer_name if model_args.tokenizer_name else WORKDIR + model_args.model_name_or_path,
    config=WORKDIR + model_args.tokenizer_config_name if model_args.tokenizer_config_name else None,
    use_fast=True
)


model = GPTJForCausalLM.from_pretrained(
    WORKDIR + model_args.model_name_or_path,
    from_tf=bool(".ckpt" in model_args.model_name_or_path),
    config=config
)
"""

generator = pipeline(
    "conversational",
    model=WORKDIR + model_args.model_name_or_path,
    config=WORKDIR + model_args.model_config_name if model_args.model_config_name else WORKDIR + model_args.model_name_or_path,
    tokenizer=WORKDIR + model_args.tokenizer_name if model_args.tokenizer_name else WORKDIR + model_args.model_name_or_path,
    framework="pt",
    device=local_rank,
    use_fast=True
)


os.system("nvidia-smi")


if infer_args.do_inference:
    with torch.no_grad():
        """
        generator.model = deepspeed.init_inference(
            generator.model,
            mp_size=world_size,
            dtype=torch.int8,
            replace_method=infer_args.replace_method,
            replace_with_kernel_inject=infer_args.replace_with_kernel_inject,
            quantization_setting=(
                infer_args.mlp_exra_grouping,
                infer_args.quantize_groups
            )
        )
        """

        conversation = Conversation()

        os.system("nvidia-smi")











app = FastAPI(version="1.0.0")

BASE_PATH = Path(__file__).resolve().parent

@app.get("/", response_class=PlainTextResponse)
def home():
    return "Server GPU ON"

@app.post("/Adulto", response_class=PlainTextResponse)
def adulto(request: Entry):

    conversation.add_user_input(request.entry)

    generator([conversation])


    return conversation.generated_responses[-1]



if __name__ == "__main__":

    with open(str(BASE_PATH/"config.json")) as file:
        config = json.load(file)

    port = eval(os.environ.get("PORT", config["port"]))

    pyngrok_config = conf.PyngrokConfig(
        ngrok_path=WORKDIR + "mcarmona/bin/ngrok",
        config_path=WORKDIR + "server_gpu/ngrok.yml"
    )

    public_url = ngrok.connect(
        port, 
        pyngrok_config=pyngrok_config
    ).public_url

    print(bcolors.OK + "Public URL" + bcolors.RESET + ": " + public_url)

    print(bcolors.WARNING + "Enviando URL al controlador..." + bcolors.RESET)
    url = config["controller_url"]
    headers = {'content-type': 'application/json'}
    response = requests.post(url + "/setURL", json={"url": public_url}, headers=headers)
    print(bcolors.OK + "INFO" + bcolors.RESET + ": " + str(response.content.decode('utf-8')))

    # killall ngrok  → Para eliminar todas las sessiones de ngrok

    uvicorn.run(app, port=port)
