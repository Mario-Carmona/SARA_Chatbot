#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

from project_arguments import ProyectArguments
from model_arguments import ModelArguments
from transformers import HfArgumentParser

from transformers import AutoTokenizer



BASE_PATH = Path(__file__).resolve().parent
CONFIG_FILE = "config_finetuning.json"


parser = HfArgumentParser(
    (
        ProyectArguments, 
        ModelArguments
    )
)

project_args, model_args = parser.parse_json_file(json_file=str(BASE_PATH/CONFIG_FILE))


WORKDIR = project_args.workdir


tokenizerConver = AutoTokenizer.from_pretrained(
    WORKDIR + model_args.model_conver_tokenizer,
    config=WORKDIR + model_args.model_conver_tokenizer_config,
    use_fast=True
)



def preprocess(dataset):
    pass























