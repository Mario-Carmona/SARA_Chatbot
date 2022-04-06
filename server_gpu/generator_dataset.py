#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import pandas as pd
import sys
import os

import torch

from transformers import AutoConfig, AutoTokenizer, MarianMTModel, TranslationPipeline


WORKDIR = "/mnt/homeGPU/mcarmona/"

configTrans_ES_EN = AutoConfig.from_pretrained(
    WORKDIR + "Helsinki-NLP/opus-mt-es-en/config.json"
)

tokenizerTrans_ES_EN = AutoTokenizer.from_pretrained(
    WORKDIR + "Helsinki-NLP/opus-mt-es-en",
    config=WORKDIR + "Helsinki-NLP/opus-mt-es-en/tokenizer_config.json",
    use_fast=True
)

modelTrans_ES_EN = MarianMTModel.from_pretrained(
    WORKDIR + "Helsinki-NLP/opus-mt-es-en",
    from_tf=bool(".ckpt" in "Helsinki-NLP/opus-mt-es-en"),
    config=configTrans_ES_EN,
    torch_dtype=torch.float16
)

local_rank = int(os.getenv("LOCAL_RANK", "0"))

es_en_translator = TranslationPipeline(
    model=modelTrans_ES_EN,
    tokenizer=tokenizerTrans_ES_EN,
    framework="pt",
    device=local_rank
)



def traducirES_EN(text):
    return es_en_translator(text)[0]["translation_text"]

def generarDatasetAdulto(dataset):

    groups = dataset.groupby(dataset.Topic)

    groups_values = dataset.Topic.to_list()

    groups_datasets = [groups.get_group(value) for value in groups_values]

    groups_datasets = [i.apply(traducirES_EN) for i in groups_datasets]

    for i in groups_datasets:
        print(i.Text)

    return None, None, None, None





if __name__ == "__main__":

    WORKDIR = "/mnt/homeGPU/mcarmona/"

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "dataset_file", 
        type = str,
        help = "El formato del archivo debe ser \'archivo.csv\'"
    )

    parser.add_argument(
        "result_dir", 
        type = str,
        help = "Debe ser un directorio existe"
    )

    parser.add_argument(
        "train_split", 
        type = float,
        help = "De ser un flotante mayor que 0 y menor que 1"
    )

    parser.add_argument(
        "dataset_type", 
        type = str,
        help = "Debe ser algunas de las siguientes cadenas: 'Niño', 'Adulto'"
    )

    try:
        args = parser.parse_args()
        assert args.dataset_file.split('.')[-1] == "csv"
        assert os.path.exists(args.result_dir)
        assert args.train_split > 0.0 and args.train_split < 1.0
        assert args.dataset_type in ["Niño", "Adulto"]
    except:
        parser.print_help()
        sys.exit(0)


    dataset = pd.read_csv(args.dataset_file)

    if args.dataset_type == "Adulto":
        train_source, train_target, val_source, val_target = generarDatasetAdulto(dataset)

