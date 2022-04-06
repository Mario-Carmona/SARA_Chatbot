#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import pandas as pd
import sys
import os

import torch

from transformers import AutoConfig, AutoTokenizer, MarianMTModel, PegasusForConditionalGeneration, TranslationPipeline, SummarizationPipeline


WORKDIR = "/mnt/homeGPU/mcarmona/"

local_rank = int(os.getenv("LOCAL_RANK", "0"))



def traducirES_EN(dataset, es_en_translator):
    aux = {}

    for column in dataset.columns.values:
        content = es_en_translator(dataset[column].to_list())
        content = [i["translation_text"] for i in content]

        aux[column] = content

    dataset = pd.DataFrame(aux)

    return dataset

def summarization(dataset, summaryPipeline):
    pass

def generarDatasetAdulto(dataset):

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

    es_en_translator = TranslationPipeline(
        model=modelTrans_ES_EN,
        tokenizer=tokenizerTrans_ES_EN,
        framework="pt",
        device=local_rank
    )



    configSum = AutoConfig.from_pretrained(
        WORKDIR + "google/pegasus-xsum/config.json"
    )

    tokenizerSum = AutoTokenizer.from_pretrained(
        WORKDIR + "google/pegasus-xsum",
        config=WORKDIR + "google/pegasus-xsum/tokenizer_config.json",
        use_fast=True
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    modelSum = PegasusForConditionalGeneration.from_pretrained(
        WORKDIR + "google/pegasus-xsum",
        from_tf=bool(".ckpt" in "google/pegasus-xsum"),
        config=configSum,
        torch_dtype=torch.float16
    ).to(device)

    """
    summaryPipeline = SummarizationPipeline(
        model=modelSum,
        tokenizer=tokenizerSum,
        framework="pt",
        device=local_rank
    )
    """










    groups = dataset.groupby(dataset.Topic)

    groups_values = dataset.Topic.to_list()

    groups_datasets = [groups.get_group(value) for value in groups_values]

    groups_datasets = [traducirES_EN(i, es_en_translator) for i in groups_datasets]

    groups_datasets = [traducirES_EN(i, es_en_translator) for i in groups_datasets]

    print(groups_datasets[0].Text.to_list()[0])
    
    src_text = groups_datasets[0].Text.to_list()[0]
    batch = tokenizerSum(src_text, truncation=True, padding="longest", return_tensors="pt").to(device)
    translated = modelSum.generate(**batch, min_length=1, max_length=29, num_beams=1, num_return_sequences=1)
    tgt_text = tokenizerSum.batch_decode(translated, skip_special_tokens=True)

    #result = summaryPipeline(groups_datasets[-2].Text.to_list()[0], min_length=1, max_length=29, num_beams=8, num_return_sequences=8, n_docs=4)
    print(tgt_text)

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

    dataset = dataset.drop(columns=["Unnamed: 0"])

    if args.dataset_type == "Adulto":
        train_source, train_target, val_source, val_target = generarDatasetAdulto(dataset)


