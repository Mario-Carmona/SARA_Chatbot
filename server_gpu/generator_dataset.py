#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import pandas as pd
import sys
import os

import torch

from transformers import AutoConfig, AutoTokenizer, MarianMTModel, PegasusForConditionalGeneration, TranslationPipeline, SummarizationPipeline
from transformers import AutoModelWithLMHead, AutoTokenizer


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

def unique(lista):
    lista_set = set(lista)
    unique_list = list(lista_set)
    return unique_list

def split(cadena, subcadena):
    lista = []

    print("\""+cadena+"\"")

    aux = cadena.find(subcadena)
    print(aux)
    while aux != -1:
        print(cadena[:aux+1])
        lista.append(cadena[:aux+1])
        cadena = cadena[aux+2:]
        print("\""+cadena+"\"")
        aux = cadena.find(subcadena)
        print(aux)
    
    lista.append(cadena)

    print(lista)

    return lista

def summarization(dataset, configSum, tokenizerSum, modelSum, device):
    text = []

    for i in dataset.Text.to_list():
        batch = tokenizerSum(i, truncation=True, padding="longest", return_tensors="pt")
        
        if(batch['input_ids'].shape[1] <= 50):
            frases = split(i, ". ")
            text += frases
        else:
            batch.to(device)
            translated = modelSum.generate(**batch, num_beams=configSum.num_beams, num_return_sequences=configSum.num_beams)
            tgt_text = tokenizerSum.batch_decode(translated, skip_special_tokens=True)
            text += unique(tgt_text)
        
    topic = dataset.Topic.to_list()[0] * len(text)

    dataset = pd.DataFrame({
        "Topic": topic,
        "Text": text
    })

    return dataset
    

def generarDatasetAdulto(dataset):

    def get_question(answer, context, max_length=64):
        input_text = "answer: %s  context: %s </s>" % (answer, context)
        features = tokenizer([input_text], return_tensors='pt').to(device)

        output = model.generate(input_ids=features['input_ids'], 
                    attention_mask=features['attention_mask'],
                    max_length=max_length)

        return tokenizer.decode(output[0])


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

    
    


    

    tokenizer = AutoTokenizer.from_pretrained(WORKDIR + "mrm8488/t5-base-finetuned-question-generation-ap")
    model = AutoModelWithLMHead.from_pretrained(WORKDIR + "mrm8488/t5-base-finetuned-question-generation-ap").to(device)



    






    groups = dataset.groupby(dataset.Topic)

    groups_values = dataset.Topic.to_list()

    groups_datasets = [groups.get_group(value) for value in groups_values]

    groups_datasets = [traducirES_EN(i, es_en_translator) for i in groups_datasets]

    groups_datasets = [summarization(i, configSum, tokenizerSum, modelSum, device) for i in groups_datasets]

    context = groups_datasets[0].Text.to_list()[1]
    answer = groups_datasets[0].Topic.to_list()[1]

    print(context)
    print(get_question(answer, context))



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


