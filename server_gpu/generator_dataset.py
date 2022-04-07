#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import pandas as pd
import sys
import os
import deepl
from tqdm.auto import tqdm

import torch

from transformers import AutoConfig, AutoTokenizer, MarianMTModel, PegasusForConditionalGeneration, TranslationPipeline, SummarizationPipeline
from transformers import T5ForConditionalGeneration, AutoTokenizer


class bcolors:
    OK = '\033[92m' #GREEN
    WARNING = '\033[93m' #YELLOW
    FAIL = '\033[91m' #RED
    RESET = '\033[0m' #RESET COLOR


WORKDIR = "/mnt/homeGPU/mcarmona/"

local_rank = int(os.getenv("LOCAL_RANK", "0"))

device = "cuda" if torch.cuda.is_available() else "cpu"





def unique(lista):
    lista_set = set(lista)
    unique_list = list(lista_set)
    return unique_list

def split(cadena, subcadena):
    lista = []

    aux = cadena.find(subcadena)
    while aux != -1:
        lista.append(cadena[:aux+1])
        cadena = cadena[aux+2:]
        aux = cadena.find(subcadena)
    
    lista.append(cadena)

    return lista


def obtenerTrainDataset(groups_datasets, train_split):

    lista = []

    for dataset in groups_datasets:
        lista.append(dataset.sample(
            frac=train_split,
            random_state=0,
            axis=0,
            ignore_index=True
        ))

    train_dataset = pd.concat(lista)

    train_dataset = train_dataset.sample(
        frac=1,
        random_state=0,
        axis=0,
        ignore_index=True
    )

    return train_dataset


def obtenerValidationDataset(dataset, train_dataset):
    validation_dataset = pd.merge(dataset, train_dataset, how='outer', indicator='Exist')
    validation_dataset = validation_dataset.loc[validation_dataset['Exist'] != 'both']
    validation_dataset = validation_dataset.drop(["Exist"], axis=1)
    validation_dataset = validation_dataset.sample(
        frac=1,
        random_state=0,
        axis=0,
        ignore_index=True
    )

    return validation_dataset
    

def generarDatasetAdulto(dataset):

    def calculateElements(groups_datasets, num_columns=1):
        num = 0
        for i in groups_datasets:
            num_rows = len(i.Topic.to_list())
            num += num_columns * num_rows
        
        return num

    def traducirES_EN(groups_datasets):
        print(bcolors.WARNING + "Realizando traducción a Inglés..." + bcolors.RESET)

        progress_bar = tqdm(range(calculateElements(groups_datasets, 2)))
        
        new_groups_datasets = []

        for dataset in groups_datasets:
            new_dataset = {}

            new_dataset[dataset.columns.values[0]] = dataset[dataset.columns.values[0]].to_list()
            for column in dataset.columns.values[1:]:
                lista = []
                for text in dataset[column].to_list():
                    lista.append(translator.translate_text(text, target_lang="EN-US").text)
                    progress_bar.update(1)
                new_dataset[column] = lista

            new_groups_datasets.append(pd.DataFrame(new_dataset))

        print(bcolors.OK + "Terminada traducción" + bcolors.RESET)

        return new_groups_datasets

    def summarization(groups_datasets):

        print(bcolors.WARNING + "Realizando resumen del texto..." + bcolors.RESET)

        progress_bar = tqdm(range(calculateElements(groups_datasets)))

        new_groups_datasets = []

        for dataset in groups_datasets:
            subject = []
            text = []

            for i, j in zip(dataset.Text.to_list(), dataset.Subject.to_list()):
                batch = tokenizerSum(i, padding="longest", return_tensors="pt")
                
                if(batch['input_ids'].shape[1] <= 50):
                    frases = split(i, ". ")
                    text += frases
                    subject += [j] * len(frases)
                else:
                    batch.to(device)
                    translated = modelSum.generate(**batch, num_beams=configSum.num_beams, num_return_sequences=configSum.num_beams)
                    tgt_text = tokenizerSum.batch_decode(translated, skip_special_tokens=True)
                    aux = unique(tgt_text)
                    text += aux
                    subject += [j] * len(aux)
                
                progress_bar.update(1)
                
            topic = [dataset.Topic.to_list()[0]] * len(text)

            new_groups_datasets.append(pd.DataFrame({
                "Topic": topic,
                "Subject": subject,
                "Text": text
            }))

        print(bcolors.OK + "Terminado resumen" + bcolors.RESET)

        return new_groups_datasets

    def generateQuestions(groups_datasets):

        print(bcolors.WARNING + "Realizando generación de preguntas..." + bcolors.RESET)

        progress_bar = tqdm(range(calculateElements(groups_datasets)))

        new_groups_datasets = []

        for dataset in groups_datasets:
            answer = []
            question = []
            subject_list = []

            for subject, text in zip(dataset.Subject.to_list(), dataset.Text.to_list()):
                input_text = "answer: %s  context: %s </s>" % (subject, text)
                features = tokenizerGenQues(input_text, return_tensors='pt').to(device)
                output = modelGenQues.generate(
                    input_ids=features['input_ids'], 
                    attention_mask=features['attention_mask'],
                    max_length=64,
                    num_beams=4, num_return_sequences=4
                )
                result = [tokenizerGenQues.decode(output[i], skip_special_tokens=True) for i in range(len(output))]
                result = unique([i[10:] for i in result])

                answer += [text] * len(result)
                question += result
                subject_list += [subject] * len(result)

                progress_bar.update(1)

            topic = [dataset.Topic.to_list()[0]] * len(answer)

            new_groups_datasets.append(pd.DataFrame({
                "Topic": topic,
                "Subject": subject_list,
                "Question": question,
                "Answer": answer
            }))

        print(bcolors.OK + "Terminada generación de preguntas" + bcolors.RESET)

        return new_groups_datasets

    
    print(bcolors.WARNING + "Cargando modelos..." + bcolors.RESET)


    auth_key = "663c05c5-179a-a54d-9dd4-85bc4edcd925:fx"
    translator = deepl.Translator(auth_key)
    
    # ---------------------------------------

    configSum = AutoConfig.from_pretrained(
        WORKDIR + "google/pegasus-xsum/config.json"
    )

    tokenizerSum = AutoTokenizer.from_pretrained(
        WORKDIR + "google/pegasus-xsum",
        config=WORKDIR + "google/pegasus-xsum/tokenizer_config.json",
        use_fast=True
    )

    modelSum = PegasusForConditionalGeneration.from_pretrained(
        WORKDIR + "google/pegasus-xsum",
        from_tf=bool(".ckpt" in "google/pegasus-xsum"),
        config=configSum,
        torch_dtype=torch.float16
    ).to(device)

    # ---------------------------------------

    configGenQues = AutoConfig.from_pretrained(
        WORKDIR + "mrm8488/t5-base-finetuned-question-generation-ap/config.json"
    )

    tokenizerGenQues = AutoTokenizer.from_pretrained(
        WORKDIR + "mrm8488/t5-base-finetuned-question-generation-ap",
        config=WORKDIR + "mrm8488/t5-base-finetuned-question-generation-ap/tokenizer_config.json",
        use_fast=True
    )

    modelGenQues = T5ForConditionalGeneration.from_pretrained(
        WORKDIR + "mrm8488/t5-base-finetuned-question-generation-ap",
        from_tf=bool(".ckpt" in "mrm8488/t5-base-finetuned-question-generation-ap"),
        config=configGenQues,
        torch_dtype=torch.float16
    ).to(device)


    print(bcolors.OK + "Modelos cargados" + bcolors.RESET)



    groups = dataset.groupby(dataset.Topic)

    groups_values = dataset.Topic.to_list()

    groups_datasets = [groups.get_group(value) for value in groups_values]

    groups_datasets = traducirES_EN(groups_datasets)

    groups_datasets = summarization(groups_datasets)

    groups_datasets = generateQuestions(groups_datasets)

    return groups_datasets





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
        groups_datasets = generarDatasetAdulto(dataset)

    total_dataset = pd.concat(groups_datasets)

    train_dataset = obtenerTrainDataset(groups_datasets, args.train_split)

    train_s_t = pd.DataFrame({
        "source": train_dataset.Question.to_list(),
        "target": train_dataset.Answer.to_list()
    })

    validation_dataset = obtenerValidationDataset(total_dataset, train_dataset)

    validation_s_t = pd.DataFrame({
        "source": validation_dataset.Question.to_list(),
        "target": validation_dataset.Answer.to_list()
    })

    dir = os.path.join(args.result_dir, f"split_{args.train_split}")
    if not os.path.exists(dir):
        os.mkdir(dir)

    train_dataset.to_csv(f"{dir}/train_resume.csv")
    train_s_t.to_csv(f"{dir}/train.csv")
    validation_dataset.to_csv(f"{dir}/validation_resume.csv")
    validation_s_t.to_csv(f"{dir}/validation.csv")






