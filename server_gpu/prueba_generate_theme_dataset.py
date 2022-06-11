#!/usr/bin/env python3
# -*- coding: utf-8 -*-

## 
# @file generate_theme_dataset.py
#
# @brief Script para la generación de datasets temáticos
# 
# @section libraries_main Libraries/Modules

# General
import os
from pathlib import Path
from tqdm.auto import tqdm
from color import bcolors
from typing import List
import numpy as np
from utils import save_csv

# Configuración
import sys
import argparse
from dataclass.theme_dataset_arguments import ThemeDatasetArguments
from transformers import HfArgumentParser

# Datos
import pandas as pd
from pandas import DataFrame

# Modelos
import torch

from transformers import (
    AutoConfig, 
    AutoTokenizer, 
    PegasusForConditionalGeneration,
    T5ForConditionalGeneration
)

from transformers import set_seed

from SentenceSimplification.muss.simplify import simplify_sentences

import deepl

from split_dataset import (
    split_by_topic
)

# -------------------------------------------------------------------------#

parser = argparse.ArgumentParser()

parser.add_argument(
    "config_file", 
    type = str,
    help = "El formato del archivo debe ser 'config.json'"
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
        ThemeDatasetArguments
    )
)

generate_args, = parser.parse_json_file(json_file=str(BASE_PATH/CONFIG_FILE))

device = "cuda" if torch.cuda.is_available() else "cpu"

os.environ["TOKENIZERS_PARALLELISM"] = "false"

set_seed(generate_args.seed)

print(bcolors.WARNING + "Cargando modelos..." + bcolors.RESET)

# ---------------------------------------
# ----- Traductor

auth_key = generate_args.auth_key_deepl
translator = deepl.Translator(auth_key)

# ---------------------------------------
# ----- Modelo summarization

configSum = AutoConfig.from_pretrained(
    generate_args.model_summary_config
)

tokenizerSum = AutoTokenizer.from_pretrained(
    generate_args.model_summary_tokenizer,
    config=generate_args.model_summary_tokenizer_config,
    use_fast=True
)

modelSum = PegasusForConditionalGeneration.from_pretrained(
    generate_args.model_summary,
    from_tf=bool(".ckpt" in generate_args.model_summary),
    config=configSum
).to(device)

# ---------------------------------------
# ----- Modelo generate question

configGenQues = AutoConfig.from_pretrained(
    generate_args.model_genQuestion_config
)

tokenizerGenQues = AutoTokenizer.from_pretrained(
    generate_args.model_genQuestion_tokenizer,
    config=generate_args.model_genQuestion_tokenizer_config,
    use_fast=True
)

modelGenQues = T5ForConditionalGeneration.from_pretrained(
    generate_args.model_genQuestion,
    from_tf=bool(".ckpt" in generate_args.model_genQuestion),
    config=configGenQues
).to(device)


print(bcolors.OK + "Modelos cargados" + bcolors.RESET)



def removeEmpty(lista: List[str]):
    return [i for i in lista if i != ""]


def unique(lista: List[str]):
    """ 
    Función para eliminar elementos repetidos en una lista 
    
    Args:
        lista (List[str])

    Returns:
        List[str]
    """

    lista_set = set(lista)
    unique_list = list(lista_set)

    return unique_list


def split(cadena: str, subcadena: str):
    """
    Función para dividir una cadena mediante una subcadena

    Args:
        cadena (str)
        subcadena (str)
    
    Returns:
        List[str]
    """

    # Lista que contendrá las divisiones de la cadena
    lista = []

    # Se busca la subcadena
    posSubCad = cadena.find(subcadena)
    # Mientras se siga encontrando la subcadena se sigue dividiendo la cadena
    while posSubCad != -1:
        # Se añade la sección dividida a la lista
        lista.append(cadena[:posSubCad+1])
        cadena = cadena[posSubCad+2:]
        posSubCad = cadena.find(subcadena)
    
    lista.append(cadena)

    return lista


def save_dataset_EN(groups_datasets):
    total_dataset = pd.concat(groups_datasets)
    filename = '.'.join(generate_args.initial_dataset_file.split('.')[:-1]) + "_EN.csv"
    save_csv(total_dataset, filename)


def calculateElements(groups_datasets: List[DataFrame], num_columns: int=1):
    """
    Función para calcular el número de elementos en base a las 
    filas y a las columnas indicadas

    Args:
        groups_datasets (List[DataFrame])
        num_columns (int)
    
    Returns:
        int
    """

    num = 0
    for i in groups_datasets:
        num_rows = len(i.Topic.to_list())
        num += num_columns * num_rows
    
    return num


def traducirES_EN(groups_datasets):
    """
    Función para traducir del Español al Inglés una lista de dataframes

    Args:
        groups_datasets (List[DataFrame])
    
    Returns:
        List[DataFrame]
    """

    print(bcolors.WARNING + "Realizando traducción a Inglés..." + bcolors.RESET)

    # Creación de la barra de progreso
    progress_bar = tqdm(range(calculateElements(groups_datasets, 2)))
    
    # Lista que contendrá los datasets después de la traducción
    new_groups_datasets = []

    # Traducción del contenido de cada dataset
    for dataset in groups_datasets:
        # Dataset que tendrá sus columnas traducidas
        new_dataset = {}

        # La primera columna se copia sin traducir
        new_dataset[dataset.columns.values[0]] = dataset[dataset.columns.values[0]].to_list()
        # Traducción y guardado del resto de columnas
        for column in dataset.columns.values[1:]:
            lista = []
            for text in dataset[column].to_list():
                lista.append(translator.translate_text(text, target_lang="EN-US").text)
                # Actualización de la barra de progreso
                progress_bar.update(1)
            new_dataset[column] = lista

        # Guardado del dataset traducido
        new_groups_datasets.append(pd.DataFrame(new_dataset))

    print(bcolors.OK + "Terminada traducción" + bcolors.RESET)

    return new_groups_datasets


def generarResumenes(question, answer):
    """
    Función para generar los posibles resumenes de un texto

    Args:
        text (str)
    
    Returns:
        List[str]
    """

    # Obtener los tokens del texto
    batch_question = tokenizerSum(question, max_length=512, padding="longest", return_tensors="pt")
    
    batch_question.to(device)

    try:
        translated_question = modelSum.generate(**batch_question, max_length=generate_args.max_length_summary, num_beams=generate_args.num_beams_summary, num_return_sequences=generate_args.num_beams_summary)
        tgt_text_question = tokenizerSum.batch_decode(translated_question, skip_special_tokens=True)

        # Eliminación de las frases repetidas
        resumenes_question = unique(tgt_text_question)
    except RuntimeError:
        resumenes_question = None

    ####################


    # Obtener los tokens del texto
    batch_answer = tokenizerSum(answer, max_length=512, padding="longest", return_tensors="pt")
    
    batch_answer.to(device)

    try:
        translated_answer = modelSum.generate(**batch_answer, max_length=generate_args.max_length_summary, num_beams=generate_args.num_beams_summary, num_return_sequences=generate_args.num_beams_summary)
        tgt_text_answer = tokenizerSum.batch_decode(translated_answer, skip_special_tokens=True)

        # Eliminación de las frases repetidas
        resumenes_answer = unique(tgt_text_answer)
    except RuntimeError:
        resumenes_answer = None

    return resumenes_question, resumenes_answer


def summarization(groups_datasets):
    """
    Función para generar las respuestas en base al resumen de los textos de los datasets

    Args:
        groups_datasets (List[DataFrame])
    
    Returns:
        List[DataFrame]
    """

    print(bcolors.WARNING + "Realizando resumen del texto..." + bcolors.RESET)


    # Creación de la barra de progreso
    progress_bar = tqdm(range(calculateElements(groups_datasets)))

    # Lista que contendrá los datasets después de la obtención de los resúmenes
    new_groups_datasets = []

    # Generación de los resúmenes del contenido de cada dataset
    for dataset in groups_datasets:
        subject = []
        question = []
        answer = []

        for i, j, k in zip(dataset.Question.to_list(), dataset.Subject.to_list(), dataset.Answer.to_list()):  
            if str(type(i)) == "<class 'str'>" and str(type(k)) == "<class 'str'>":              
                with torch.no_grad():
                    resumenes_question, resumenes_answer = generarResumenes(i, k)

                if resumenes_question != None and resumenes_answer != None:
                    resumenes_question = removeEmpty(resumenes_question)
                    resumenes_answer = removeEmpty(resumenes_answer)

                    question += resumenes_question
                    answer += resumenes_answer
                    subject += [j] * len(resumenes_question)

            progress_bar.update(1)
            
        topic = [dataset.Topic.to_list()[0]] * len(question)

        if len(question) != 0:
            # Guardado del dataset tras la generación de resúmenes
            new_groups_datasets.append(pd.DataFrame({
                "Topic": topic,
                "Subject": subject,
                "Question": question,
                "Answer": answer
            }))

    print(bcolors.OK + "Terminado resumen" + bcolors.RESET)

    return new_groups_datasets


def generarQuestions(subject, text):
    """
    Función para generar las preguntas hacia un sujeto sobre una cierta respuesta  

    Args:
        subject (str)
        text (str)
    
    Returns:
        List[str]
    """
    
    # Formatear texto de entrada
    input_text = "answer: %s  context: %s </s>" % (subject, text)
    # Obtención de las preguntas
    features = tokenizerGenQues(input_text, return_tensors='pt').to(device)
    output = modelGenQues.generate(
        input_ids=features['input_ids'], 
        attention_mask=features['attention_mask'],
        max_length=generate_args.max_length_question,
        num_beams=generate_args.num_beams_question, num_return_sequences=generate_args.num_beams_question
    )
    result = [tokenizerGenQues.decode(output[i], skip_special_tokens=True) for i in range(len(output))]
    # Eliminación de las preguntas repetidas
    result = unique([i[10:] for i in result])

    return result


def generateQuestions(groups_datasets):
    """
    Función para generar las preguntas de las respuestas 

    Args:
        groups_datasets (List[DataFrame])
    
    Returns:
        List[DataFrame]
    """

    print(bcolors.WARNING + "Realizando generación de preguntas..." + bcolors.RESET)

    progress_bar = tqdm(range(calculateElements(groups_datasets)))

    new_groups_datasets = []

    for dataset in groups_datasets:
        answer = []
        question = []
        subject_list = []

        for subject, text in zip(dataset.Subject.to_list(), dataset.Text.to_list()):
            with torch.no_grad():
                result = generarQuestions(subject, text)
            result = removeEmpty(result)

            answer += [text] * len(result)
            question += result
            subject_list += [subject] * len(result)

            progress_bar.update(1)

        topic = [dataset.Topic.to_list()[0]] * len(answer)

        if len(question) != 0:
            new_groups_datasets.append(pd.DataFrame({
                "Topic": topic,
                "Subject": subject_list,
                "Question": question,
                "Answer": answer
            }))

    print(bcolors.OK + "Terminada generación de preguntas" + bcolors.RESET)

    return new_groups_datasets


def simplify(groups_datasets):
    """
    
    """

    print(bcolors.WARNING + "Realizando simplificación..." + bcolors.RESET)

    progress_bar = tqdm(range(calculateElements(groups_datasets)))

    new_groups_datasets = []

    for dataset in groups_datasets:
        answer = []
        question = []

        for a, q in zip(dataset.Answer.to_list(), dataset.Question.to_list()):
            with torch.no_grad():
                a_simplify, q_simplify = simplify_sentences([a, q], model_name=generate_args.model_simplify)
            
            if a_simplify != "" and q_simplify != "":
                answer.append(a_simplify)
                question.append(q_simplify)

            progress_bar.update(1)

        if len(question) != 0:
            new_groups_datasets.append(pd.DataFrame({
                "Topic": dataset.Topic.to_list(),
                "Subject": dataset.Subject.to_list(),
                "Question": question,
                "Answer": answer
            }))

    print(bcolors.OK + "Terminada simplificación" + bcolors.RESET)

    return new_groups_datasets



def generate_theme_dataset(dataset):
    # División del dataset en base al campo Topic
    groups_datasets = split_by_topic(dataset)

    if not generate_args.translated:
        # Traducción de los datasets al Inglés
        groups_datasets = traducirES_EN(groups_datasets)

        save_dataset_EN(groups_datasets)

    # Generación de las distintas respuestas mediante el resumen de los
    # textos de los distintos datasets
    groups_datasets = summarization(groups_datasets)

    adult_dataset = pd.concat(groups_datasets)

    groups_datasets = simplify(groups_datasets)

    child_dataset = pd.concat(groups_datasets)

    return adult_dataset, child_dataset



if __name__ == "__main__":

    # Lectura del dataset
    dataset = pd.read_csv(generate_args.initial_dataset_file)

    adult_dataset, child_dataset = generate_theme_dataset(dataset)

    save_csv(adult_dataset, os.path.join(generate_args.theme_result_dir, generate_args.adult_dataset_file))
    save_csv(child_dataset, os.path.join(generate_args.theme_result_dir, generate_args.child_dataset_file))
