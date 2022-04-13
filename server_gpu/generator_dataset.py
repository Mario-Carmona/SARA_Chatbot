#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Script para generar datasets para el entrenamiento """

# General
import os
from pathlib import Path
from tqdm.auto import tqdm
from color import bcolors
from typing import List
import numpy as np

# Configuración
import sys
import argparse
from dataclass.generate_dataset_arguments import GenerateDatasetArguments
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

# -------------------------------------------------------------------------#

os.environ["TOKENIZERS_PARALLELISM"] = "false"

set_seed(0)

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
        GenerateDatasetArguments
    )
)

generate_args, = parser.parse_json_file(json_file=str(BASE_PATH/CONFIG_FILE))

device = "cuda" if torch.cuda.is_available() else "cpu"



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
    config=configSum,
    torch_dtype=torch.float16
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
    config=configGenQues,
    torch_dtype=torch.float16
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


def obtenerTrainDataset(groups_datasets: List[DataFrame], train_split: float):
    """
    Función para obtener el dataset de training a partir de un
    dataset general y el porcentaje de contenido que se debe extraer
    del dataset

    Args:
        groups_datasets (List[DataFrame])
        train_split (float)
    
    Returns:
        DataFrame
    """

    # Lista que contendrá la partición de training de cada dataset
    lista = []

    # Obtención de las particiones de training de cada dataset, de esta 
    # forma el dataset para training mantendrá la proporción de datos de 
    # cada Topic que existe en el dataset original
    for dataset in groups_datasets:
        lista.append(dataset.sample(
            frac=train_split,
            random_state=generate_args.seed,
            axis=0,
            ignore_index=True
        ))

    # Unión de las particiones de training
    train_dataset = pd.concat(lista)

    # Barajar el dataset de training
    train_dataset = train_dataset.sample(
        frac=1,
        random_state=generate_args.seed,
        axis=0,
        ignore_index=True
    )

    return train_dataset


def obtenerValidationDataset(dataset: DataFrame, train_dataset: DataFrame):
    """
    Función para obtener el dataset de validation a partir de un
    dataset general y el dataset de training

    Args:
        dataset (DataFrame)
        train_dataset (DataFrame)
    
    Returns:
        DataFrame
    """

    # Obtener la diferencia entre el dataset completo y el dataset de training
    validation_dataset = pd.merge(dataset, train_dataset, how='outer', indicator='Exist')
    validation_dataset = validation_dataset.loc[validation_dataset['Exist'] != 'both']
    validation_dataset = validation_dataset.drop(["Exist"], axis=1)
    
    # Barajar el dataset de validation
    validation_dataset = validation_dataset.sample(
        frac=1,
        random_state=generate_args.seed,
        axis=0,
        ignore_index=True
    )

    return validation_dataset


def save_dataset_EN(groups_datasets):
    total_dataset = pd.concat(groups_datasets)
    name_file = '.'.join(generate_args.dataset_file.split('.')[:-1]) + "_EN.csv"
    total_dataset.to_csv(name_file)


def save_dataset_train_valid(groups_datasets, dir):
    # Unión de todos los datasets
    total_dataset = pd.concat(groups_datasets)

    # Obtención del dataset de training
    train_dataset = obtenerTrainDataset(groups_datasets, generate_args.train_split)

    # Generación del dataset de training con el formato para el entrenamiento
    train_s_t = pd.DataFrame({
        "source": train_dataset.Question.to_list(),
        "target": train_dataset.Answer.to_list()
    })

    # Obtención del dataset de validation
    validation_dataset = obtenerValidationDataset(total_dataset, train_dataset)

    # Generación del dataset de validation con el formato para el entrenamiento
    validation_s_t = pd.DataFrame({
        "source": validation_dataset.Question.to_list(),
        "target": validation_dataset.Answer.to_list()
    })

    # Guardado de todos los datasets en el directorio de destino
    train_dataset.to_csv(f"{dir}/train_resume.csv")
    train_s_t.to_csv(f"{dir}/train.csv")
    validation_dataset.to_csv(f"{dir}/validation_resume.csv")
    validation_s_t.to_csv(f"{dir}/validation.csv")


def split_by_topic(dataset: DataFrame):
    groups = dataset.groupby(dataset.Topic)
    groups_values = dataset.Topic.to_list()
    groups_datasets = [groups.get_group(value) for value in groups_values]

    return groups_datasets


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


def generarResumenes(text):
    """
    Función para generar los posibles resumenes de un texto

    Args:
        text (str)
    
    Returns:
        List[str]
    """

    # Obtener los tokens del texto
    batch = tokenizerSum(text, padding="longest", return_tensors="pt")
    
    # Si el número de tokens es menor al límite para resumir, se divide el texto
    # en las distinas frases que lo componen; sino se procede a generar los
    # resúmenes a partir del texto
    if(batch['input_ids'].shape[1] <= generate_args.limit_summary):
        resumenes = split(text, ". ")
    else:
        batch.to(device)
        translated = modelSum.generate(**batch, max_length=generate_args.max_length_summary, num_beams=generate_args.num_beams_summary, num_return_sequences=generate_args.num_beams_summary)
        tgt_text = tokenizerSum.batch_decode(translated, skip_special_tokens=True)

        # Eliminación de las frases repetidas
        
        resumenes = unique(tgt_text)
        print(resumenes)

    return resumenes


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
        text = []

        for i, j in zip(dataset.Text.to_list(), dataset.Subject.to_list()):                
            with torch.no_grad():
                resumenes = generarResumenes(i)
            resumenes = removeEmpty(resumenes)
            text += resumenes
            subject += [j] * len(resumenes)
            progress_bar.update(1)
            
        topic = [dataset.Topic.to_list()[0]] * len(text)

        if len(text) != 0:
            # Guardado del dataset tras la generación de resúmenes
            new_groups_datasets.append(pd.DataFrame({
                "Topic": topic,
                "Subject": subject,
                "Text": text
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
        answer = simplify_sentences(dataset.Answer.to_list(), model_name=generate_args.model_simplify)
        question = simplify_sentences(dataset.Question.to_list(), model_name=generate_args.model_simplify)

        progress_bar.update(len(dataset.Answer.to_list()))

        answer = removeEmpty(answer)
        question = removeEmpty(question)

        if len(question) != 0:
            new_groups_datasets.append(pd.DataFrame({
                "Topic": dataset.Topic.to_list(),
                "Subject": dataset.Subject.to_list(),
                "Question": question,
                "Answer": answer
            }))

    print(bcolors.OK + "Terminada simplificación" + bcolors.RESET)

    return new_groups_datasets


def generarDatasetAdulto(groups_datasets: List[DataFrame], dir: str):
    """
    Función para generar el conjunto de datasets para Adultos

    Args:
        dataset (DataFrame)
        dir (str)
    
    Returns:
        List[DataFrame]
    """

    save_dataset_train_valid(groups_datasets, dir)


def generarDatasetNiño(groups_datasets: List[DataFrame], dir: str):
    """
    Función para generar el conjunto de datasets para Niños

    Args:
        dataset (DataFrame)
        dir (str)
    
    Returns:
        List[DataFrame]
    """

    groups_datasets = simplify(groups_datasets)

    save_dataset_train_valid(groups_datasets, dir)






if __name__ == "__main__":

    # Lectura del dataset
    dataset = pd.read_csv(generate_args.dataset_file)

    # Eliminación de una columna que se añade al guardar el archivo CSV
    dataset = dataset.drop(columns=["Unnamed: 0"])

    # División del dataset en base al campo Topic
    groups_datasets = split_by_topic(dataset)

    if not generate_args.translated:
        # Traducción de los datasets al Inglés
        groups_datasets = traducirES_EN(groups_datasets)

        save_dataset_EN(groups_datasets)

    # Generación de las distintas respuestas mediante el resumen de los
    # textos de los distintos datasets
    groups_datasets = summarization(groups_datasets)

    # Generación de las preguntas a las respuestas obtenidas en el paso anterior
    groups_datasets = generateQuestions(groups_datasets)

    # Directorio donde guardar los resultados
    dirAdulto = os.path.join(generate_args.result_dir, f"split_{generate_args.train_split}_Adulto")
    # Creación del directorio en caso de que no exista
    if not os.path.exists(dirAdulto):
        os.mkdir(dirAdulto)

    generarDatasetAdulto(groups_datasets, dirAdulto)

    dirNiño = os.path.join(generate_args.result_dir, f"split_{generate_args.train_split}_Niño")
    # Creación del directorio en caso de que no exista
    if not os.path.exists(dirNiño):
        os.mkdir(dirNiño)

    generarDatasetNiño(groups_datasets, dirNiño)

