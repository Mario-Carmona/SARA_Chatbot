#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""! @brief Script para la división de un conjunto de datos."""


##
# @file generate_theme_dataset.py
#
# @brief Programa para la división de un conjunto de datos.
#
# @section description_main Descripción
# Programa para la división de un conjunto de datos.
#
# @section libraries_main Librerías/Módulos
# - Librería estándar os (https://docs.python.org/3/library/os.html)
#   - Acceso a la función path.join
#   - Acceso a la función environ
# - Librería estándar pathlib (https://docs.python.org/3/library/pathlib.html)
#   - Acceso a la función Path
# - Librería tqdm.auto (https://tqdm.github.io/)
#   - Acceso a la función tqdm
# - Librería color
#   - Acceso a la clase bcolors
# - Librería estándar typing (https://docs.python.org/3/library/typing.html)
#   - Acceso a la clase List
# - Librería utils
#   - Acceso a la función save_csv
# - Librería estándar sys (https://docs.python.org/3/library/sys.html)
#   - Acceso a la función exit
# - Librería estándar argparse (https://docs.python.org/3/library/argparse.html)
#   - Acceso a la función ArgumentParser
# - Librería dataclass.theme_dataset_arguments
#   - Acceso a la clase ThemeDatasetArguments
# - Librería transformers (https://pypi.org/project/transformers)
#   - Acceso a la función HfArgumentParser
#   - Acceso a la función set_seed
#   - Acceso a la clase AutoConfig
#   - Acceso a la clase AutoTokenizer
#   - Acceso a la clase PegasusForConditionalGeneration
#   - Acceso a la clase T5ForConditionalGeneration
# - Librería pandas (https://pandas.pydata.org/docs/)
#   - Acceso a la función DataFrame
#   - Acceso a la función concat
# - Librería torch (https://pypi.org/project/torch/) 
#   - Acceso a la función cuda.is_available
#   - Acceso a la función no_grad
# - Librería SentenceSimplification.muss.simplify
#   - Acceso a la función simplify_sentences
# - Librería deepl (https://www.deepl.com/es/docs-api/) 
#   - Acceso a la clase Translator
# - Librería split_dataset
#   - Acceso a la función split_by_topic
#
# @section author_doxygen_example Autor
# - Created by Mario Carmona Segovia.
#
# Copyright (c) 2022.  All rights reserved.


# Imports

# General
import os
from pathlib import Path
from tqdm.auto import tqdm
from color import bcolors
from typing import List
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

from split_dataset import split_by_topic



# Analizador de argumentos
parser = argparse.ArgumentParser()

# Añadir un argumento para el archivo de configuración
parser.add_argument(
    "config_file", 
    type = str,
    help = "El formato del archivo debe ser 'config.json'"
)

try:
    # Obtención de los argumentos
    args = parser.parse_args()

    # Comprobaciones de los argumentos
    assert args.config_file.split('.')[-1] == "json"
except:
    # Visualización de las ayudas de los argumentos en caso de error en la comprobación de los mismos
    parser.print_help()

    # Finalización forzosa del programa
    sys.exit(0)

# Constantes globales
## Ruta base del programa python.
BASE_PATH = Path(__file__).resolve().parent

## Archivo de configuración
CONFIG_FILE = args.config_file

# Analizador de argumentos de la librería transformers
parser = HfArgumentParser(
    (
        ThemeDatasetArguments
    )
)

# Obtención de los argumentos de generación de datasets
generate_args, = parser.parse_json_file(json_file=str(BASE_PATH/CONFIG_FILE))

# Obtener el dispositivo usado por el proceso
device = "cuda" if torch.cuda.is_available() else "cpu"

# Fijar como desactivado el paralelismo al convertir las frases en tokens para evitar problemas
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Fijar semilla del generador de números aleatorios
set_seed(generate_args.seed)

print(bcolors.WARNING + "Cargando modelos..." + bcolors.RESET)


# ---------------------------------------
# ----- Traductor

# Obtener la key de autenticación de DeepL
auth_key = generate_args.auth_key_deepl

# Creación del traductor
translator = deepl.Translator(auth_key)


# ---------------------------------------
# ----- Modelo summarization


# Carga de la configuración del modelo para resumir
configSum = AutoConfig.from_pretrained(
    generate_args.model_summary_config
)

# Carga del tokenizer del modelo para resumir
tokenizerSum = AutoTokenizer.from_pretrained(
    generate_args.model_summary_tokenizer,
    config=generate_args.model_summary_tokenizer_config,
    use_fast=True
)

# Carga del modelo para resumir
modelSum = PegasusForConditionalGeneration.from_pretrained(
    generate_args.model_summary,
    from_tf=bool(".ckpt" in generate_args.model_summary),
    config=configSum
)


print(bcolors.OK + "Modelos cargados" + bcolors.RESET)



def unique(lista: List[str]):
    """! Eliminar frases repetidas.
    
    @param lista  Lista de frases

    @return Lista sin frases repetidas.
    """

    # Eliminación de las frases repetidas
    lista_set = set(lista)
    unique_list = list(lista_set)

    return unique_list


def save_dataset_EN(groups_datasets):
    """! Guardar lista de Dataframes con el contenido en inglés en un archivo único.
    
    @param groups_datasets  Lista de Dataframes con el contenido en inglés
    """

    # Unir los Dataframes
    total_dataset = pd.concat(groups_datasets)

    # Creación del nombre del archivo que contendrá el Dataframe
    filename = '.'.join(generate_args.initial_dataset_file.split('.')[:-1]) + "_EN.csv"
    
    # Guardado del Dataframe en el archivo
    save_csv(total_dataset, filename)


def calculateElements(groups_datasets: List[DataFrame], num_columns: int=1):
    """! Traducir al inglés una lista de Dataframes.
    
    @param groups_datasets  Lista de Dataframes
    @param num_columns      Número de columnas

    @return Número de elementos.
    """

    # Variable para el cálculo de elementos
    num = 0

    for i in groups_datasets:
        # Cálculo del número de filas
        num_rows = len(i.Topic.to_list())

        # Incrementar el número de elementos con el producto del número de columnas y el número de filas
        num += num_columns * num_rows
    
    return num


def traducirES_EN(groups_datasets):
    """! Traducir al inglés una lista de Dataframes.
    
    @param groups_datasets  Lista de Dataframes con el contenido en español

    @return Lista de Dataframes con el contenido en inglés.
    """

    print(bcolors.WARNING + "Realizando traducción a Inglés..." + bcolors.RESET)

    # Creación de la barra de progreso
    progress_bar = tqdm(range(calculateElements(groups_datasets, 2)))
    
    # Lista que contendrá los datasets después de la traducción
    new_groups_datasets = []

    # Traducción del contenido de cada conjunto de datos
    for dataset in groups_datasets:
        # Dataset que tendrá su contenido traducido
        new_dataset = {}

        # La primera columna se copia sin traducir
        new_dataset[dataset.columns.values[0]] = dataset[dataset.columns.values[0]].to_list()
        
        # Traducción y guardado del resto de columnas
        for column in dataset.columns.values[1:]:
            # Lista que contendrá el contenido de la columna
            lista = []
            for text in dataset[column].to_list():
                # Añadir traducción del texto a la lista
                lista.append(translator.translate_text(text, target_lang="EN-US").text)
                
                # Actualización de la barra de progreso
                progress_bar.update(1)

            # Añadir contenido a la columna del conjunto de datos
            new_dataset[column] = lista

        # Guardado del conjunto de datos traducido
        new_groups_datasets.append(DataFrame(new_dataset))

    print(bcolors.OK + "Terminada traducción" + bcolors.RESET)

    return new_groups_datasets


def generarResumenes(question, answer):
    """! Traducir al inglés una lista de Dataframes.
    
    @param question  Texto de la pregunta
    @param answer    Texto de la respuesta

    @return Texto resumido de la pregunta.
    @return Texto resumido de la respuesta.
    """

    # Obtener los tokens del texto de la pregunta
    # Se coloca el valor 500 como máxima longitud porque el modelo no generá frases de mayor longitud
    batch_question = tokenizerSum(question, max_length=500, truncation=True, return_tensors="pt")

    # Generación de la pregunta resumida
    translated_question = modelSum.generate(**batch_question, max_length=generate_args.max_length_summary, num_beams=generate_args.num_beams_summary, num_return_sequences=generate_args.num_beams_summary)
    
    # Decodificación de la pregunta
    tgt_text_question = tokenizerSum.batch_decode(translated_question, skip_special_tokens=True)

    # Eliminación de las frases repetidas en el caso de haber generado más de una pregunta
    resumenes_question = unique(tgt_text_question)


    # Obtener los tokens del texto de la respuesta
    batch_answer = tokenizerSum(answer, max_length=500, truncation=True, return_tensors="pt")

    # Generación de la respuesta resumida
    translated_answer = modelSum.generate(**batch_answer, max_length=generate_args.max_length_summary, num_beams=generate_args.num_beams_summary, num_return_sequences=generate_args.num_beams_summary)
    
    # Decodificación de la respuesta
    tgt_text_answer = tokenizerSum.batch_decode(translated_answer, skip_special_tokens=True)

    # Eliminación de las frases repetidas en el caso de haber generado más de una respuesta
    resumenes_answer = unique(tgt_text_answer)

    return resumenes_question, resumenes_answer


def summarization(groups_datasets):
    """! Realizar resumen del contenido de una lista de Dataframes.
    
    @param groups_datasets  Lista de Dataframes

    @return Lista de Dataframes con el contenido resumido.
    """

    print(bcolors.WARNING + "Realizando resumen del texto..." + bcolors.RESET)


    # Creación de la barra de progreso
    progress_bar = tqdm(range(calculateElements(groups_datasets)))

    # Lista que contendrá los datasets después de la obtención de los resúmenes
    new_groups_datasets = []

    # Generación de los resúmenes del contenido de cada conjunto de datos
    for dataset in groups_datasets:
        # Lista que contiene el contenido de la columna Subject
        subject = []
        # Lista que contiene el contenido de la columna Question
        question = []
        # Lista que contiene el contenido de la columna Answer
        answer = []

        for i, j, k in zip(dataset.Question.to_list(), dataset.Subject.to_list(), dataset.Answer.to_list()):  
            if str(type(i)) == "<class 'str'>" and str(type(k)) == "<class 'str'>":              
                with torch.no_grad():
                    # Generación del resumen de la pregunta y la respuesta
                    resumenes_question, resumenes_answer = generarResumenes(i, k)

                # Añadir pregunta a la lista
                question += resumenes_question

                # Añadir respuesta a la lista
                answer += resumenes_answer

                # Añadir elementos a la lista de la columna Subject en relación a la longitud de la lista de preguntas
                subject += [j] * len(resumenes_question)

            # Actualización de la barra de progreso
            progress_bar.update(1)
            
        # Creación de la lista que contiene el contenido de la columna Topic
        topic = [dataset.Topic.to_list()[0]] * len(question)

        # Si hay alguna pregunta en la lista
        if len(question) != 0:
            # Creación del conjunto de datos tras la generación de resúmenes
            new_groups_datasets.append(DataFrame({
                "Topic": topic,
                "Subject": subject,
                "Question": question,
                "Answer": answer
            }))

    print(bcolors.OK + "Terminado resumen" + bcolors.RESET)

    return new_groups_datasets


def simplify(groups_datasets):
    """! Simplificación del contenido de una lista de Dataframes.
    
    @param groups_datasets  Lista de Dataframes

    @return Lista de Dataframes con el contenido simplificado.
    """

    print(bcolors.WARNING + "Realizando simplificación..." + bcolors.RESET)

    # Creación de la barra de progreso
    progress_bar = tqdm(range(len(groups_datasets)))

    # Lista que contendrá los datasets después de la simplificación
    new_groups_datasets = []

    for dataset in groups_datasets:
        # Simplificación de las respuestas
        answer = simplify_sentences(dataset.Answer.to_list(), model_name=generate_args.model_simplify)
        
        # Simplificación de las preguntas
        question = simplify_sentences(dataset.Question.to_list(), model_name=generate_args.model_simplify)

        # Si hay alguna pregunta en la lista
        if len(question) != 0:
            # Creación del conjunto de datos tras la simplificación
            new_groups_datasets.append(DataFrame({
                "Topic": dataset.Topic.to_list(),
                "Subject": dataset.Subject.to_list(),
                "Question": question,
                "Answer": answer
            }))

        # Actualización de la barra de progreso
        progress_bar.update(1)

    print(bcolors.OK + "Terminada simplificación" + bcolors.RESET)

    return new_groups_datasets


def generate_theme_dataset(dataset):
    """! Generación de los datasets temáticos.
    
    @param dataset  Dataframe inicial

    @return Dataframe para adultos.
    @return Dataframe para niños.
    """

    # División del conjunto de datos en base al campo Topic
    groups_datasets = split_by_topic(dataset)

    # Si el contenido del conjunto de datos no estaba traducido
    if not generate_args.translated:
        # Traducción de los datasets al Inglés
        groups_datasets = traducirES_EN(groups_datasets)

        # Guardado del conjunto de datos traducido
        save_dataset_EN(groups_datasets)

    # Generación de las distintas respuestas mediante el resumen de los
    # textos de los distintos datasets

    # Generación de los datasets resumidos
    groups_datasets = summarization(groups_datasets)

    # Creación del conjunto de datos para adultos
    adult_dataset = pd.concat(groups_datasets)

    # Generación de los datasets simplificados
    groups_datasets = simplify(groups_datasets)

    # Creación del conjunto de datos para niños
    child_dataset = pd.concat(groups_datasets)

    return adult_dataset, child_dataset


def main():
    # Lectura del conjunto de datos inicial
    dataset = pd.read_csv(generate_args.initial_dataset_file)

    # Generación de los datasets temáticos
    adult_dataset, child_dataset = generate_theme_dataset(dataset)

    # Guardado del Dataframe para adultos
    save_csv(adult_dataset, os.path.join(generate_args.theme_result_dir, generate_args.adult_dataset_file))
    # Guardado del Dataframe para niños
    save_csv(child_dataset, os.path.join(generate_args.theme_result_dir, generate_args.child_dataset_file))


if __name__ == "__main__":
    main()
