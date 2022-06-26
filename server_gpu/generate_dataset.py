#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""! @brief Script para la obtención de un dataset para el entrenamiento de un modelo."""


##
# @file generate_dataset.py
#
# @brief Programa para la obtención de un dataset para el entrenamiento de un modelo.
#
# @section description_main Descripción
# Programa para la obtención de un dataset para el entrenamiento de un modelo.
#
# @section libraries_main Librerías/Módulos
# - Librería pandas (https://pandas.pydata.org/docs/)
#   - Acceso a la función read_csv
# - Librería estándar argparse (https://docs.python.org/3/library/argparse.html)
#   - Acceso a la función ArgumentParser
# - Librería estándar pathlib (https://docs.python.org/3/library/pathlib.html)
#   - Acceso a la función Path
# - Librería estándar sys (https://docs.python.org/3/library/sys.html)
#   - Acceso a la función exit
# - Librería estándar os (https://docs.python.org/3/library/os.html)
#   - Acceso a la función path.join
#   - Acceso a la función mkdir
#   - Acceso a la función path.exists
# - Librería utils
#   - Acceso a la función save_csv
# - Librería generate_theme_dataset
#   - Acceso a la función generate_theme_dataset
# - Librería join_datasets
#   - Acceso a la función join_datasets
# - Librería split_dataset
#   - Acceso a la función split_dataset
# - Librería generate_finetuning_dataset
#   - Acceso a la función obtain_finetuning_dataset
# - Librería reorder_dataset
#   - Acceso a la función ordenar_dataset
# - Librería dataclass.generate_dataset_arguments
#   - Acceso a la clase GenerateDatasetArguments
# - Librería transformers (https://pypi.org/project/transformers)
#   - Acceso a la función HfArgumentParser
#
# @section author_doxygen_example Autor
# - Created by Mario Carmona Segovia.
#
# Copyright (c) 2022.  All rights reserved.


# Imports

import argparse
from pathlib import Path
import sys
import pandas as pd
import os
from utils import save_csv

from generate_theme_dataset import generate_theme_dataset
from join_datasets import join_datasets
from split_dataset import split_dataset
from generate_finetuning_dataset import obtain_finetuning_dataset
from reorder_dataset import ordenar_dataset

from dataclass.generate_dataset_arguments import GenerateDatasetArguments
from transformers import HfArgumentParser



def save_dataset(dataset, archivo):
    """! Guardar el contenido de un Dataframe de forma ordenada en un archivo.
    
    @param dataset  Dataframe que contiene los datos
    @param archivo  Ruta al archivo donde guardar el Dataframe
    """

    # Guardado del Dataframe
    save_csv(dataset, archivo)

    # Obtener el Dataframe de forma ordenada
    order_dataset = ordenar_dataset(archivo)

    # Guardado del Dataframe ordenado
    save_csv(order_dataset, archivo)


def main():
    """! Entrada al programa."""

    # Analizador de argumentos
    parser = argparse.ArgumentParser()

    # Añadir un argumento para el archivo de configuración
    parser.add_argument(
        'config_file', 
        type=str, 
        help="El formato del archivo debe ser \'config.json\'"
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
            GenerateDatasetArguments
        )
    )

    # Obtención de los argumentos de generación de datasets
    generate_args, = parser.parse_json_file(json_file=str(BASE_PATH/CONFIG_FILE))


    # Obtención del Dataframe inicial
    dataset_initial = pd.read_csv(generate_args.initial_dataset_file)

    # Obtención del Dataframe para adultos y para niños
    adult_dataset, child_dataset = generate_theme_dataset(dataset_initial)

    # Generación de la ruta del archivo para el Dataframe para adultos
    adult_dataset_path = os.path.join(generate_args.theme_result_dir, generate_args.adult_dataset_file)
    # Guardado ordenado del Dataframe
    save_dataset(adult_dataset, adult_dataset_path)

    # Generación de la ruta del archivo para el Dataframe para niños
    child_dataset_path = os.path.join(generate_args.theme_result_dir, generate_args.child_dataset_file)
    # Guardado ordenado del Dataframe
    save_dataset(child_dataset, child_dataset_path)

    for dataset_path, cadena in zip([adult_dataset_path, child_dataset_path],["_adult", "_child"]):
        # Lista de datasets a unir
        new_list_datasets = generate_args.list_datasets

        # Añadir el dataset temático a la lista
        new_list_datasets.append(dataset_path)

        # Relalizar la unión de los datasets
        join_dataset = join_datasets(new_list_datasets, generate_args.remove_source_files)

        # Generación de la ruta del archivo para el Dataframe resultado de la unión 
        new_filename = generate_args.join_dataset_file.split('.')[0] + cadena + ".csv"
        # Guardado del Dataframe resultado de la unión
        save_dataset(join_dataset, new_filename)

        # Generación del Dataframe de entrenamiento y de validación resultado de la división del Dataframe resultado de la unión
        train_dataset, valid_dataset = split_dataset(join_dataset, generate_args.train_split, generate_args.seed)

        # Generación de la ruta de la carpeta que contendrá los datasets para el entrenamiento
        dir_path = os.path.join(generate_args.split_result_dir, f"split_{generate_args.train_split}{cadena}")
        
        # Si la carpeta no existe
        if not os.path.exists(dir_path):
            # Se crea la carpeta
            os.mkdir(dir_path)

        # Guardado del Dataframe de entrenamiento
        save_dataset(train_dataset, os.path.join(dir_path, generate_args.train_dataset_file))
        # Guardado del Dataframe de validación
        save_dataset(valid_dataset, os.path.join(dir_path, generate_args.valid_dataset_file))

        # Generación de los Dataframes con el formato de entrenamiento
        train_s_t, validation_s_t = obtain_finetuning_dataset(train_dataset, valid_dataset)

        # Guardado del Dataframe de entrenamiento con el formato de entrenamiento
        save_csv(train_s_t, os.path.join(dir_path, "train.csv"))
        # Guardado del Dataframe de validación con el formato de entrenamiento
        save_csv(validation_s_t, os.path.join(dir_path, "validation.csv"))


if __name__ == "__main__":
    main()
