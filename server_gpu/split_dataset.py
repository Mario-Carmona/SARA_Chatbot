#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""! @brief Script para la división de un conjunto de datos."""


##
# @file split_dataset.py
#
# @brief Programa para la división de un conjunto de datos.
#
# @section description_main Descripción
# Programa para la división de un conjunto de datos.
#
# @section libraries_main Librerías/Módulos
# - Librería pandas (https://pandas.pydata.org/docs/)
#   - Acceso a la función DataFrame
#   - Acceso a la función read_csv
#   - Acceso a la función merge
#   - Acceso a la función concat
# - Librería estándar argparse (https://docs.python.org/3/library/argparse.html)
#   - Acceso a la función ArgumentParser
# - Librería utils
#   - Acceso a la función save_csv
# - Librería estándar sys (https://docs.python.org/3/library/sys.html)
#   - Acceso a la función exit
# - Librería estándar os (https://docs.python.org/3/library/os.html)
#   - Acceso a la función path.join
# - Librería estándar pathlib (https://docs.python.org/3/library/pathlib.html)
#   - Acceso a la función Path
# - Librería estándar typing (https://docs.python.org/3/library/typing.html)
#   - Acceso a la clase List
# - Librería dataclass.split_dataset_arguments
#   - Acceso a la clase SplitDatasetArguments
# - Librería transformers (https://pypi.org/project/transformers)
#   - Acceso a la función HfArgumentParser
#
# @section author_doxygen_example Autor
# - Created by Mario Carmona Segovia.
#
# Copyright (c) 2022.  All rights reserved.


# Imports

from pathlib import Path
import argparse
import pandas as pd
import sys
import os
from utils import save_csv

from dataclass.split_dataset_arguments import SplitDatasetArguments
from transformers import HfArgumentParser

from pandas import DataFrame
from typing import List



def obtenerTrainDataset(groups_datasets: List[DataFrame], train_split: float, seed: int = 0):
    """! Obtener el Dataframe de entrenamiento de una lista de Dataframes.
    
    @param groups_datasets  Lista de Dataframes
    @param train_split      Porcentaje de ejemplos que forman parte del conjunto de entrenamiento
    @param seed             Semilla del generador de números aleatorios

    @return Dataframe de entrenamiento.
    """

    # Lista que contendrá la partición de entrenamiento de cada Dataframe
    lista = []

    # Obtención de las particiones de entrenamiento de cada Dataframe, de esta 
    # forma el Dataframe de entrenamiento mantendrá la proporción de datos de 
    # cada Topic que existe en el conjunto de datos original
    for dataset in groups_datasets:
        lista.append(dataset.sample(
            frac=train_split,
            random_state=seed,
            axis=0,
            ignore_index=True
        ))

    # Unión de las particiones de entrenamiento
    train_dataset = pd.concat(lista)

    # Barajar el Dataframe de entrenamiento
    train_dataset = train_dataset.sample(
        frac=1,
        random_state=seed,
        axis=0,
        ignore_index=True
    )

    return train_dataset


def obtenerValidationDataset(dataset: DataFrame, train_dataset: DataFrame, seed: int = 0):
    """! Obtener el Dataframe de validación de una lista de Dataframes.
    
    @param dataset        Dataframe completo
    @param train_dataset  Dataframe de entrenamiento
    @param seed           Semilla del generador de números aleatorios

    @return Dataframe de validación.
    """

    # Obtener la diferencia entre el Dataframe completo y el Dataframe de entrenamiento
    validation_dataset = pd.merge(dataset, train_dataset, how='outer', indicator='Exist')
    validation_dataset = validation_dataset.loc[validation_dataset['Exist'] != 'both']
    validation_dataset = validation_dataset.drop(["Exist"], axis=1)
    
    # Barajar el Dataframe de validación
    validation_dataset = validation_dataset.sample(
        frac=1,
        random_state=seed,
        axis=0,
        ignore_index=True
    )

    return validation_dataset


def split_by_topic(dataset: DataFrame):
    """! Dividir Dataframe en base a su columna Topic.
    
    @param dataset  Dataframe a dividir

    @return Lista de Dataframes resultante de la división.
    """

    # Obtener todos los valores de la columna Topic
    groups = dataset.groupby(dataset.Topic)

    # Nos quedamos con los valores únicos de todos los obtenidos
    groups_values = list(set(dataset.Topic.to_list()))

    # Creación de la lista en base a los valores obtenidos en el paso anterior
    groups_datasets = [groups.get_group(value) for value in groups_values]

    return groups_datasets


def split_dataset(dataset, train_split, seed):
    """! Dividir Dataframe en su conjunto de entrenamiento y de validación.
    
    @param dataset      Dataframe a dividir
    @param train_split  Porcentaje de ejemplos que forman parte del conjunto de entrenamiento
    @param seed         Semilla del generador de números aleatorios

    @return Dataframe de entrenamiento.
    @return Dataframe de validación.
    """

    # Obtener Dataframe de entrenamiento
    train_dataset = obtenerTrainDataset(
        split_by_topic(dataset), 
        train_split,
        seed
    )

    # Obtener Dataframe de validación
    valid_dataset = obtenerValidationDataset(
        dataset,
        train_dataset,
        seed
    )

    return train_dataset, valid_dataset


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
            SplitDatasetArguments
        )
    )

    # Obtención de los argumentos de división de datasets
    split_args, = parser.parse_json_file(json_file=str(BASE_PATH/CONFIG_FILE))

    # Obtención del Dataframe
    dataset = pd.read_csv(split_args.split_dataset_file)

    # División del Dataframe en su conjunto de entrenamiento y de validación
    train_dataset, valid_dataset = split_dataset(
        dataset, 
        split_args.train_split, 
        split_args.seed
    )

    # Guardado del Dataframe de entrenamiento
    save_csv(train_dataset, os.path.join(split_args.split_result_dir, split_args.train_dataset_file))
    
    # Guardado del Dataframe de validación
    save_csv(valid_dataset, os.path.join(split_args.split_result_dir, split_args.valid_dataset_file))


if __name__ == "__main__":
    main()
    
