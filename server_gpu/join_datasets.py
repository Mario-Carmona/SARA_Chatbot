#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""! @brief Script para la unión de datasets."""


##
# @file join_datasets.py
#
# @brief Programa para la unión de datasets.
#
# @section description_main Descripción
# Programa para la unión de datasets.
#
# @section libraries_main Librerías/Módulos
# - Librería pandas (https://pandas.pydata.org/docs/)
#   - Acceso a la función read_csv
#   - Acceso a la función concat
# - Librería estándar argparse (https://docs.python.org/3/library/argparse.html)
#   - Acceso a la función ArgumentParser
# - Librería utils
#   - Acceso a la función save_csv
# - Librería estándar os (https://docs.python.org/3/library/os.html)
#   - Acceso a la función remove
# - Librería estándar sys (https://docs.python.org/3/library/sys.html)
#   - Acceso a la función exit
# - Librería estándar pathlib (https://docs.python.org/3/library/pathlib.html)
#   - Acceso a la función Path
# - Librería dataclass.join_datasets_arguments
#   - Acceso a la clase JoinDatasetsArguments
# - Librería transformers (https://pypi.org/project/transformers)
#   - Acceso a la función HfArgumentParser
#
# @section author_doxygen_example Autor
# - Created by Mario Carmona Segovia.
#
# Copyright (c) 2022.  All rights reserved.


# Imports

import pandas as pd
import argparse
import os
import sys
from pathlib import Path
from utils import save_csv

from dataclass.join_datasets_arguments import JoinDatasetsArguments
from transformers import HfArgumentParser



def join_datasets(list_datasets, remove_source_files):
    """! Unir datasets.
    
    @param list_datasets  Lista de datasets
    @param remove_source_files  Indicación de eliminar o no los archivos originales tras la unión
    
    @return Dataframe tras la unión.
    """

    # Lista que contendrá los Dataframes de cada uno de los datasets
    lista = []

    # Obtener los Dataframes de los distintos datasets
    for i in list_datasets:
        dataset = pd.read_csv(i)
        lista.append(dataset)

    # Unión de los Dataframes
    result_dataset = pd.concat(lista)
    
    # Si se indica la eliminación de los archivos originales
    if remove_source_files:
        for dataset_path in list_datasets:
            # Eliminación del conjunto de datos
            os.remove(dataset_path)
    
    return result_dataset


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
            JoinDatasetsArguments
        )
    )

    # Obtención de los argumentos de unión de datasets
    join_args, = parser.parse_json_file(json_file=str(BASE_PATH/CONFIG_FILE))

    # Unión de los datasets
    join_dataset = join_datasets(join_args.list_datasets, join_args.remove_source_files)

    # Guardado del conjunto de datos producto de la unión
    save_csv(join_dataset, join_args.join_dataset_file)


if __name__ == "__main__":
    main()
