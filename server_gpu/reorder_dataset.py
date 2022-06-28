#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""! @brief Script para la ordenación de un conjunto de datos."""


##
# @file reorder_dataset.py
#
# @brief Programa para la ordenación de un conjunto de datos.
#
# @section description_main Descripción
# Programa para la ordenación de un conjunto de datos.
#
# @section libraries_main Librerías/Módulos
# - Librería pandas (https://pandas.pydata.org/docs/)
#   - Acceso a la función DataFrame
#   - Acceso a la función read_csv
# - Librería estándar argparse (https://docs.python.org/3/library/argparse.html)
#   - Acceso a la función ArgumentParser
# - Librería utils
#   - Acceso a la función save_csv
#
# @section author_doxygen_example Autor
# - Created by Mario Carmona Segovia.
#
# Copyright (c) 2022.  All rights reserved.


# Imports

import pandas as pd
import argparse

from utils import save_csv



def ordenar_dataset(archivo):
    """! Ordenar un conjunto de datos.
    
    @param archivo  Nombre del archivo que contiene el conjunto de datos a ordenar
    
    @return Dataframe ordenado.
    """

    # Lectura del conjunto de datos
    dataset = pd.read_csv(archivo)

    # Ordenar el conjunto de datos en base a la columna Topic
    dataset = dataset.sort_values(dataset.columns.values[0])

    # Creación del nuevo conjunto de datos ordenado
    new_dataset = pd.DataFrame({})
    for column in dataset.columns.values:
        new_dataset[column] = dataset[column].to_list()

    return new_dataset


def main():
    """! Entrada al programa."""

    # Analizador de argumentos
    parser = argparse.ArgumentParser()

    # Añadir un argumento para el nombre del archivo que contiene el conjunto de datos
    parser.add_argument(
        'dataset_file', 
        type=str, 
        help='El formato del archivo debe ser \'dataset.csv\''
    )

    # Obtención de los argumentos
    args = parser.parse_args()

    # Ordenación del conjunto de datos
    order_dataset = ordenar_dataset(args.dataset_file)

    # Guardado del conjunto de datos ordenado en el mismo archivo que contenía el conjunto de datos original
    save_csv(order_dataset, args.dataset_file)


if __name__ == "__main__":
    main()
