#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import argparse



def ordenar_dataset(archivo):
    # Lectura del dataset
    dataset = pd.read_csv(archivo)

    # Eliminación de una columna que se añade al guardar el archivo CSV
    dataset = dataset.drop(columns=["Unnamed: 0"])

    # Ordenar el dataset en base a la columna Topic
    dataset = dataset.sort_values(dataset.columns.values[0])

    new_dataset = pd.DataFrame({})
    for column in dataset.columns.values:
        new_dataset[column] = dataset[column].to_list()

    # Guardar el dataset tras su ordenación
    new_dataset.to_csv(archivo)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        'dataset_file', 
        type=str, 
        help=''
    )
    args = parser.parse_args()

    ordenar_dataset(args.dataset_file)
