#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import argparse



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        'dataset_file', 
        type=str, 
        help=''
    )
    args = parser.parse_args()


    # Lectura del dataset
    dataset = pd.read_csv(args.dataset_file)

    # Eliminación de una columna que se añade al guardar el archivo CSV
    dataset = dataset.drop(columns=["Unnamed: 0"])

    # Ordenar el dataset en base a la columna Topic
    dataset = dataset.sort_values('Topic')

    new_dataset = pd.DataFrame({})
    for column in dataset.columns.values:
        new_dataset[column] = dataset[column].to_list()

    # Guardar el dataset tras su ordenación
    new_dataset.to_csv(args.dataset_file)
