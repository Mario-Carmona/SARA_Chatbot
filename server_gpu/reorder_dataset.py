#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import argparse

from utils import save_csv



def ordenar_dataset(archivo):
    # Lectura del dataset
    dataset = pd.read_csv(archivo)

    # Ordenar el dataset en base a la columna Topic
    dataset = dataset.sort_values(dataset.columns.values[0])

    new_dataset = pd.DataFrame({})
    for column in dataset.columns.values:
        new_dataset[column] = dataset[column].to_list()

    return new_dataset


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        'dataset_file', 
        type=str, 
        help=''
    )
    args = parser.parse_args()

    order_dataset = ordenar_dataset(args.dataset_file)
    save_csv(order_dataset, args.dataset_file)
