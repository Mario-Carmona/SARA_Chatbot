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

    # Ordenar el dataset en base a la columna Topic
    dataset = dataset.sort_values('Topic')

    # Eliminación de una columna que se añade al guardar el archivo CSV
    dataset = dataset.drop(columns=["Unnamed: 0"])

    # Guardar el dataset tras su ordenación
    dataset.to_csv(args.dataset_file)
