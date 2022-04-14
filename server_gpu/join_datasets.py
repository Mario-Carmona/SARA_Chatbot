#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import argparse
import os
import pathlib



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        'list_datasets', 
        type=list, 
        help=''
    )
    parser.add_argument(
        'result_file', 
        type=str, 
        help=''
    )
    args = parser.parse_args()


    lista = []

    for dataset_path in args.list_datasets:
        dataset = pd.read_csv(dataset_path)
        lista.append(dataset)

    result_dataset = pd.concat(lista)


    dir = '/'.join(args.result_file.split('/')[:-1])

    if not os.path.exists(dir):
        pathlib.Path(dir).mkdir(parents=True, exist_ok=True)

    result_dataset.to_csv(f"{args.result_file}.csv")
