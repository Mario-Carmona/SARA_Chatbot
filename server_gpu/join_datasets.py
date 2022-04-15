#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import argparse
import os
import pathlib
import sys
import Path

from dataclass.join_datasets_arguments import JoinDatasetsArguments
from transformers import HfArgumentParser



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        'config_file', 
        type=str, 
        help="El formato del archivo debe ser \'config.json\'"
    )
    
    try:
        args = parser.parse_args()
        assert args.config_file.split('.')[-1] == "json"
    except:
        parser.print_help()
        sys.exit(0)

    BASE_PATH = Path(__file__).resolve().parent
    CONFIG_FILE = args.config_file


    parser = HfArgumentParser(
        (
            JoinDatasetsArguments
        )
    )

    join_args, = parser.parse_json_file(json_file=str(BASE_PATH/CONFIG_FILE))


    lista = []

    for dataset_path in join_args.list_datasets:
        dataset = pd.read_csv(dataset_path)
        lista.append(dataset)

    result_dataset = pd.concat(lista)

    result_dataset.to_csv(join_args.result_file)

    if join_args.remove_source_files:
        for dataset_path in join_args.list_datasets:
            os.remove(dataset_path)
