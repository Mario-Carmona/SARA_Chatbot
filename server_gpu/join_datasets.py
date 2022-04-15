#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import argparse
import os
import sys
import Path

from dataclass.join_datasets_arguments import JoinDatasetsArguments
from transformers import HfArgumentParser



def join_datasets(list_datasets, remove_source_files):
    result_dataset = pd.concat([pd.read_csv(dataset) for dataset in list_datasets])
    
    if remove_source_files:
        for dataset_path in list_datasets:
            os.remove(dataset_path)
    
    return result_dataset


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


    join_dataset = join_datasets(join_args.list_datasets, join_args.remove_source_files)

    join_dataset.to_csv(join_args.join_dataset_file)

