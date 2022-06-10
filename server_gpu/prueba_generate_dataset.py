#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import sys
import pandas as pd
import os
from utils import save_csv

from prueba_generate_theme_dataset import generate_theme_dataset
from extract_empathetic_dialogues import extract_dataset_sentiment, clean_trash_csv
from join_datasets import join_datasets
from split_dataset import split_dataset
from generate_finetuning_dataset import obtain_finetuning_dataset
from reorder_dataset import ordenar_dataset

from dataclass.generate_dataset_arguments import GenerateDatasetArguments
from transformers import HfArgumentParser



def save_dataset(dataset, archivo):
    save_csv(dataset, archivo)
    order_dataset = ordenar_dataset(archivo)
    save_csv(order_dataset, archivo)


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
            GenerateDatasetArguments
        )
    )

    generate_args, = parser.parse_json_file(json_file=str(BASE_PATH/CONFIG_FILE))


    # Lectura del dataset
    dataset_initial = pd.read_csv(generate_args.initial_dataset_file)

    adult_dataset, child_dataset = generate_theme_dataset(dataset_initial)

    adult_dataset_path = os.path.join(generate_args.theme_result_dir, generate_args.adult_dataset_file)
    save_dataset(adult_dataset, adult_dataset_path)

    child_dataset_path = os.path.join(generate_args.theme_result_dir, generate_args.child_dataset_file)
    save_dataset(child_dataset, child_dataset_path)


    for dataset_path, cadena in zip([adult_dataset_path, child_dataset_path],["_adult", "_child"]):
        new_list_datasets = generate_args.list_datasets
        new_list_datasets.append(dataset_path)

        join_dataset = join_datasets(new_list_datasets, generate_args.remove_source_files)

        new_filename = generate_args.join_dataset_file.split('.')[0] + cadena + ".csv"
        save_dataset(join_dataset, new_filename)


        train_dataset, valid_dataset = split_dataset(join_dataset, generate_args.train_split, generate_args.seed)

        dir_path = os.path.join(generate_args.split_result_dir, f"split_{generate_args.train_split}" + cadena)
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)

        save_dataset(train_dataset, os.path.join(dir_path, generate_args.train_dataset_file))
        save_dataset(valid_dataset, os.path.join(dir_path, generate_args.valid_dataset_file))


        train_s_t, validation_s_t = obtain_finetuning_dataset(train_dataset, valid_dataset)

        save_csv(train_s_t, os.path.join(dir_path, "train.csv"))
        clean_trash_csv(os.path.join(dir_path, "train.csv"), 2)

        save_csv(validation_s_t, os.path.join(dir_path, "validation.csv"))
        clean_trash_csv(os.path.join(dir_path, "validation.csv"), 2)
