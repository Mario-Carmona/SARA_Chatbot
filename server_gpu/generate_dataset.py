#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import Path
import sys
import pandas as pd
import os

from generate_theme_dataset import generate_theme_dataset
from extract_empathetic_dialogues import extract_dataset_sentiment
from join_datasets import join_datasets
from split_dataset import split_dataset
from generate_finetuning_dataset import obtain_finetuning_dataset

from dataclass.generate_dataset_arguments import GenerateDatasetArguments
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
            GenerateDatasetArguments
        )
    )

    generate_args, = parser.parse_json_file(json_file=str(BASE_PATH/CONFIG_FILE))


    # Lectura del dataset
    dataset_initial = pd.read_csv(generate_args.initial_dataset_file)

    adult_dataset, child_dataset = generate_theme_dataset(dataset_initial)

    adult_dataset_path = os.path.join(generate_args.theme_result_dir, generate_args.adult_dataset_file)
    adult_dataset.to_csv(adult_dataset_path)

    child_dataset_path = os.path.join(generate_args.theme_result_dir, generate_args.child_dataset_file)
    child_dataset.to_csv(child_dataset_path)
    

    sentiment_dataset = extract_dataset_sentiment(generate_args.list_sentiment, generate_args.num_samples, generate_args.seed)

    sentiment_dataset.to_csv(generate_args.attitude_dataset_file)

    generate_args.list_datasets.append(generate_args.attitude_dataset_file)


    for dataset_path, cadena in zip([adult_dataset_path, child_dataset_path],["_adult", "_child"]):
        new_list_datasets = generate_args.list_datasets
        new_list_datasets.append(dataset_path)

        join_dataset = join_datasets(new_list_datasets, generate_args.remove_source_files)

        new_filename = generate_args.join_dataset_file.split('.')[0] + cadena + ".csv"
        join_dataset.to_csv(new_filename)


        train_dataset, valid_dataset = split_dataset(join_dataset, generate_args.train_split, generate_args.seed)

        dir_path = os.path.join(generate_args.split_result_dir, f"split_{generate_args.train_split}" + cadena)
        os.mkdir(dir_path)

        train_dataset.to_csv(os.path.join(dir_path, generate_args.train_dataset_file))
        valid_dataset.to_csv(os.path.join(dir_path, generate_args.valid_dataset_file))


        train_s_t, validation_s_t = obtain_finetuning_dataset(train_dataset, valid_dataset)

        train_s_t.to_csv(os.path.join(dir_path, "train.csv"))
        validation_s_t.to_csv(os.path.join(dir_path, "validation.csv"))
