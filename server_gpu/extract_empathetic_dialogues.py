#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import argparse
import os
import pathlib
import sys
import Path
import json
from datasets import load_dataset

from dataclass.extract_sentiments_arguments import ExtractSentimentsArguments
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
            ExtractSentimentsArguments
        )
    )

    extract_args, = parser.parse_json_file(json_file=str(BASE_PATH/CONFIG_FILE))


    seed = 0

    dataset = load_dataset("empathetic_dialogues")

    dataset_train = pd.DataFrame({
        "Sentiment": dataset["train"]["context"],
        "Question": dataset["train"]["prompt"],
        "Answer": dataset["train"]["utterance"]
    })

    dataset_valid = pd.DataFrame({
        "Sentiment": dataset["validation"]["context"],
        "Question": dataset["validation"]["prompt"],
        "Answer": dataset["validation"]["utterance"]
    })

    num_elems_per_sentiment = int(extract_args.num_samples/len(extract_args.list_sentiment))

    lista_datasets_train = []
    lista_datasets_valid = []

    for sentiment in extract_args.list_sentiment:
        print(sentiment)
        dataset_train_sentiment = dataset_train.loc[dataset_train['Sentiment'] == sentiment]
        dataset_valid_sentiment = dataset_valid.loc[dataset_valid['Sentiment'] == sentiment]
        print(dataset_train_sentiment)
        aux = input("-->")

        num_elems_train = min(num_elems_per_sentiment*extract_args.train_split, len(dataset_train_sentiment.Sentiment.to_list()))

        dataset_train_sentiment = dataset_train_sentiment.sample(
            int(num_elems_train), 
            random_state=seed
        )

        dataset_train_sentiment = dataset_train_sentiment.rename(columns={"Sentiment":"Subject"})
        dataset_train_sentiment["Topic"] = ["Actitud"] * len(dataset_train_sentiment.Question.to_list())

        num_elems_valid = (num_elems_train/extract_args.train_split)*(1-extract_args.train_split)

        dataset_valid_sentiment = dataset_valid_sentiment.sample(
            int(num_elems_valid), 
            random_state=seed
        )

        dataset_valid_sentiment = dataset_valid_sentiment.rename(columns={"Sentiment":"Subject"})
        dataset_valid_sentiment["Topic"] = ["Actitud"] * len(dataset_valid_sentiment.Question.to_list())

        lista_datasets_train.append(dataset_train_sentiment)
        lista_datasets_valid.append(dataset_valid_sentiment)

    total_dataset_train = pd.concat(lista_datasets_train)
    total_dataset_valid = pd.concat(lista_datasets_valid)


    total_dataset_train.to_csv(os.path.join(extract_args.result_dir, "sentiments_train.csv"))
    total_dataset_valid.to_csv(os.path.join(extract_args.result_dir, "sentiments_validation.csv"))
