#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import argparse
import sys
from pathlib import Path
from datasets import load_dataset

from dataclass.attitude_dataset_arguments import AttitudeDatasetArguments
from transformers import HfArgumentParser



def modify_dataset(dataset):
    dataset = dataset.rename(columns={"Sentiment":"Subject"})

    return dataset


def extract_dataset_sentiment(list_sentiment, num_samples, seed):
    dataset = load_dataset("empathetic_dialogues")

    dataset_train = pd.DataFrame({
        "Topic": ["Actitud"] * len(dataset["train"]["context"]),
        "Sentiment": dataset["train"]["context"],
        "Question": dataset["train"]["prompt"],
        "Answer": dataset["train"]["utterance"]
    })

    dataset_valid = pd.DataFrame({
        "Topic": ["Actitud"] * len(dataset["validation"]["context"]),
        "Sentiment": dataset["validation"]["context"],
        "Question": dataset["validation"]["prompt"],
        "Answer": dataset["validation"]["utterance"]
    })

    dataset = pd.concat([dataset_train, dataset_valid])

    num_elems_per_sentiment = int(num_samples/len(list_sentiment))

    lista_datasets = []

    for sentiment in list_sentiment:
        dataset_sentiment = dataset.loc[dataset['Sentiment'] == sentiment]

        dataset_sentiment = dataset_sentiment.sample(
            num_elems_per_sentiment,
            random_state=seed
        )

        dataset_sentiment = modify_dataset(dataset_sentiment)

        lista_datasets.append(dataset_sentiment)

    total_dataset = pd.concat(lista_datasets)
    
    for i in range(len(total_dataset)):
        print(len(total_dataset.iloc[i,:]))
    

    return total_dataset


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
            AttitudeDatasetArguments
        )
    )

    extract_args, = parser.parse_json_file(json_file=str(BASE_PATH/CONFIG_FILE))


    dataset = extract_dataset_sentiment(extract_args.list_sentiment, extract_args.num_samples, extract_args.seed)

    dataset.to_csv(extract_args.attitude_dataset_file)
