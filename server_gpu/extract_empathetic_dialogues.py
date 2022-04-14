#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import argparse
import os
import pathlib
import json
from datasets import load_dataset



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        'config_file', 
        type=str, 
        help=''
    )
    args = parser.parse_args()

    with open(args.config_file) as file:
        config = json.load(file)


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

    num_elems_per_sentiment = int(config["num_samples"]/len(config["list_sentiment"]))

    lista_datasets_train = []
    lista_datasets_valid = []

    for sentiment in config["list_sentiment"]:
        print(sentiment)
        dataset_train_sentiment = dataset_train.loc[dataset_train['Sentiment'] == sentiment]
        dataset_valid_sentiment = dataset_valid.loc[dataset_valid['Sentiment'] == sentiment]
        print(dataset_train_sentiment)
        aux = input("-->")

        num_elems_train = min(num_elems_per_sentiment*config["train_split"], len(dataset_train_sentiment.Sentiment.to_list()))

        dataset_train_sentiment = dataset_train_sentiment.sample(
            int(num_elems_train), 
            random_state=seed
        )

        dataset_train_sentiment = dataset_train_sentiment.rename(columns={"Sentiment":"Subject"})
        dataset_train_sentiment["Topic"] = ["Actitud"] * len(dataset_train_sentiment.Question.to_list())

        num_elems_valid = (num_elems_train/config["train_split"])*(1-config["train_split"])

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

    
    dir = '/'.join(config["result_file"].split('/')[:-1])

    if not os.path.exists(dir):
        pathlib.Path(dir).mkdir(parents=True, exist_ok=True)

    total_dataset_train.to_csv(config["result_file"] + "_train.csv")
    total_dataset_valid.to_csv(config["result_file"] + "_validation.csv")
