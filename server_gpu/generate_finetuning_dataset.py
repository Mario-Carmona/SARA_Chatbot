#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import argparse
import pandas as pd
import sys
import os
from utils import save_csv

from extract_empathetic_dialogues import clean_trash_csv



def obtain_finetuning_dataset(train_dataset, valid_dataset):
    # Generación del dataset de training con el formato para el entrenamiento
    train_s_t = pd.DataFrame({
        "source": train_dataset.Question.to_list(),
        "target": train_dataset.Answer.to_list()
    })

    # Generación del dataset de validation con el formato para el entrenamiento
    validation_s_t = pd.DataFrame({
        "source": valid_dataset.Question.to_list(),
        "target": valid_dataset.Answer.to_list()
    })

    return train_s_t, validation_s_t


if __name__ == "__main__":

    BASE_PATH = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "train_dataset", 
        type = str,
        help = "El formato del archivo debe ser \'archivo.csv\'"
    )
    parser.add_argument(
        "valid_dataset", 
        type = str,
        help = "El formato del archivo debe ser \'archivo.csv\'"
    )

    try:
        args = parser.parse_args()
        assert args.input_dataset.split('.')[-1] == "csv"
        assert args.output_dataset.split('.')[-1] == "csv"
    except:
        parser.print_help()
        sys.exit(0)


    train_dataset = pd.read_csv(os.path.join(BASE_PATH,args.train_dataset))
    valid_dataset = pd.read_csv(os.path.join(BASE_PATH,args.valid_dataset))

    train_s_t, validation_s_t = obtain_finetuning_dataset(train_dataset, valid_dataset)

    dir = os.path.join(BASE_PATH,'/'.join(args.train_dataset.split('/')[:-1]))

    save_csv(train_s_t, os.path.join(dir,"train.csv"))
    clean_trash_csv(os.path.join(dir,"train.csv"), 3)

    save_csv(validation_s_t, os.path.join(dir,"validation.csv"))
    clean_trash_csv(os.path.join(dir,"validation.csv"), 3)
