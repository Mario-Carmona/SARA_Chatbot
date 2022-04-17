#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import argparse
import pandas as pd
import sys
import os

from dataclass.split_dataset_arguments import SplitDatasetArguments
from transformers import HfArgumentParser

from pandas import DataFrame
from typing import List



def obtenerTrainDataset(groups_datasets: List[DataFrame], train_split: float, seed: int = 0):
    """
    Función para obtener el dataset de training a partir de un
    dataset general y el porcentaje de contenido que se debe extraer
    del dataset

    Args:
        groups_datasets (List[DataFrame])
        train_split (float)
        seed (int)
    
    Returns:
        DataFrame
    """

    # Lista que contendrá la partición de training de cada dataset
    lista = []

    # Obtención de las particiones de training de cada dataset, de esta 
    # forma el dataset para training mantendrá la proporción de datos de 
    # cada Topic que existe en el dataset original
    for dataset in groups_datasets:
        lista.append(dataset.sample(
            frac=train_split,
            random_state=seed,
            axis=0,
            ignore_index=True
        ))

    # Unión de las particiones de training
    train_dataset = pd.concat(lista)

    # Barajar el dataset de training
    train_dataset = train_dataset.sample(
        frac=1,
        random_state=seed,
        axis=0,
        ignore_index=True
    )

    return train_dataset


def obtenerValidationDataset(dataset: DataFrame, train_dataset: DataFrame, seed: int = 0):
    """
    Función para obtener el dataset de validation a partir de un
    dataset general y el dataset de training

    Args:
        dataset (DataFrame)
        train_dataset (DataFrame)
        seed (int)
    
    Returns:
        DataFrame
    """

    # Obtener la diferencia entre el dataset completo y el dataset de training
    validation_dataset = pd.merge(dataset, train_dataset, how='outer', indicator='Exist')
    validation_dataset = validation_dataset.loc[validation_dataset['Exist'] != 'both']
    validation_dataset = validation_dataset.drop(["Exist"], axis=1)
    
    # Barajar el dataset de validation
    validation_dataset = validation_dataset.sample(
        frac=1,
        random_state=seed,
        axis=0,
        ignore_index=True
    )

    return validation_dataset


def split_by_topic(dataset: DataFrame):
    groups = dataset.groupby(dataset.Topic)
    groups_values = list(set(dataset.Topic.to_list()))
    groups_datasets = [groups.get_group(value) for value in groups_values]

    return groups_datasets


def split_dataset(dataset, train_split, seed):
    if "Unnamed: 0" in dataset.columns.values:
        dataset = dataset.drop(columns=["Unnamed: 0"])
    
    train_dataset = obtenerTrainDataset(
        split_by_topic(dataset), 
        train_split,
        seed
    )

    valid_dataset = obtenerValidationDataset(
        dataset,
        train_dataset,
        seed
    )

    return train_dataset, valid_dataset



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
            SplitDatasetArguments
        )
    )

    split_args, = parser.parse_json_file(json_file=str(BASE_PATH/CONFIG_FILE))


    dataset = pd.read_csv(split_args.split_dataset_file)

    train_dataset, valid_dataset = split_dataset(
        dataset, 
        split_args.train_split, 
        split_args.seed
    )

    train_dataset.to_csv(os.path.join(split_args.split_result_dir, split_args.train_dataset_file))
    valid_dataset.to_csv(os.path.join(split_args.split_result_dir, split_args.valid_dataset_file))
