#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import pandas as pd
import sys
import os


def obtenerTrainDataset(dataset, train_split):
    train_dataset = dataset.sample(
        frac=train_split,
        random_state=0,
        axis=0,
        ignore_index=True
    )

    return train_dataset
    
def obtenerValidationDataset(dataset, train_dataset):
    validation_dataset = pd.merge(dataset, train_dataset, how='outer', indicator='Exist')
    validation_dataset = validation_dataset.loc[validation_dataset['Exist'] != 'both']
    validation_dataset = validation_dataset.drop(["Exist"], axis=1)
    validation_dataset = validation_dataset.sample(
        frac=1,
        random_state=0,
        axis=0,
        ignore_index=True
    )

    return validation_dataset

def obtenerSourceTarget(dataset):
    dataset["Source"] = dataset["Nombre"]
    dataset["Source"] += ". "
    for i, nom in enumerate(dataset["Nombre_Alternativo"]):
        if str(nom) != "nan":
            dataset.at[i, "Source"] += nom + ". "
    dataset["Source"] += dataset["Definición"] + ". "
    dataset["Source"] += dataset["Pregunta"] + "."

    dataset["Target"] = dataset["Respuesta"]

    return dataset["Source"].tolist(), dataset["Target"].tolist()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "dataset_file", 
        type = str,
        help = "El formato del archivo debe ser \'archivo.csv\'"
    )

    parser.add_argument(
        "result_dir", 
        type = str,
        help = "Debe ser un directorio existe"
    )

    parser.add_argument(
        "train_split", 
        type = float,
        help = "De ser un flotante mayor que 0 y menor que 1'"
    )

    try:
        args = parser.parse_args()
        assert args.dataset_file.split('.')[-1] == "csv"
        assert os.path.exists(args.result_dir)
        assert args.train_split > 0.0 and args.train_split < 1.0
    except:
        parser.print_help()
        sys.exit(0)


    dataset = pd.read_csv(args.dataset_file)

    train_dataset = obtenerTrainDataset(dataset, args.train_split)

    validation_dataset = obtenerValidationDataset(dataset, train_dataset)

    train_source, train_target = obtenerSourceTarget(train_dataset)

    train_aux = pd.DataFrame({
        "source": train_source,
        "target": train_target
    })

    val_source, val_target = obtenerSourceTarget(validation_dataset)

    val_aux = pd.DataFrame({
        "source": val_source,
        "target": val_target
    })

    dir = os.path.join(args.result_dir, f"split_{args.train_split}")
    if not os.path.exists(dir):
        os.mkdir(dir)
    
    train_aux.to_csv(f"{dir}/train.csv")
    val_aux.to_csv(f"{dir}/val.csv")
