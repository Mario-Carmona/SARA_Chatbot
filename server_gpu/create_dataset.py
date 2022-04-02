#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from cmath import nan
import pandas as pd
import sys


def generarContexto(dataset):
    context = []
    nombre = dataset["Nombre"].tolist()
    nombre_alter = dataset["Nombre_Alternativo"].tolist()
    definicion = dataset["Definición"].tolist()

    for i in range(len(nombre)):
        aux = nombre[i]
        if str(nombre_alter[i]) != "nan":
            aux += ". " + nombre_alter[i]
        aux += ". " + definicion[i]

        context.append(aux)

    dataset = dataset.drop(
        ["Nombre", "Nombre_Alternativo", "Definición"], 
        axis=1
    )
    dataset["Contexto"] = context

    return dataset

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


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "datasetfile", 
        type = str,
        help = "El formato del archivo debe ser \'archivo.csv\'"
    )

    parser.add_argument(
        "trainfile", 
        type = str,
        help = "El formato del archivo debe ser \'archivo.csv\'"
    )
    parser.add_argument(
        "validationfile", 
        type = str,
        help = "El formato del archivo debe ser \'archivo.csv\'"
    )

    parser.add_argument(
        "train_split", 
        type = float,
        help = "De ser un flotante mayor que 0 y menor que 1'"
    )

    try:
        args = parser.parse_args()
        assert args.datasetfile.split('.')[-1] == "csv"
        assert args.trainfile.split('.')[-1] == "csv"
        assert args.validationfile.split('.')[-1] == "csv"
        assert args.train_split > 0.0 and args.train_split < 1.0
    except:
        parser.print_help()
        sys.exit(0)


    dataset = pd.read_csv(args.datasetfile)

    train_dataset = obtenerTrainDataset(dataset, args.train_split)

    validation_dataset = obtenerValidationDataset(dataset, train_dataset)

    train_dataset = generarContexto(train_dataset)

    validation_dataset = generarContexto(validation_dataset)

    nombre_train_file = args.trainfile.split('.')[0] + "_" + str(args.train_split) + "." + args.trainfile.split('.')[-1]
    train_dataset.to_csv(nombre_train_file)

    nombre_vali_file = args.validationfile.split('.')[0] + "_" + str(args.train_split) + "." + args.validationfile.split('.')[-1]
    validation_dataset.to_csv(nombre_vali_file)
