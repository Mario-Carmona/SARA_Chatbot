#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import argparse
import pandas as pd
import sys
import os


if __name__ == "__main__":

    BASE_PATH = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "input_dataset", 
        type = str,
        help = "El formato del archivo debe ser \'archivo.csv\'"
    )
    parser.add_argument(
        "output_dataset", 
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


    input_dataset = pd.read_csv(args.input_dataset)

    output_dataset = input_dataset.copy(deep=True)

    output_dataset["Text"] = output_dataset["Definición"] + " " + output_dataset["Respuesta"]

    output_dataset = output_dataset.drop(
        columns=["Definición", "Pregunta", "Respuesta"]
    )

    output_dataset = output_dataset.rename(columns={"Nombre":"Topic", "Tema":"Subject"})

    output_dataset.to_csv(args.output_dataset)
