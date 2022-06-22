#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""! @brief Script para la obtención de un dataset con el formato adecuado para realizar el finetuning."""


##
# @file generate_finetuning_dataset.py
#
# @brief Programa para la obtención de un dataset con el formato adecuado para realizar el finetuning.
#
# @section description_main Descripción
# Programa para la obtención de un dataset con el formato adecuado para realizar el finetuning.
#
# @section libraries_main Librerías/Módulos
# - Librería pandas (https://pandas.pydata.org/docs/)
#   - Acceso a la función DataFrame
#   - Acceso a la función read_csv
# - Librería estándar argparse (https://docs.python.org/3/library/argparse.html)
#   - Acceso a la función ArgumentParser
# - Librería utils
#   - Acceso a la función save_csv
# - Librería estándar sys (https://docs.python.org/3/library/sys.html)
#   - Acceso a la función exit
# - Librería estándar os (https://docs.python.org/3/library/os.html)
#   - Acceso a la función path.join
# - Librería estándar pathlib (https://docs.python.org/3/library/pathlib.html)
#   - Acceso a la función Path
#
# @section author_doxygen_example Autor
# - Created by Mario Carmona Segovia.
#
# Copyright (c) 2022.  All rights reserved.


# Imports

from pathlib import Path
import argparse
import pandas as pd
import sys
import os
from utils import save_csv



def obtain_finetuning_dataset(train_dataset, valid_dataset):
    """! Obtener dataset con el formato para realizar el finetuning.
    
    @param train_dataset  Dataframe de entrenamiento
    @param valid_dataset  Dataframe de validación
    
    @return Dataframe de entranamiento con formato para el finetuning.
    @return Dataframe de validación con formato para el finetuning.
    """

    # Generación del Dataframe de entrenamiento con el formato para el finetuning
    train_s_t = pd.DataFrame({
        "source": train_dataset.Question.to_list(),
        "target": train_dataset.Answer.to_list()
    })

    # Generación del Dataframe de validación con el formato para el finetuning
    validation_s_t = pd.DataFrame({
        "source": valid_dataset.Question.to_list(),
        "target": valid_dataset.Answer.to_list()
    })

    return train_s_t, validation_s_t


def main():
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

    save_csv(validation_s_t, os.path.join(dir,"validation.csv"))


if __name__ == "__main__":
    main()    
