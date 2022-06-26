#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""! @brief Script para el entrenamiento de un modelo."""


##
# @file fine_tuning.py
#
# @brief Programa para el entrenamiento de un modelo.
#
# @section description_main Descripción
# Programa para el entrenamiento de un modelo.
#
# @section libraries_main Librerías/Módulos
# - Librería estándar pathlib (https://docs.python.org/3/library/pathlib.html)
#   - Acceso a la función Path
# - Librería estándar argparse (https://docs.python.org/3/library/argparse.html)
#   - Acceso a la función ArgumentParser
# - Librería estándar json (https://docs.python.org/3/library/json.html)
#   - Acceso a la función dump
# - Librería estándar sys (https://docs.python.org/3/library/sys.html)
#   - Acceso a la función exit
# - Librería estándar os (https://docs.python.org/3/library/os.html)
#   - Acceso a la función listdir
#   - Acceso a la función path.exists
#   - Acceso a la función path.join
#   - Acceso a la función environ
#   - Acceso a la función system
# - Librería estándar logging (https://docs.python.org/3/library/logging.html)
#   - Acceso a la función getLogger
#   - Acceso a la función basicConfig
#   - Acceso a la clase INFO
#   - Acceso a la clase WARN
# - Librería color
#   - Acceso a la clase bcolors
# - Librería datasets (https://huggingface.co/docs/datasets/index)
#   - Acceso a la función load_dataset
#   - Acceso a la clase Dataset
# - Librería dataclass.finetuning_arguments
#   - Acceso a la clase FinetuningArguments
# - Librería transformers (https://pypi.org/project/transformers)
#   - Acceso a la función HfArgumentParser
#   - Acceso a la clase TrainingArguments
#   - Acceso a la clase DataCollatorWithPadding
#   - Acceso a la función set_seed
#   - Acceso a la clase AutoConfig
#   - Acceso a la clase AutoTokenizer
#   - Acceso a la clase BlenderbotForConditionalGeneration
#   - Acceso a la clase EvalPrediction
#   - Acceso a la clase Trainer
#   - Acceso a la clase ParallelMode
#   - Acceso a la clase EvaluationStrategy
#   - Acceso a la función is_main_process
#   - Acceso a la función utils.logging.set_verbosity_info
#   - Acceso a la función utils.logging.enable_default_handler
#   - Acceso a la función utils.logging.enable_explicit_format
# - Librería nltk (https://www.nltk.org/)
#   - Acceso a la función translate.bleu_score.sentence_bleu
# - Librería numpy (https://numpy.org/doc/1.23/)
#   - Acceso a la función argmax
#
# @section author_doxygen_example Autor
# - Created by Mario Carmona Segovia.
#
# Copyright (c) 2022.  All rights reserved.


# Imports

from pathlib import Path
import argparse
import json
import sys
import os
import logging
from color import bcolors

from datasets import load_dataset, Dataset

from dataclass.finetuning_arguments import FinetuningArguments
from transformers import HfArgumentParser
from transformers import TrainingArguments

from transformers import DataCollatorWithPadding

import transformers
from transformers import set_seed
from transformers import AutoConfig, AutoTokenizer, BlenderbotForConditionalGeneration
from transformers import EvalPrediction, Trainer
from transformers.trainer_utils import is_main_process, EvaluationStrategy
from transformers.training_args import ParallelMode

import numpy as np

from nltk.translate.bleu_score import sentence_bleu



def main():
    """! Entrada al programa."""

    # Crear el registro del proceso de aprendizaje
    logger = logging.getLogger(__name__)

    def check_output_dir(args, expected_items=0):
        """! Comprobar el estado del directorio de salida para los resultados.
    
        @param args            Argumentos del programa
        @param expected_items  Número de elementos esperados en el directorio
        """

        # Si se cumplen las condicciones se produce un error indicando al usuario el estado del directorio de salida
        if (
            os.path.exists(args.output_dir)
            and len(os.listdir(args.output_dir)) > expected_items
            and args.do_train
            and not args.overwrite_output_dir
        ):
            raise ValueError(
                f"Output directory ({args.output_dir}) already exists and "
                f"has {len(os.listdir(args.output_dir))} items in it (expected {expected_items} items). "
                "Use --overwrite_output_dir to overcome."
            )


    def save_json(content, path, indent=4, **json_dump_kwargs):
        """! Guardar el contenido en el archivo indicado.
    
        @param args            Argumentos del programa
        @param expected_items  Número de elementos esperados en el directorio
        """

        # Apertura del archivo en modo escritura
        with open(path, "w") as f:
            # Volcado de la información del diccionario en el archivo JSON
            json.dump(content, f, indent=indent, sort_keys=True, **json_dump_kwargs)


    def handle_metrics(split, metrics, output_dir):
        """! Registro y guardado de las métricas.
    
        @param split       Sección del entrenamiento al que pertencen las métricas (entrenamiento o validación)
        @param metrics     Metricas a guardar
        @param output_dir  Archivo donde guardar las métricas
        """

        # Registro de información de las métricas
        logger.info(bcolors.OK + f"***** {split} metrics *****" + bcolors.RESET)
        for key in sorted(metrics.keys()):
            logger.info(bcolors.OK + f"  {key} = {metrics[key]}" + bcolors.RESET)
        
        # Guardado de las métricas
        save_json(metrics, os.path.join(output_dir, f"{split}_results.json"))


    def preprocess_function(examples):
        """! Preprocesado de los ejemplos.
    
        @param examples  Lista de ejemplos

        @return Datos preprocesados
        """

        # Extracción de los tokens de la columna source de los ejemplos
        model_inputs = tokenizerConver(list(examples["source"]), max_length=finetuning_args.max_source_length, truncation=True, padding="max_length")

        # Extracción de los tokens de la columna target de los ejemplos
        labels = tokenizerConver(list(examples["target"]), max_length=finetuning_args.max_target_length, truncation=True, padding="max_length")

        # Asignación como etiquetas los tokens de la columna target de los ejemplos
        model_inputs["labels"] = labels["input_ids"]

        return model_inputs


    def compute_metrics(eval_pred: EvalPrediction):
        """! Cálculo de las métricas.
    
        @param eval_pred  Predicción del modelo

        @return Resultados del cálculo de las métricas
        """


        # No se si es el índice 0 ó 1, se podrá comprobar cuando
        # se tengan más datos porque no se si es la predicción
        # ó la máscara. Parece que es el cero porque la tercera
        # dimensión es igual a 8008 al igual que logits en la versión
        # de Pytorch y es igual al tamaño del vocabulario del modelo

        # Cálculo del máximo de las predicciones
        predictions = np.argmax(eval_pred.predictions[0], axis=-1)

        # Extracción de los tokens de las predicciones
        batch_pred = tokenizerConver.batch_decode(predictions, skip_special_tokens=True)
        y_pred = []
        for sentence in batch_pred:
            sentence = sentence.strip()
            y_pred.append(sentence)

        batch_labels = tokenizerConver.batch_decode(eval_pred.label_ids, skip_special_tokens=True)
        y_true = []
        for sentence in batch_labels:
            sentence = sentence.strip()
            y_true.append(sentence)
        
        bleu_score = 0.0
        for i in y_true:
            score = sentence_bleu(y_pred, i)
            bleu_score += score
        bleu_score /= len(y_true)

        return {"bleu": bleu_score}



    # Analizador de argumentos
    parser = argparse.ArgumentParser()

    # Añadir un argumento para el archivo de configuración
    parser.add_argument(
        "config_file", 
        type = str,
        help = "El formato del archivo debe ser \'config.json\'"
    )

    try:
        # Obtención de los argumentos
        args = parser.parse_args()

        # Comprobaciones de los argumentos
        assert args.config_file.split('.')[-1] == "json"
    except:
        # Visualización de las ayudas de los argumentos en caso de error en la comprobación de los mismos
        parser.print_help()

        # Finalización forzosa del programa
        sys.exit(0)

    # Constantes globales
    ## Ruta base del programa python.
    BASE_PATH = Path(__file__).resolve().parent

    ## Archivo de configuración
    CONFIG_FILE = args.config_file

    # Analizador de argumentos de la librería transformers
    parser = HfArgumentParser(
        (
            FinetuningArguments,
            TrainingArguments
        )
    )

    # Obtención de los argumentos de finetuning y de entrenamiento
    finetuning_args, training_args = parser.parse_json_file(json_file=str(BASE_PATH/CONFIG_FILE))

    ## Directorio de trabajo del programa
    WORKDIR = finetuning_args.workdir

    # Creación de la ruta al directorio de salida de los resultados
    training_args.output_dir = os.path.join(WORKDIR, training_args.output_dir)

    # Comprobación del estado del directorio de salida
    check_output_dir(training_args)


    # Fijar ruta donde instalar las extensiones de Pytorch
    os.environ["TORCH_EXTENSIONS_DIR"] = os.path.join(WORKDIR, "torch_extensions")

    # Fijar como desactivado el paralelismo al convertir las frases en tokens para evitar problemas
    os.environ["TOKENIZERS_PARALLELISM"] = "false"  # To avoid warnings about



    # Fijar formato del registro
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )

    # Registro de aviso sobre el estado del entorno del proceso
    logger.warning(
        bcolors.WARNING + "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s" + bcolors.RESET,
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.parallel_mode == ParallelMode.DISTRIBUTED),
        training_args.fp16,
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    # Si se trata del proceso principal
    if is_main_process(training_args.local_rank):
        # Activar el registro de información del registro de Transformers
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    
    # Registro de información sobre los parámetros de entrenamiento
    logger.info(bcolors.OK + "Training/evaluation parameters %s" + bcolors.RESET, training_args)

    # Fijar semilla del generador de números aleatorios
    set_seed(training_args.seed)


    # Carga de la configuración del modelo conversacional
    configConver = AutoConfig.from_pretrained(
        finetuning_args.model_conver_config,
        task_specific_params={
            finetuning_args.task: {
                "max_length": finetuning_args.max_length,
                "min_length": finetuning_args.min_length
            }
        }
    )

    # Carga del tokenizer del modelo conversacional
    tokenizerConver = AutoTokenizer.from_pretrained(
        finetuning_args.model_conver_tokenizer,
        config=finetuning_args.model_conver_tokenizer_config,
        use_fast=True,
        add_prefix_space=True
    )

    # Carga del modelo conversacional
    modelConver = BlenderbotForConditionalGeneration.from_pretrained(
        finetuning_args.model_conver,
        from_tf=bool(".ckpt" in finetuning_args.model_conver),
        config=configConver
    )


    # Carga de los datasets de entrenamiento y validación
    data_files = {}

    # Si se realiza el entrenamiento
    if training_args.do_train:
        # Selección del dataset de entrenamiento
        data_files["train"] = finetuning_args.train_dataset

    # Si se realiza la evaluación
    if training_args.do_eval or training_args.evaluation_strategy != EvaluationStrategy.NO:    
        # Selección del dataset de validación
        data_files["validation"] = finetuning_args.validation_dataset

    # Carga de los datasets seleccionados
    datasets = load_dataset("csv", data_files=data_files)

    # Si se realiza el entrenamiento
    if training_args.do_train:
        # Selección del número de filas del dataset de entrenamiento
        if finetuning_args.n_train != -1:
            datasets["train"] = Dataset.from_dict(datasets["train"][:finetuning_args.n_train])
        else:
            datasets["train"] = Dataset.from_dict(datasets["train"][:])
    
    # Si se realiza la evaluación
    if training_args.do_eval or training_args.evaluation_strategy != EvaluationStrategy.NO:
        # Selección del número de filas del dataset de validación
        if finetuning_args.n_val != -1:
            datasets["validation"] = Dataset.from_dict(datasets["validation"][:finetuning_args.n_val])
        else:
            datasets["validation"] = Dataset.from_dict(datasets["validation"][:])


    # Asignar como token de padding el token de final de frase
    tokenizerConver.pad_token = tokenizerConver.eos_token

    # Preprocesado de los datasets
    tokenized_datasets = datasets.map(preprocess_function, batched=True)

    # Eliminación de los títulos de las columnas
    tokenized_datasets = tokenized_datasets.remove_columns(["source", "target"])

    # Recolector de datos
    data_collator = DataCollatorWithPadding(tokenizer=tokenizerConver)

    # Creación del objeto de entrenamiento
    trainer = Trainer(
        model=modelConver,
        args=training_args,
        train_dataset=tokenized_datasets["train"] if training_args.do_train else None,
        eval_dataset=tokenized_datasets["validation"] if training_args.do_eval or training_args.evaluation_strategy != EvaluationStrategy.NO else None,
        tokenizer=tokenizerConver,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    # Variable para guardar las métricas generadas
    all_metrics = {}

    # Entrenamiento
    if training_args.do_train:
        # Registro de información sobre el inicio del entrenamiento
        logger.info(bcolors.OK + "*** Train ***" + bcolors.RESET)

        # Realizar entrenamiento y obtener los resultados del mismo
        train_result = trainer.train(
            resume_from_checkpoint=training_args.resume_from_checkpoint
        )

        # Extraer las métricas generadas de los resultados del entrenamiento
        metrics = train_result.metrics

        # Guardado del número de elementos usados para el entrenamiento
        metrics["train_n_objs"] = finetuning_args.n_train

        # Guardado del modelo obtenido tras el entrenamiento, esta función no guarda el estado del modelo
        trainer.save_model()

        # Si es el proceso principal
        if trainer.is_world_process_zero():
            # Registro y guardado de las métricas del entrenamiento
            handle_metrics("train", metrics, training_args.output_dir)

            # Actualización de las métricas generales
            all_metrics.update(metrics)

            # Guardado del estado del modelo obtenido tras el entrenamiento
            trainer.state.save_to_json(os.path.join(training_args.output_dir, "trainer_state.json"))

            # Por conveniencia se vuelve a guardar el tokenizer del modelo obtenido tras el entrenamiento
            tokenizerConver.save_pretrained(training_args.output_dir)

    # Evaluación
    if training_args.do_eval:
        # Registro de información sobre el inicio de la evaluación
        logger.info(bcolors.OK + "*** Evaluate ***" + bcolors.RESET)

        # Realizar la evaluación y obtener los resultados de la misma
        metrics = trainer.evaluate(
            metric_key_prefix="val"
        )

        # Guardado del número de elementos usados para la evaluación
        metrics["val_n_objs"] = finetuning_args.n_val

        # Guardado del error obtenido de la evaluación
        metrics["val_loss"] = round(metrics["val_loss"], 4)

        # Si es el proceso principal
        if trainer.is_world_process_zero():
            # Registro y guardado de las métricas de la evaluación
            handle_metrics("val", metrics, training_args.output_dir)

            # Actualización de las métricas generales
            all_metrics.update(metrics)

    # Si es el proceso principal
    if trainer.is_world_process_zero():
        # Guardado de las métricas de todo el proceso de finetuning
        save_json(all_metrics, os.path.join(training_args.output_dir, "all_results.json"))

    return all_metrics



if __name__ == "__main__":
    main()
