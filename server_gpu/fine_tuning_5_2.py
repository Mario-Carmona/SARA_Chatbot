#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from cProfile import label
from pathlib import Path
import argparse
import json
import sys
import os
import logging
from typing import Dict, Tuple, List, Callable, Iterable
from color import bcolors

from datasets import load_dataset, load_metric, Dataset
import torch.nn as nn

from dataclass.finetuning_arguments import FinetuningArguments
from transformers import HfArgumentParser
from transformers import Seq2SeqTrainingArguments

from transformers import DataCollatorForSeq2Seq

import transformers
from transformers import set_seed
from transformers import AutoConfig, AutoTokenizer, AutoModelForSeq2SeqLM, BlenderbotForConditionalGeneration
from transformers import EvalPrediction, Seq2SeqTrainer
from transformers.trainer_utils import is_main_process, EvaluationStrategy
from transformers.training_args import ParallelMode

import torch

from sacrebleu import corpus_bleu

import numpy as np

from torch.utils.data import DataLoader









def main():

    logger = logging.getLogger(__name__)

    def check_output_dir(args, expected_items=0):
        """
        Checks whether to bail out if output_dir already exists and has more than expected_items in it
        `args`: needs to have the following attributes of `args`:
        - output_dir
        - do_train
        - overwrite_output_dir
        `expected_items`: normally 0 (default) - i.e. empty dir, but in some cases a few files are expected (e.g. recovery from OOM)
        """
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
        with open(path, "w") as f:
            json.dump(content, f, indent=indent, sort_keys=True, **json_dump_kwargs)


    def handle_metrics(split, metrics, output_dir):
        """
        Log and save metrics
        Args:
        - split: one of train, val, test
        - metrics: metrics dict
        - output_dir: where to save the metrics
        """

        logger.info(bcolors.OK + f"***** {split} metrics *****" + bcolors.RESET)
        for key in sorted(metrics.keys()):
            logger.info(bcolors.OK + f"  {key} = {metrics[key]}" + bcolors.RESET)
        save_json(metrics, os.path.join(output_dir, f"{split}_results.json"))





    parser = argparse.ArgumentParser()

    parser.add_argument(
        "config_file", 
        type = str,
        help = "El formato del archivo debe ser \'config.json\'"
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
            FinetuningArguments,
            Seq2SeqTrainingArguments
        )
    )

    finetuning_args, training_args = parser.parse_json_file(json_file=str(BASE_PATH/CONFIG_FILE))


    WORKDIR = finetuning_args.workdir

    training_args.output_dir = os.path.join(WORKDIR, training_args.output_dir)


    check_output_dir(training_args)


    # Ruta donde instalar las extensiones de Pytorch
    os.environ["TORCH_EXTENSIONS_DIR"] = os.path.join(WORKDIR, "torch_extensions")

    os.environ["TOKENIZERS_PARALLELISM"] = "false"  # To avoid warnings about



    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        bcolors.WARNING + "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s" + bcolors.RESET,
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.parallel_mode == ParallelMode.DISTRIBUTED),
        training_args.fp16,
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info(bcolors.OK + "Training/evaluation parameters %s" + bcolors.RESET, training_args)





    # Set seed before initializing model.
    set_seed(training_args.seed)





    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    configConver = AutoConfig.from_pretrained(
        finetuning_args.model_conver_config,
        task_specific_params={
            finetuning_args.task: {
                "do_sample": finetuning_args.do_sample,
                "temperature": finetuning_args.temperature,
                "top_p": finetuning_args.top_p,
                "max_length": finetuning_args.max_length,
                "min_length": finetuning_args.min_length
            }
        }
    )

    extra_model_params = ("encoder_layerdrop", "decoder_layerdrop", "dropout", "attention_dropout")
    for p in extra_model_params:
        if getattr(training_args, p, None):
            assert hasattr(configConver, p), f"({configConver.__class__.__name__}) doesn't have a `{p}` attribute"
            setattr(configConver, p, getattr(training_args, p))

    tokenizerConver = AutoTokenizer.from_pretrained(
        finetuning_args.model_conver_tokenizer,
        config=finetuning_args.model_conver_tokenizer_config,
        use_fast=True
    )

    modelConver = AutoModelForSeq2SeqLM.from_pretrained(
        finetuning_args.model_conver,
        from_tf=bool(".ckpt" in finetuning_args.model_conver),
        config=configConver,
        torch_dtype=torch.float16
    )



    # set num_beams for evaluation
    if finetuning_args.eval_beams is None:
        finetuning_args.eval_beams = modelConver.config.num_beams




    # Carga de los datasets
    data_files = {}
    if training_args.do_train:
        data_files["train"] = finetuning_args.train_dataset
    if training_args.do_eval or training_args.evaluation_strategy != EvaluationStrategy.NO:    
        data_files["validation"] = finetuning_args.validation_dataset
    datasets = load_dataset("csv", data_files=data_files)

    if training_args.do_train:
        datasets["train"] = Dataset.from_dict(datasets["train"][:finetuning_args.n_train])
    if training_args.do_eval or training_args.evaluation_strategy != EvaluationStrategy.NO:
        datasets["validation"] = Dataset.from_dict(datasets["validation"][:finetuning_args.n_val])



    def preprocess_function(examples):
        inputs = [example for example in examples["source"]]
        targets = [example for example in examples["target"]]
        model_inputs = tokenizerConver(inputs, max_length=finetuning_args.max_source_length, truncation=True, padding="max_length")

        with tokenizerConver.as_target_tokenizer():
            labels = tokenizerConver(targets, max_length=finetuning_args.max_target_length, truncation=True, padding="max_length")

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs




    tokenized_datasets = datasets.map(preprocess_function, batched=True)



    tokenized_datasets = tokenized_datasets.remove_columns(["Unnamed: 0", "source", "target"])

    tokenized_datasets.set_format("torch")






    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizerConver, model=modelConver)





    metric = load_metric("accuracy")

    def compute_metrics(eval_pred: EvalPrediction):
        # No se si es el índice 0 ó 1, se podrá comprobar cuando
        # se tengan más datos porque no se si es la predicción
        # ó la máscara. Parece que es el cero porque la tercera
        # dimensión es igual a 8008 al igual que logits en la versión
        # de Pytorch y es igual al tamaño del vocabulario del modelo
        predictions, labels = eval_pred
        predictions = np.argmax(predictions[0], axis=-1)

        result_metric = metric.compute(predictions=predictions, references=labels)


        
        #loss = nn.CrossEntropyLoss(predictions, )

        print("-------------------------------------")

        print(eval_pred)

        print("-------------------------------------")

        print(result_metric)

        print("-------------------------------------")

        aux = input("---------->")


        return result_metric



    trainer = Seq2SeqTrainer(
        model=modelConver,
        args=training_args,
        train_dataset=tokenized_datasets["train"] if training_args.do_train else None,
        eval_dataset=tokenized_datasets["validation"] if training_args.do_eval or training_args.evaluation_strategy != EvaluationStrategy.NO else None,
        tokenizer=tokenizerConver,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )


    os.system("nvidia-smi")


    all_metrics = {}
    # Training
    if training_args.do_train:
        logger.info(bcolors.OK + "*** Train ***" + bcolors.RESET)

        train_result = trainer.train(
            resume_from_checkpoint=training_args.resume_from_checkpoint
        )
        metrics = train_result.metrics
        metrics["train_n_objs"] = finetuning_args.n_train

        trainer.save_model()  # this also saves the tokenizer

        if trainer.is_world_process_zero():
            handle_metrics("train", metrics, training_args.output_dir)
            all_metrics.update(metrics)

            # Need to save the state, since Trainer.save_model saves only the tokenizer with the model
            trainer.state.save_to_json(os.path.join(training_args.output_dir, "trainer_state.json"))

            # For convenience, we also re-save the tokenizer to the same directory,
            # so that you can share your model easily on huggingface.co/models =)
            tokenizerConver.save_pretrained(training_args.output_dir)


    # Evaluation
    if training_args.do_eval:
        logger.info(bcolors.OK + "*** Evaluate ***" + bcolors.RESET)

        metrics = trainer.evaluate(
            metric_key_prefix="val", max_length=finetuning_args.val_max_target_length, num_beams=finetuning_args.eval_beams
        )
        metrics["val_n_objs"] = finetuning_args.n_val
        metrics["val_loss"] = round(metrics["val_loss"], 4)

        if trainer.is_world_process_zero():

            handle_metrics("val", metrics, training_args.output_dir)
            all_metrics.update(metrics)


    if trainer.is_world_process_zero():
        save_json(all_metrics, os.path.join(training_args.output_dir, "all_results.json"))

    return all_metrics



if __name__ == "__main__":
    main()
