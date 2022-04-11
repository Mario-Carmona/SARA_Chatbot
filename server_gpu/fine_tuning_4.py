#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from cProfile import label
from pathlib import Path
import argparse
import sys
import os
import logging
from typing import Dict, Tuple, List, Callable, Iterable

from datasets import load_dataset, load_metric

from dataclass.finetuning_arguments import FinetuningArguments
from transformers import HfArgumentParser

from transformers import Seq2SeqTrainingArguments

from transformers import DataCollatorForSeq2Seq

import transformers
from transformers import set_seed
from transformers import AutoConfig, AutoTokenizer, BlenderbotForConditionalGeneration
from transformers import EvalPrediction, Seq2SeqTrainer
from transformers.trainer_utils import is_main_process, EvaluationStrategy
from transformers.training_args import ParallelMode

from utils import (
    Seq2SeqDataCollator,
    Seq2SeqDataset,
    check_output_dir,
    save_json,
)

import torch

from sacrebleu import corpus_bleu

import numpy as np

from torch.utils.data import DataLoader




logger = logging.getLogger(__name__)




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
        FinetuningArguments
    )
)

finetuning_args, = parser.parse_json_file(json_file=str(BASE_PATH/CONFIG_FILE))


WORKDIR = finetuning_args.workdir


check_output_dir(finetuning_args)




# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO if finetuning_args.local_rank in [-1, 0] else logging.WARN,
)
logger.warning(
    "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
    finetuning_args.local_rank,
    finetuning_args.device,
    finetuning_args.n_gpu,
    bool(finetuning_args.parallel_mode == ParallelMode.DISTRIBUTED),
    finetuning_args.fp16,
)
# Set the verbosity to info of the Transformers logger (on main process only):
if is_main_process(finetuning_args.local_rank):
    transformers.utils.logging.set_verbosity_info()
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
logger.info("Training/evaluation parameters %s", finetuning_args)





# Set seed before initializing model.
set_seed(finetuning_args.seed)





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
            "min_length": finetuning_args.min_length,
            "use_cache": finetuning_args.use_cache
        }
    }
)

extra_model_params = ("encoder_layerdrop", "decoder_layerdrop", "dropout", "attention_dropout")
for p in extra_model_params:
    if getattr(finetuning_args, p, None):
        assert hasattr(configConver, p), f"({configConver.__class__.__name__}) doesn't have a `{p}` attribute"
        setattr(configConver, p, getattr(finetuning_args, p))


tokenizerConver = AutoTokenizer.from_pretrained(
    finetuning_args.model_conver_tokenizer,
    config=finetuning_args.model_conver_tokenizer_config,
    use_fast=True
)


modelConver = BlenderbotForConditionalGeneration.from_pretrained(
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
data_files["train"] = finetuning_args.train_dataset
data_files["validation"] = finetuning_args.validation_dataset
datasets = load_dataset("csv", data_files=data_files)




def preprocess_function(examples):
    inputs = [example for example in examples["source"]]
    targets = [example for example in examples["target"]]
    model_inputs = tokenizerConver(inputs, max_length=finetuning_args.max_source_length, truncation=True, padding="max_length")

    with tokenizerConver.as_target_tokenizer():
        labels = tokenizerConver(targets, max_length=finetuning_args.max_target_length, truncation=True, padding="max_length")

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


tokenized_datasets = datasets.map(preprocess_function, batched=True)


"""

tokenized_datasets = tokenized_datasets.remove_columns(["Unnamed: 0", "source", "target"])

tokenized_datasets.set_format("torch")






data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizerConver, model=modelConver)


training_args = Seq2SeqTrainingArguments(
    output_dir=finetuning_args.output_dir,
    overwrite_output_dir=finetuning_args.overwrite_output_dir,
    evaluation_strategy=finetuning_args.evaluation_strategy,
    num_train_epochs=finetuning_args.num_train_epochs,
    log_level=finetuning_args.log_level,
    logging_strategy=finetuning_args.logging_strategy,
    save_strategy=finetuning_args.save_strategy,
    load_best_model_at_end=finetuning_args.load_best_model_at_end,
    metric_for_best_model=finetuning_args.metric_for_best_model,
    greater_is_better=finetuning_args.greater_is_better,
    debug=finetuning_args.debug,
    per_device_train_batch_size=finetuning_args.per_device_train_batch_size,
    per_device_eval_batch_size=finetuning_args.per_device_eval_batch_size,
    fp16=finetuning_args.fp16,
    fp16_opt_level=finetuning_args.fp16_opt_level,
    half_precision_backend=finetuning_args.half_precision_backend,
    dataloader_drop_last=finetuning_args.dataloader_drop_last,
    learning_rate=finetuning_args.learning_rate
)



metric = load_metric("accuracy")

def compute_metrics(eval_pred: EvalPrediction):
    # No se si es el índice 0 ó 1, se podrá comprobar cuando
    # se tengan más datos porque no se si es la predicción
    # ó la máscara. Parece que es el cero porque la tercera
    # dimensión es igual a 8008 al igual que logits en la versión
    # de Pytorch y es igual al tamaño del vocabulario del modelo
    predictions = np.argmax(eval_pred.predictions[0], axis=-1)
    predictions = predictions.flatten()
    references = eval_pred.label_ids.flatten()
    return metric.compute(predictions=predictions, references=references)



trainer = Seq2SeqTrainer(
    model=modelConver,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizerConver,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)


trainer.train(resume_from_checkpoint=finetuning_args.resume_from_checkpoint)


"""