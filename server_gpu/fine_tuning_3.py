#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import argparse
import sys
import os
import logging
from typing import Dict, Tuple, List, Callable, Iterable

from datasets import load_dataset, load_metric

from project_arguments import ProyectArguments
from model_arguments import ModelArguments
from generate_arguments import GenerateArguments
from data_training_arguments import DataTrainingArguments
from transformers import Seq2SeqTrainingArguments, HfArgumentParser

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


def handle_metrics(split, metrics, output_dir):
    """
    Log and save metrics
    Args:
    - split: one of train, val, test
    - metrics: metrics dict
    - output_dir: where to save the metrics
    """

    logger.info(f"***** {split} metrics *****")
    for key in sorted(metrics.keys()):
        logger.info(f"  {key} = {metrics[key]}")
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
        ProyectArguments, 
        ModelArguments,
        GenerateArguments,
        DataTrainingArguments,
        Seq2SeqTrainingArguments
    )
)

project_args, model_args, generate_args, data_args, training_args = parser.parse_json_file(json_file=str(BASE_PATH/CONFIG_FILE))


WORKDIR = project_args.workdir
training_args.output_dir = WORKDIR + training_args.output_dir


check_output_dir(training_args)




# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
)
logger.warning(
    "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
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
logger.info("Training/evaluation parameters %s", training_args)





# Set seed before initializing model.
set_seed(training_args.seed)





# Load pretrained model and tokenizer
#
# Distributed training:
# The .from_pretrained methods guarantee that only one local process can concurrently
# download model & vocab.
configConver = AutoConfig.from_pretrained(
    WORKDIR + model_args.model_conver_config,
    task_specific_params={
        data_args.task: {
            "do_sample": generate_args.do_sample,
            "temperature": generate_args.temperature,
            "top_p": generate_args.top_p,
            "max_length": generate_args.max_length,
            "min_length": generate_args.min_length,
            "use_cache": generate_args.use_cache
        }
    }
)

extra_model_params = ("encoder_layerdrop", "decoder_layerdrop", "dropout", "attention_dropout")
for p in extra_model_params:
    if getattr(training_args, p, None):
        assert hasattr(configConver, p), f"({configConver.__class__.__name__}) doesn't have a `{p}` attribute"
        setattr(configConver, p, getattr(training_args, p))


tokenizerConver = AutoTokenizer.from_pretrained(
    WORKDIR + model_args.model_conver_tokenizer,
    config=WORKDIR + model_args.model_conver_tokenizer_config,
    use_fast=True
)


modelConver = BlenderbotForConditionalGeneration.from_pretrained(
    WORKDIR + model_args.model_conver,
    from_tf=bool(".ckpt" in model_args.model_conver),
    config=configConver,
    torch_dtype=torch.float16
)



# set num_beams for evaluation
if data_args.eval_beams is None:
    data_args.eval_beams = modelConver.config.num_beams



dataset_class = Seq2SeqDataset




# Get datasets
train_dataset = (
    dataset_class(
        tokenizerConver,
        type_path="train",
        data_dir=data_args.data_dir,
        n_obs=data_args.n_train,
        max_target_length=data_args.max_target_length,
        max_source_length=data_args.max_source_length,
        prefix=modelConver.config.prefix or "",
    )
    if training_args.do_train
    else None
)
eval_dataset = (
    dataset_class(
        tokenizerConver,
        type_path="val",
        data_dir=data_args.data_dir,
        n_obs=data_args.n_val,
        max_target_length=data_args.val_max_target_length,
        max_source_length=data_args.max_source_length,
        prefix=modelConver.config.prefix or "",
    )
    if training_args.do_eval or training_args.evaluation_strategy != EvaluationStrategy.NO
    else None
)


print(type(train_dataset))

"""
# TODO: Once the fix lands in a Datasets release, remove the _local here and the squad_v2_local folder.
metric = load_metric("bleu")


print(metric.inputs_description)

def compute_metrics(p: EvalPrediction):
    return metric.compute(predictions=p.predictions, references=p.label_ids)



trainer = Seq2SeqTrainer(
    model=modelConver,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=Seq2SeqDataCollator(tokenizerConver, data_args, training_args.tpu_num_cores),
    compute_metrics=compute_metrics,
    tokenizer=tokenizerConver,
)






all_metrics = {}
# Training
if training_args.do_train:
    logger.info("*** Train ***")

    os.system("nvidia-smi")

    train_result = trainer.train()
    metrics = train_result.metrics
    metrics["train_n_objs"] = data_args.n_train

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
    logger.info("*** Evaluate ***")

    metrics = trainer.evaluate(
        metric_key_prefix="val", max_length=data_args.val_max_target_length, num_beams=data_args.eval_beams
    )
    metrics["val_n_objs"] = data_args.n_val
    metrics["val_loss"] = round(metrics["val_loss"], 4)

    if trainer.is_world_process_zero():

        handle_metrics("val", metrics, training_args.output_dir)
        all_metrics.update(metrics)




if trainer.is_world_process_zero():
    save_json(all_metrics, os.path.join(training_args.output_dir, "all_results.json"))



print(all_metrics)



"""
