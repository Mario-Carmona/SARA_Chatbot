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





data_files = {}
data_files["train"] = WORKDIR + "datasets/v1/split_0.8/train_EN.csv"
data_files["validation"] = WORKDIR + "datasets/v1/split_0.8/val_EN.csv"
datasets = load_dataset("csv", data_files=data_files)




def preprocess_function(examples):
    inputs = [example for example in examples["source"]]
    targets = [example for example in examples["target"]]
    model_inputs = tokenizerConver(inputs, max_length=128, truncation=True, padding="max_length")

    with tokenizerConver.as_target_tokenizer():
        labels = tokenizerConver(targets, max_length=128, truncation=True, padding="max_length")

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


tokenized_datasets = datasets.map(preprocess_function, batched=True)


"""

from transformers import DataCollatorForSeq2Seq

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizerConver, model=modelConver)




training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=1,
)


metric = load_metric("f1")


def compute_metrics(eval_pred: EvalPrediction):
    logits, labels = eval_pred
    print(labels)
    predictions = np.argmax(logits, axis=2)
    return metric.compute(predictions=predictions, references=labels)


def compute_metrics(p: EvalPrediction):
    pred_flat = np.argmax(p.predictions, axis=1).flatten()
    logits, labels = p
    print(logits)
    print(label)
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


trainer = Seq2SeqTrainer(
    model=modelConver,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizerConver,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)



trainer.train()

"""



tokenized_datasets = tokenized_datasets.remove_columns(["Unnamed: 0", "source", "target"])

tokenized_datasets.set_format("torch")


from torch.utils.data import DataLoader

train_dataloader = DataLoader(tokenized_datasets["train"], shuffle=True, batch_size=8)
eval_dataloader = DataLoader(tokenized_datasets["validation"], batch_size=8)



from torch.optim import AdamW

optimizer = AdamW(modelConver.parameters(), lr=5e-5)


from transformers import get_scheduler

num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
modelConver.to(device)




from tqdm.auto import tqdm

progress_bar = tqdm(range(num_training_steps))

modelConver.train()
for epoch in range(num_epochs):
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = modelConver(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)



metric = load_metric("accuracy")
modelConver.eval()
for batch in eval_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = modelConver(**batch)

    logits = outputs.logits
    print(logits)
    predictions = torch.argmax(logits, dim=-1)
    predictions = predictions.flatten()
    print(batch["labels"])
    
    metric.add_batch(predictions=predictions, references=batch["labels"].flatten())

metric.compute()


