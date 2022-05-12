#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from cProfile import label
from pathlib import Path
import argparse
import sys
import os
import logging
from typing import Dict, Tuple, List, Callable, Iterable

from datasets import load_dataset, load_metric, Dataset

from dataclass.finetuning_arguments import FinetuningArguments
from transformers import Seq2SeqTrainingArguments, HfArgumentParser

import transformers
from transformers import set_seed
from transformers import AutoConfig, AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM, BlenderbotForConditionalGeneration
from transformers import EvalPrediction, Seq2SeqTrainer
from transformers.trainer_utils import is_main_process, EvaluationStrategy
from transformers.training_args import ParallelMode


import torch

from sacrebleu import corpus_bleu

import numpy as np

from torch.utils.data import DataLoader




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

tokenizerConver = AutoTokenizer.from_pretrained(
    finetuning_args.model_conver_tokenizer,
    config=finetuning_args.model_conver_tokenizer_config,
    use_fast=True
)

tokenizerConver.pad_token = tokenizerConver.eos_token

modelConver = AutoModelForSeq2SeqLM.from_pretrained(
    finetuning_args.model_conver,
    from_tf=bool(".ckpt" in finetuning_args.model_conver),
    config=configConver,
    torch_dtype=torch.float16
)







data_files = {}
if training_args.do_train:
    data_files["train"] = finetuning_args.train_dataset
if training_args.do_eval or training_args.evaluation_strategy != EvaluationStrategy.NO:    
    data_files["validation"] = finetuning_args.validation_dataset
datasets = load_dataset("csv", data_files=data_files)

if training_args.do_train and finetuning_args.n_train != -1:
    datasets["train"] = Dataset.from_dict(datasets["train"][:finetuning_args.n_train])
if (training_args.do_eval or training_args.evaluation_strategy != EvaluationStrategy.NO) and finetuning_args.n_val != -1:
    datasets["validation"] = Dataset.from_dict(datasets["validation"][:finetuning_args.n_val])





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

train_dataloader = DataLoader(tokenized_datasets["train"], shuffle=True, batch_size=4)
eval_dataloader = DataLoader(tokenized_datasets["validation"], batch_size=4)



from torch.optim import AdamW

optimizer = AdamW(modelConver.parameters(), lr=3e-5)


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

"""
modelConver.train()
for epoch in range(num_epochs):
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = modelConver(**batch)
        loss = outputs.loss
        print(loss)
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
    predictions = torch.argmax(logits, dim=-1)
    predictions = predictions.flatten()
    
    metric.add_batch(predictions=predictions, references=batch["labels"].flatten())

print(metric.compute())

"""

metric = load_metric("accuracy")

modelConver.eval()
for batch in eval_dataloader:
    #print(batch.items())
    batch = {k: v.to(device) for k, v in batch.items()}
    inputs = {"input_ids": batch["input_ids"]}
    with torch.no_grad():
        outputs = modelConver(**inputs)

    print(type(outputs))
    logits = outputs.last_hidden_state
    predictions = torch.argmax(logits, dim=-1)
    predictions = predictions.flatten()
    
    metric.add_batch(predictions=predictions, references=batch["labels"].flatten())

print(metric.compute())

modelConver.train()
for epoch in range(num_epochs):
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        inputs = {"input_ids": batch["input_ids"]}
        outputs = modelConver(**inputs)
        print(outputs.keys())
        loss = outputs.loss
        print(loss)
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)
    
    for batch in eval_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = modelConver(**batch)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        predictions = predictions.flatten()
        
        metric.add_batch(predictions=predictions, references=batch["labels"].flatten())

    print(metric.compute())

