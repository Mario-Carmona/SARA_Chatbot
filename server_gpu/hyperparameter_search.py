#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from cProfile import label
from pathlib import Path
import argparse
import sys
import os
import logging
from typing import Dict, Tuple, List, Callable, Iterable
import ray
from ray import tune
from ray.tune.schedulers import PopulationBasedTraining
from ray.tune import CLIReporter

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
        FinetuningArguments,
        Seq2SeqTrainingArguments
    )
)

finetuning_args, training_args = parser.parse_json_file(json_file=str(BASE_PATH/CONFIG_FILE))


WORKDIR = finetuning_args.workdir

training_args.output_dir = WORKDIR + training_args.output_dir


check_output_dir(training_args)

# Ruta donde instalar las extensiones de Pytorch
os.environ["TORCH_EXTENSIONS_DIR"] = WORKDIR + "torch_extensions"

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



smoke_test = True


task_name = "conversational"

num_samples = 1


ray.init(local_mode=True)


















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



tokenized_datasets = tokenized_datasets.remove_columns(["Unnamed: 0", "source", "target"])

tokenized_datasets.set_format("torch")



def model_init():
    return BlenderbotForConditionalGeneration.from_pretrained(
        finetuning_args.model_conver,
        from_tf=bool(".ckpt" in finetuning_args.model_conver),
        config=configConver,
        torch_dtype=torch.float16
    )


data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizerConver, model=modelConver)





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
    model_init=model_init,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizerConver,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)


#trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)


tune_config = {
    "per_device_train_batch_size": 16,
    "per_device_eval_batch_size": 16,
    "num_train_epochs": tune.choice([2,3]),
    "max_steps": 1 if smoke_test else -1,  # Used for smoke test.
}

scheduler = PopulationBasedTraining(
    time_attr="training_iteration",
    metric="eval_acc",
    mode="max",
    perturbation_interval=1,
    hyperparam_mutations={
        "weight_decay": tune.uniform(0.0, 0.3),
        "learning_rate": tune.uniform(1e-5, 5e-5),
        "per_device_train_batch_size": [16, 32, 64],
    })

reporter = CLIReporter(
    parameter_columns={
        "weight_decay": "w_decay",
        "learning_rate": "lr",
        "per_device_train_batch_size": "train_bs/gpu",
        "num_train_epochs": "num_epochs"
    },
    metric_columns=[
        "eval_acc", "eval_loss", "epoch", "training_iteration"
    ])



best_trial = trainer.hyperparameter_search(
    direction="minimize",
    hp_space=lambda _: tune_config,
    backend="ray",
    n_trials=num_samples,
    resources_per_trial={
        "cpu": 1,
        "gpu": 1
    },
    scheduler=scheduler,
    keep_checkpoints_num=1,
    checkpoint_score_attr="training_iteration",
    stop={"training_iteration": 1} if smoke_test else None,
    progress_reporter=reporter,
    local_dir="./ray_results/",
    name="tune_transformer_pbt",
    log_to_file=True)





"""
best_trial = trainer.hyperparameter_search(
    direction="minimize", 
    backend="ray", 
    n_trials=2
)
"""

print(best_trial)
