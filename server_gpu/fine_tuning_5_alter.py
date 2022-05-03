#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


### CONFIG
weights = "/mnt/homeGPU/mcarmona/prueba/microsoft/DialoGPT-medium"

# Initialize tokenizer and model
print("Loading model... ", end='', flush=True)
tokenizer = AutoTokenizer.from_pretrained(weights, add_prefix_space=True)
model = AutoModelForCausalLM.from_pretrained(weights)
print('DONE')

### FINE TUNING ###
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, DataCollatorWithPadding



data_files = {}
data_files["train"] = "/mnt/homeGPU/mcarmona/server_gpu/datasets/v3/split_0.8_Adulto/train.csv"
data_files["validation"] = "/mnt/homeGPU/mcarmona/server_gpu/datasets/v3/split_0.8_Adulto/validation.csv"
datasets = load_dataset("csv", data_files=data_files)

datasets["train"] = Dataset.from_dict(datasets["train"][:2])
datasets["validation"] = Dataset.from_dict(datasets["validation"][:2])


tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token = tokenizer.eos_token

"""
def tokenize_function(example):
    return tokenizer([example["source"], example["target"]], truncation=True, is_split_into_words=True)
"""
def tokenize_function(examples):
    inputs = [example for example in examples["source"]]
    targets = [example for example in examples["target"]]
    model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding="max_length")

    labels = tokenizer(targets, max_length=128, truncation=True, padding="max_length")

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_datasets = datasets.map(tokenize_function, batched=True)


print(tokenized_datasets)

# Poner los elementos de source y target de cada línea en una lista en el preprocesado
aux = input()

"""
raw_datasets = load_dataset("glue", "mrpc")

tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(example):
    return tokenizer([example["sentence1"], example["sentence2"]], truncation=True)

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
"""
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


from transformers import TrainingArguments
training_args = TrainingArguments("test-trainer")


from transformers import Trainer
trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
)

trainer.train()
