import os
import json
import sys

import torch

from model_arguments import ModelArguments
from data_training_arguments import DataTrainingArguments
from transformers import TrainingArguments, HfArgumentParser

from transformers import set_seed

from datasets import load_dataset

from transformers import AutoTokenizer, GPTJConfig, GPTJForQuestionAnswering, QuestionAnsweringPipeline



def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    '''
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    '''
    parser = HfArgumentParser((ModelArguments, TrainingArguments))
    '''
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    with open(WORKDIR + model_args.generate_args_path) as file:
        generate_args = json.load(file)
    '''
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, training_args = parser.parse_args_into_dataclasses()

    WORKDIR = model_args.workdir

    with open(WORKDIR + model_args.generate_args_path) as file:
        generate_args = json.load(file)



    


    # Ruta donde instalar las extensiones de Pytorch
    os.environ["TORCH_EXTENSIONS_DIR"] = WORKDIR + "torch_extensions"


    # distributed setup
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))



    '''
    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty."
            "Use --overwrite_output_dir to overcome."
        )
    '''

    # Set seed before initializing model.
    '''
    set_seed(training_args.seed)
    '''

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    '''
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        datasets = load_dataset(data_args.dataset_name, data_args.dataset_config_name)
    else:
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
        extension = data_args.train_file.split(".")[-1]
        datasets = load_dataset(extension, data_files=data_files, field="data")
    '''
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.
    

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = GPTJConfig.from_pretrained(
        WORKDIR + model_args.model_config_name if model_args.model_config_name else WORKDIR + model_args.model_name_or_path,
        revision=model_args.model_revision,
        torch_dtype=model_args.torch_dtype,
        task_specific_params=generate_args
    )


    tokenizer = AutoTokenizer.from_pretrained(
        WORKDIR + model_args.tokenizer_name if model_args.tokenizer_name else WORKDIR + model_args.model_name_or_path,
        config=WORKDIR + model_args.tokenizer_config_name if model_args.tokenizer_config_name else None,
        use_fast=True,
        revision=model_args.model_revision,
    )

    print(config)


    '''
    model = GPTJForQuestionAnswering.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        revision=model_args.model_revision,
        torch_dtype=model_args.model_torch_dtype
    )


    generator = QuestionAnsweringPipeline(
        task="question-answering",
        model=model,
        tokenizer=tokenizer,
        framework="pt",
        device=local_rank
    )
    '''



if __name__ == "__main__":
    main()