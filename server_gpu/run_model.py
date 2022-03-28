import os
import json
import sys
import time
import argparse
import logging

import torch

from project_arguments import ProyectArguments
from model_arguments import ModelArguments
from inference_arguments import InferenceArguments
from data_training_arguments import DataTrainingArguments
from transformers import TrainingArguments, HfArgumentParser



from datasets import load_dataset, load_metric

from transformers import set_seed
from transformers.trainer_utils import is_main_process
from transformers import AutoTokenizer, GPTJConfig, GPTJForQuestionAnswering, QuestionAnsweringPipeline

import deepspeed


logger = logging.getLogger(__name__)


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parserScript = argparse.ArgumentParser()
    parserScript.add_argument(
        "--config",
        help=""
    )
    args = parserScript.parse_args()


    parser = HfArgumentParser(
        (
            ProyectArguments, 
            ModelArguments, 
            DataTrainingArguments,
            InferenceArguments, 
            TrainingArguments
        )
    )
    

    project_args, model_args, data_args, infer_args, training_args = parser.parse_json_file(json_file=args.config)


    WORKDIR = project_args.workdir


    with open(WORKDIR + model_args.generate_args_path) as file:
        generate_args = json.load(file)



    # Ruta donde instalar las extensiones de Pytorch
    os.environ["TORCH_EXTENSIONS_DIR"] = WORKDIR + "torch_extensions"


    # distributed setup
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))



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


    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
    )
    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)


    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
    logger.info("Training/evaluation parameters %s", training_args)



    # Set seed before initializing model.

    set_seed(training_args.seed)



    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.

    data_files = {}
    if data_args.train_file is not None:
        data_files["train"] = data_args.train_file
    if data_args.validation_file is not None:
        data_files["validation"] = data_args.validation_file
    extension = data_args.train_file.split(".")[-1]
    datasets = load_dataset(extension, data_files=data_files, field="data")

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


    model = GPTJForQuestionAnswering.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        revision=model_args.model_revision,
        torch_dtype=model_args.model_torch_dtype
    )


    generator = QuestionAnsweringPipeline(
        model=model,
        tokenizer=tokenizer,
        framework="pt",
        device=local_rank
    )



    # Preprocessing the datasets.
    # Preprocessing is slighlty different for training and evaluation.
    if training_args.do_train:
        column_names = datasets["train"].column_names
    else:
        column_names = datasets["validation"].column_names
    question_column_name = "question" if "question" in column_names else column_names[0]
    context_column_name = "context" if "context" in column_names else column_names[1]
    answer_column_name = "answers" if "answers" in column_names else column_names[2]



    if training_args.do_train:
        pass







    if training_args.do_eval:
        pass





    if infer_args.do_inference:
        with torch.no_grad():
            generator.model = deepspeed.init_inference(
                generator.model,
                mp_size=world_size,
                dtype=infer_args.inference_dtype,
                replace_method=infer_args.replace_method,
                replace_with_kernel_inject=infer_args.replace_with_kernel_inject,
                quantization_setting=(
                    infer_args.mlp_exra_grouping,
                    infer_args.quantize_groups
                )
            )

            inicio = time.time()


            
            string = generator("DeepSpeed is")
            os.system("nvidia-smi")

            fin = time.time()

            if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
                print(string)
            
            print(fin-inicio)

    


if __name__ == "__main__":
    main()