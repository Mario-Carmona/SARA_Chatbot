#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

from project_arguments import ProyectArguments
from dataset_arguments import DatasetArguments
from transformers import HfArgumentParser



BASE_PATH = Path(__file__).resolve().parent
CONFIG_FILE = "config_dataset.json"


parser = HfArgumentParser(
    (
        ProyectArguments,
        DatasetArguments
    )
)

project_args, dataset_args = parser.parse_json_file(json_file=str(BASE_PATH/CONFIG_FILE))


WORKDIR = project_args.workdir









