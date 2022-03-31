#!/usr/bin/env bash

PATH="/opt/anaconda/anaconda3/bin:$PATH"

PATH="/opt/anaconda/bin:$PATH"

eval "$(conda shell.bash hook)"

conda activate /mnt/homeGPU/mcarmona/mcarmona

TFHUB_CACHE_DIR=.


HOME=/mnt/homeGPU/mcarmona

pip install -r requirements.txt
