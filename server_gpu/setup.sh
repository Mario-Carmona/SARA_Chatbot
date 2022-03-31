#!/bin/bash

export PATH="/opt/anaconda/anaconda3/bin:$PATH"

export PATH="/opt/anaconda/bin:$PATH"

eval "$(conda shell.bash hook)"

conda activate /mnt/homeGPU/mcarmona/mcarmona

export TFHUB_CACHE_DIR=.

# ------------------------------------

export HOME=/mnt/homeGPU/mcarmona

pip install -r requirements.txt
