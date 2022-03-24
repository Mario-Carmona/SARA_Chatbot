#!/bin/bash


#SBATCH --job-name Prueba                 # Nombre del proceso

#SBATCH --partition dios   # Cola para ejecutar

#SBATCH --gres=gpu:1                           # Numero de gpus a usar

#SBATCH -w hera

#SBATCH --mem 20GB

export PATH="/opt/anaconda/anaconda3/bin:$PATH"

export PATH="/opt/anaconda/bin:$PATH"

eval "$(conda shell.bash hook)"

conda activate /mnt/homeGPU/mcarmona/mcarmona

export TFHUB_CACHE_DIR=.


#python inference_gptj.py
echo "Hola"
mail -s "Proceso finalizado" e.mcs2000carmona@go.ugr.es <<< "El proceso ha finalizado"