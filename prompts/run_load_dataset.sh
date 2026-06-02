#!/bin/bash
#BSUB -n 1
#BSUB -W 720
#BSUB -J download_images
#BSUB -o /share/ftrscape/lmiddha/logs/stdout.%J
#BSUB -e /share/ftrscape/lmiddha/logs/stderr.%J
source ~/.bashrc
conda activate /usr/local/usrapps/ftrscape/lmiddha/env_AiPipeline
cd /share/ftrscape/lmiddha/dataset
python load_dataset.py
conda deactivate
