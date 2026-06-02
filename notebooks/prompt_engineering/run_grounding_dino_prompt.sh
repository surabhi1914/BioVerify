#!/bin/bash
#BSUB -n 1
#BSUB -q gpu 
#BSUB -gpu "num=1"
#BSUB -W 900
#BSUB -J grounding_dino_prompts
#BSUB -o /share/ftrscape/lmiddha/logs/gd_prompts_stdout.%J
#BSUB -e /share/ftrscape/lmiddha/logs/gd_prompts_stderr.%J
source ~/.bashrc
conda activate /usr/local/usrapps/ftrscape/lmiddha/env_ai
cd /share/ftrscape/lmiddha/dataset
python add_grounding_dino_prompts.py
conda deactivate
