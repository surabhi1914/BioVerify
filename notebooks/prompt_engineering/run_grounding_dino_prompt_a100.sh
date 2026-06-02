#!/bin/bash
#BSUB -n 1
#BSUB -q gpu 
#BSUB -gpu "num=1"
#BSUB -m gpu_a100
#BSUB -W 900
#BSUB -J grounding_dino_prompts_a100
#BSUB -o /share/ftrscape/lmiddha/logs/gd_prompts_stdout_a100.%J
#BSUB -e /share/ftrscape/lmiddha/logs/gd_prompts_stderr_a100.%J
source ~/.bashrc
conda activate /usr/local/usrapps/ftrscape/lmiddha/env_ai
cd /share/ftrscape/lmiddha/dataset
python add_grounding_dino_prompts_a100.py
conda deactivate
