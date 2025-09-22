#!/bin/bash
#SBATCH --job-name=Eval
#SBATCH --output=logs/Job_%j_Evaluation.out
#SBATCH --partition=base
##SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --mail-user=mbanzhaf@geomar.de
#SBATCH --mail-type=END,FAIL


source /gxfs_home/geomar/smomw693/Documents/GEOMAR-DeepLearning/.venv/bin/activate


python 04_V3_model_evaluation.py
