#!/bin/bash
#SBATCH --job-name=Versuch1_DL
#SBATCH --output=logs/Job_%j_Versuch1_DL.out
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --mail-user=mbanzhaf@geomar.de
#SBATCH --mail-type=END,FAIL

# Argumente (können per sbatch --export übergeben werden)

source /gxfs_home/geomar/smomw693/Documents/Masterarbeit_code/.venv/bin/activate


python 05_Versuch1_DL_OPTUNA.py
