#!/bin/bash
#SBATCH --job-name=feature_importance
#SBATCH --output=logs/feature_importance_%j.out
#SBATCH --partition=base
##SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --mail-user=mbanzhaf@geomar.de
#SBATCH --mail-type=END,FAIL

# Argumente (können per sbatch --export übergeben werden)
: "${n_trials:=100}"      # Default: 50 Optuna-Trials
: "${ocean_points:=30}"      # Default: 50 Optuna-Trials
source /gxfs_home/geomar/smomw693/Documents/Masterarbeit_code/.venv/bin/activate

python 03_feature_importance.py --n_trials "$n_trials" --ocean_points "$ocean_points"