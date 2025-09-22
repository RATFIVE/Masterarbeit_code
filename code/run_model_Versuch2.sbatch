#!/bin/bash
#SBATCH --job-name=Versuch2_ML
#SBATCH --output=logs/ml_optuna_%j_Versuch2.out
#SBATCH --partition=base
##SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --mail-user=mbanzhaf@geomar.de
#SBATCH --mail-type=END,FAIL

# Argumente (können per sbatch --export übergeben werden)
: "${N_TRIALS:=1000}"      # Default: 100 Optuna-Trials
: "${timeout:=7200}"     # Timeout in Sekunden (Standard: 1 Stunde)
: "${model:=sarima}"       # Modelltyp (Standard: lgbm) # Auswahl: random_forest, svr, xgboost, lgbm, lr, sarima
source /gxfs_home/geomar/smomw693/Documents/Masterarbeit_code/.venv/bin/activate


python 04_Versuch2_MODEL_TRAINING.py --n_trials "$N_TRIALS" --timeout "$timeout" --model "$model"
