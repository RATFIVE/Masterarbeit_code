#!/bin/bash
#SBATCH --job-name=ML_Versuch3
#SBATCH --output=logs/Job_%j_Versuch3_ML.out
#SBATCH --partition=base
##SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --mail-user=mbanzhaf@geomar.de
#SBATCH --mail-type=END,FAIL

: "${model:=LGBM}" # # z.B. "RandomForest", "SVR", "XGBoost", "LGBM", "Linear"
source /gxfs_home/geomar/smomw693/Documents/GEOMAR-DeepLearning/.venv/bin/activate


python 04_Versuch3_ML_OPTUNA.py --model "$model"
