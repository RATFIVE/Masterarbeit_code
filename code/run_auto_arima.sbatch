#!/bin/bash
#SBATCH --job-name=auto_arima
#SBATCH --output=logs/Job_%j_auto_arima.out
#SBATCH --partition=base
##SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --mail-user=mbanzhaf@geomar.de
#SBATCH --mail-type=END,FAIL


source /gxfs_home/geomar/smomw693/Documents/Masterarbeit_code/.venv/bin/activate


python auto_arima.py 
