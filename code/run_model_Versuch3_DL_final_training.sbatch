#!/bin/bash
#SBATCH --job-name=Final_Training_Versuch3_DL
#SBATCH --output=logs/Job_%j_Final_Training_Versuch3_DL.out
#SBATCH --partition=gpu
##SBATCH --nodelist=nesh-gpu01,nesh-gpu02,nesh-gpu03,nesh-gpu04,nesh-gpu05,nesh-gpu06
#SBATCH --gres=gpu:1
##SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --mail-user=mbanzhaf@geomar.de
#SBATCH --mail-type=END,FAIL

# Aktivieren der virtuellen Umgebung
source /gxfs_home/geomar/smomw693/Documents/Masterarbeit_code/.venv/bin/activate

# Umgebungsvariable für PyTorch setzen
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Python-Script starten
python 05_Versuch3_final_Training.py
