#!/bin/bash
#SBATCH --job-name=cas-preprocess
#SBATCH --partition=sfscai
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/preprocess_%j.out
#SBATCH --error=logs/preprocess_%j.err

source ~/.bashrc
conda activate T2S

cd /scratch/gw2556/ODE/CasGen
mkdir -p logs

echo "=== Preprocessing cascades.pt ==="
python preprocess.py
echo "=== Done ==="
