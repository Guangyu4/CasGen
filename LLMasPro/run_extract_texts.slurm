#!/bin/bash
#SBATCH --job-name=extract-texts
#SBATCH --partition=sfscai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/gw2556/ODE/CasGen/logs/extract_texts_%j.out
#SBATCH --error=/scratch/gw2556/ODE/CasGen/logs/extract_texts_%j.err

source ~/.bashrc
conda activate T2S

cd /scratch/gw2556/ODE/CasGen
mkdir -p logs

echo "=== Extract Texts ==="
python LLMasPro/extract_texts.py
echo "=== Done ==="
