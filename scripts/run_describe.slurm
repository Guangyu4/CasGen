#!/bin/bash
#SBATCH --job-name=describe-ds
#SBATCH --partition=sfscai
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/describe_%j.out
#SBATCH --error=logs/describe_%j.err

source ~/.bashrc
conda activate T2S

cd /scratch/gw2556/ODE/CasGen
mkdir -p logs

echo "=== Dataset Description ==="
python describe_datasets.py
echo "=== Done ==="
