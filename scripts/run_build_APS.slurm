#!/bin/bash
#SBATCH --job-name=build-aps
#SBATCH --partition=sfscai
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=logs/build_aps_%j.out
#SBATCH --error=logs/build_aps_%j.err

source ~/.bashrc
conda activate T2S

cd /scratch/gw2556/ODE/CasGen
mkdir -p logs

echo "=== Building APS.pt dataset ==="
python build_APS.py
echo "=== Done ==="
