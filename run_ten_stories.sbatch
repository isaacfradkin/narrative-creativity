#!/bin/bash
#SBATCH --job-name=nt_ten_stories
#SBATCH --partition=salmon
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/nt_ten_stories_%j.out

cd /sci/labs/arieljaffe/dan.abergel1/narrative-creativity

# CUDA runtime libs installed via pip (libcudart, libcublas); libcuda.so.1
# comes from the GPU node's driver.
export LD_LIBRARY_PATH=$(ls -d $PWD/.venv/lib/python3.11/site-packages/nvidia/*/lib | tr '\n' ':')$LD_LIBRARY_PATH

.venv/bin/python run_texts.py \
    --input "data/ten stories with clear NTs.xlsx" \
    --output ten_stories_NT.xlsx \
    --backend llamacpp
