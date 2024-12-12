#!/bin/bash
#SBATCH -p gpu --gres=gpu:a40:1
#SBATCH --job-name=ND
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8 --mem=250000M
#SBATCH --time=24:00:00

./train.sh