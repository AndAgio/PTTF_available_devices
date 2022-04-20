#!/bin/bash
#SBATCH --job-name=testing-devices
#SBATCH --mail-type=ALL
#SBATCH --mail-user=andrea.agiollo@unibo.it
#SBATCH --time=15-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1

source /public/andrea.agiollo/venvs/pttf/bin/activate

python3 main.py
