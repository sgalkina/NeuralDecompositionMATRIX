module load cuda/11.8
module load gcc/11.2.0
module load anaconda3/2023.03-py3.10

source ~/.bashrc
conda activate ND

/home/nmb127/.conda/envs/ND/bin/python optimize.py --n_iter=50000 --columns Plot
