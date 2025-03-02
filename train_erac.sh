#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL # required to send email notifcations
#SBATCH --mail-user=aa8920@ic.ac.uk # required to send email notifcations - please replace <your_username> with your college login name or email address
export PATH=/vol/bitbucket/aa8920/miniconda3/bin/:$PATH
source activate
source /vol/cuda/10.0.130/setup.sh
TERM=vt100 # or TERM=xterm
/usr/bin/nvidia-smi
uptime
conda activate erac
cd /vol/bitbucket/aa8920/ERAC-VAML/mt/erac/
CUDA_VISIBLE_DEVICES=0 python train_erac.py --cuda --actor_path PATH_TO_ACTOR_FOLDER/20210507-171129/model_best.pt --critic_path PATH_TO_CRITIC_FOLDER/20210430-185527/model_best.pt --use_unsuper_reward --nsample 1 --train_bs 35 --valid_bs 35 --test_bs 35
conda deactivate

