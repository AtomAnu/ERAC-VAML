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
python train_actor.py --cuda --work_dir PATH_TO_ACTOR_FOLDER --train_bs 50 --valid_bs 50 --test_bs 50
conda deactivate
