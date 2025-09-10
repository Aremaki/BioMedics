#!/bin/bash
#SBATCH --job-name=group_data_in_brat
#SBATCH -t 48:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=50000
#SBATCH --partition gpuV100
#SBATCH --output=logs/slurm-%j-stdout.log
#SBATCH --error=logs/slurm-%j-stderr.log
#SBATCH --container-image /scratch/images/sparkhadoop.sqsh  --container-mounts=/export/home/$USER:/export/home/$USER,/data/scratch/$USER:/data/scratch/$USER --container-mount-home --container-writable
source $HOME/.user_conda/miniconda/etc/profile.d/conda.sh # appel de ce script
cd "/export/home/cse200093/Adam/biomedics/scripts/group_data_in_brat"
source "/export/home/cse200093/Adam/biomedics/.venv/bin/activate"
conda deactivate

echo -----------------
echo GROUP ALL DATA IN BRAT
echo -----------------

python run.py --config ../../configs/end2end/config_study_cortico_v1.cfg

echo ---------------
