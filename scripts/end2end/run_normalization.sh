#!/bin/bash
#SBATCH --job-name=Normalization
#SBATCH -t 48:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=40000
#SBATCH --partition gpuV100
#SBATCH --output=logs/slurm-%j-stdout.log
#SBATCH --error=logs/slurm-%j-stderr.log
#SBATCH --container-image /scratch/images/sparkhadoop.sqsh  --container-mounts=/export/home/$USER:/export/home/$USER,/data/scratch/$USER:/data/scratch/$USER --container-mount-home --container-writable
source $HOME/.user_conda/miniconda/etc/profile.d/conda.sh # appel de ce script
cd "/export/home/cse200093/Adam/biomedics/scripts/normalization"
source "/export/home/cse200093/Adam/biomedics/.venv/bin/activate"
conda deactivate

# Set config file path using existing env var `config`
config="${1:-${config:-../../configs/end2end/config_patient_similarity.cfg}}"
export config
echo "Using config: $config"


echo -----------------
echo DRUG NORMALIZATION
echo -----------------
start_time="$(date -u +%s)"

python run_fuzzy_inference.py --config "$config"

end_time="$(date -u +%s)"
elapsed="$(($end_time-$start_time))"

echo -----------------
echo "Total of $elapsed seconds elapsed for DRUG NORMALIZATION"
echo -----------------

echo -----------------
echo LAB TEST NORMALIZATION
echo -----------------

start_time="$(date -u +%s)"

python run_embedding_similarity.py --config "$config"

end_time="$(date -u +%s)"
elapsed="$(($end_time-$start_time))"
echo -----------------
echo "Total of $elapsed seconds elapsed for LAB TEST NORMALIZATION"
echo -----------------
