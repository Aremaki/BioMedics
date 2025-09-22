#!/bin/bash
#SBATCH --job-name=NER
#SBATCH -t 48:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=40000
#SBATCH --partition gpuV100
#SBATCH --output=logs/slurm-%j-stdout.log
#SBATCH --error=logs/slurm-%j-stderr.log
#SBATCH --container-image /scratch/images/sparkhadoop.sqsh  --container-mounts=/export/home/$USER:/export/home/$USER,/data/scratch/$USER:/data/scratch/$USER --container-mount-home --container-writable
source $HOME/.user_conda/miniconda/etc/profile.d/conda.sh # appel de ce script
cd "/export/home/cse200093/Adam/biomedics/scripts/ner"
source "/export/home/cse200093/Adam/biomedics/.venv/bin/activate"
conda deactivate

# Set config file path (override with first arg or existing env var `config`)
config="${1:-${config:-../../configs/end2end/config_patient_similarity.cfg}}"
export config
echo "Using config: $config"

echo -----------------
echo INFERENCE NER
echo -----------------
start_time="$(date -u +%s)"

python infer.py --config "$config"

end_time="$(date -u +%s)"
elapsed="$(($end_time-$start_time))"
echo -----------------
echo "Total of $elapsed seconds elapsed for NER INFERENCE"
echo -----------------
