#!/bin/bash
#SBATCH --job-name=NER
#SBATCH -t 48:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=40000
#SBATCH --output=logs/slurm-%j-stdout.log
#SBATCH --error=logs/slurm-%j-stderr.log
#SBATCH --container-image /scratch/images/sparkhadoop.sqsh --container-mount-home --container-writable
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPTS_DIR="${SCRIPTS_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
BIOMEDICS_ROOT="${BIOMEDICS_ROOT:-$(cd "$SCRIPTS_DIR/.." && pwd)}"
VENV_PATH="${VENV_PATH:-$BIOMEDICS_ROOT/.venv/bin/activate}"

cd "$SCRIPTS_DIR/ner"
if [ -f "$VENV_PATH" ]; then
	source "$VENV_PATH"
else
	echo "Warning: venv not found at $VENV_PATH"
fi

# Set config file path (override with first arg or existing env var `config`)
config="${1:-${config:-../../configs/end2end/config_end_to_end_public.cfg}}"
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
