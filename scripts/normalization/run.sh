#!/bin/bash
#SBATCH --job-name=coder_inference
#SBATCH -t 48:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=20000
#SBATCH --partition gpuV100
#SBATCH --output=logs/slurm-%j-stdout.log
#SBATCH --error=logs/slurm-%j-stderr.log
#SBATCH --container-image /scratch/images/sparkhadoop.sqsh  --container-mounts=/export/home/$USER:/export/home/$USER,/data/scratch/$USER:/data/scratch/$USER --container-mount-home --container-writable
source $HOME/.user_conda/miniconda/etc/profile.d/conda.sh # appel de ce script
cd "/export/home/cse200093/Adam/biomedics/scripts/normalization"
source "/export/home/cse200093/Adam/biomedics/.venv/bin/activate"
conda deactivate

echo -----------------
echo NORMALIZE BIO LABELS TOTAL APHP DOCS
echo -----------------

python run_coder_inference.py ../../../../../../../data/scratch/cse200093/word-embedding/coder_all ../../data/annotated_CRH/post_processed/expe_complete_pipe/pred/ner ../../data/annotated_CRH/post_processed/expe_complete_pipe/pred/NER_Norm/pred_bio_coder_all.pkl ../../configs/normalization/bio_config.cfg

echo -----------------
echo NORMALIZE MED LABELS TOTAL APHP DOCS
echo -----------------

python run_fuzzy_inference.py ../../data/drug_knowledge/final_dict.pkl ../../data/annotated_CRH/post_processed/expe_complete_pipe/pred/ner ../../data/annotated_CRH/post_processed/expe_complete_pipe/pred/NER_Norm/pred_med_fuzzy_jw.pkl Chemical_and_drugs True jaro_winkler 0.8

start_time="$(date -u +%s)"

echo -----------------
echo NORMALIZE BIO AND MED
echo -----------------

for disease in "lupus_erythemateux_dissemine" "maladie_de_takayasu" "sclerodermie_systemique" "syndrome_des_anti-phospholipides"
do

    echo -----------------
    echo PROCESS $disease
    echo -----------------

    echo -----------------
    echo NORMALIZE BIO LABELS
    echo -----------------

    python run_coder_inference.py ../../../../../../../data/scratch/cse200093/word-embedding/coder_all ../../data/final_results/$disease/pred_with_measurement.pkl ../../data/final_results/$disease/pred_bio_coder_all.pkl ../../configs/normalization/bio_config.cfg

    echo -----------------
    echo NORMALIZE MED LABELS
    echo -----------------

    python run_fuzzy_inference.py ../../data/drug_knowledge/final_dict.pkl ../../data/CRH/pred/$disease ../../data/final_results/$disease/pred_med_fuzzy_jaro_winkler.pkl Chemical_and_drugs True jaro_winkler 0.8
    end_time="$(date -u +%s)"
    elapsed="$(($end_time-$start_time))"
    echo "Total of $elapsed seconds elapsed for $disease"

done

end_time="$(date -u +%s)"
elapsed="$(($end_time-$start_time))"
echo "Total of $elapsed seconds elapsed for process"

echo --EXTRACTION_FINISHED---

echo ---------------
