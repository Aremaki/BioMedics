cd "/export/home/cse200093/Adam/biomedics/scripts/patient_similarity"
source "/export/home/cse200093/Adam/biomedics/.venv/bin/activate"
conda deactivate

echo -----------------
echo COMPUTE PATIENT SIMILARITY
echo -----------------

python run.py
