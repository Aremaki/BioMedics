cd ../../
source .venv/bin/activate
cd scripts/extract_measurement
eds-toolbox spark submit --config ../../configs/extract_measurement/config.cfg --log-path logs/ run.py
