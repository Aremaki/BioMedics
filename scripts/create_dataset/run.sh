cd ../../
source .venv/bin/activate
cd scripts/create_dataset
eds-toolbox spark submit --config ../../configs/create_dataset/config.cfg --log-path logs/ run.py
