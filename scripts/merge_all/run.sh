cd ../../
source .venv/bin/activate
cd scripts/merge_all
eds-toolbox spark submit --config ../../configs/merge_all/config.cfg --log-path logs/ run.py
