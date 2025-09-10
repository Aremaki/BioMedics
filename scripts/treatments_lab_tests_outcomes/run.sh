cd ../../
source .venv/bin/activate
cd scripts/treatments_lab_tests_outcomes
eds-toolbox spark submit --config ../../configs/end2end/config_study_cortico_v1.cfg --log-path logs/ ../treatments_lab_tests_outcomes/run.py
