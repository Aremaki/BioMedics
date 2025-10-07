#!/bin/bash
source "/export/home/cse200093/Adam/biomedics/.venv/bin/activate"

# Set config file path (override with first arg or existing env var `config`)
config="${1:-${config:-../../configs/end2end/config_patient_similarity.cfg}}"
export config

# Ensure logs directory exists for SLURM and other logs
mkdir -p logs

echo "Using config: $config"


#######################
## NER + Qualif
#######################

# Submit job and extract job ID
JOB_ID=$(sbatch run_ner.sh | awk '{print $NF}')
echo "Submitted job $JOB_ID."

# Log file names based on SLURM options
STDOUT_LOG="logs/slurm-${JOB_ID}-stdout.log"
STDERR_LOG="logs/slurm-${JOB_ID}-stderr.log"

# Wait for log files to appear
echo "Waiting for logs to be created..."
while [ ! -f "$STDOUT_LOG" ] || [ ! -f "$STDERR_LOG" ]; do
    sleep 1
done

# Tail both logs in background
echo "Tailing logs:"
echo "  STDOUT: $STDOUT_LOG"
echo "  STDERR: $STDERR_LOG"
tail -n 20 -f "$STDOUT_LOG" &
TAIL_STDOUT_PID=$!
tail -n 20 -f "$STDERR_LOG" &
TAIL_STDERR_PID=$!

# Trap Ctrl+C to clean up tails early if interrupted
trap "kill $TAIL_STDOUT_PID $TAIL_STDERR_PID; exit" SIGINT

# Wait for job to disappear from the queue
echo "Waiting for job $JOB_ID to complete..."
while squeue -j "$JOB_ID" > /dev/null 2>&1 && squeue -j "$JOB_ID" | grep -q "$JOB_ID"; do
    sleep 5
done

# Kill tail processes
kill $TAIL_STDOUT_PID $TAIL_STDERR_PID
wait $TAIL_STDOUT_PID $TAIL_STDERR_PID 2>/dev/null

echo "Job $JOB_ID finished. Continuing..."



#######################
## Extract Measurement
#######################

eds-toolbox spark submit --config "$config" --log-path logs/ ../extract_measurement/run.py


#######################
## NORMALIZATION
#######################

# Submit job and extract job ID
JOB_ID=$(sbatch run_normalization.sh | awk '{print $NF}')
echo "Submitted job $JOB_ID."

# Log file names based on SLURM options
STDOUT_LOG="logs/slurm-${JOB_ID}-stdout.log"
STDERR_LOG="logs/slurm-${JOB_ID}-stderr.log"

# Wait for log files to appear
echo "Waiting for logs to be created..."
while [ ! -f "$STDOUT_LOG" ] || [ ! -f "$STDERR_LOG" ]; do
    sleep 1
done

# Tail both logs in background
echo "Tailing logs:"
echo "  STDOUT: $STDOUT_LOG"
echo "  STDERR: $STDERR_LOG"
tail -n 20 -f "$STDOUT_LOG" &
TAIL_STDOUT_PID=$!
tail -n 20 -f "$STDERR_LOG" &
TAIL_STDERR_PID=$!

# Trap Ctrl+C to clean up tails early if interrupted
trap "kill $TAIL_STDOUT_PID $TAIL_STDERR_PID; exit" SIGINT

# Wait for job to disappear from the queue
echo "Waiting for job $JOB_ID to complete..."
while squeue -j "$JOB_ID" > /dev/null 2>&1 && squeue -j "$JOB_ID" | grep -q "$JOB_ID"; do
    sleep 5
done

# Kill tail processes
kill $TAIL_STDOUT_PID $TAIL_STDERR_PID
wait $TAIL_STDOUT_PID $TAIL_STDERR_PID 2>/dev/null

echo "Job $JOB_ID finished. Continuing..."
echo "All Good"

#######################
## GROUP ALL IN BRAT
#######################

# Submit job and extract job ID
JOB_ID=$(sbatch run_group_data_in_brat.sh | awk '{print $NF}')
echo "Submitted job $JOB_ID."

# Log file names based on SLURM options
STDOUT_LOG="logs/slurm-${JOB_ID}-stdout.log"
STDERR_LOG="logs/slurm-${JOB_ID}-stderr.log"

# Wait for log files to appear
echo "Waiting for logs to be created..."
while [ ! -f "$STDOUT_LOG" ] || [ ! -f "$STDERR_LOG" ]; do
    sleep 1
done

# Tail both logs in background
echo "Tailing logs:"
echo "  STDOUT: $STDOUT_LOG"
echo "  STDERR: $STDERR_LOG"
tail -n 20 -f "$STDOUT_LOG" &
TAIL_STDOUT_PID=$!
tail -n 20 -f "$STDERR_LOG" &
TAIL_STDERR_PID=$!

# Trap Ctrl+C to clean up tails early if interrupted
trap "kill $TAIL_STDOUT_PID $TAIL_STDERR_PID; exit" SIGINT

# Wait for job to disappear from the queue
echo "Waiting for job $JOB_ID to complete..."
while squeue -j "$JOB_ID" > /dev/null 2>&1 && squeue -j "$JOB_ID" | grep -q "$JOB_ID"; do
    sleep 5
done

# Kill tail processes
kill $TAIL_STDOUT_PID $TAIL_STDERR_PID
wait $TAIL_STDOUT_PID $TAIL_STDERR_PID 2>/dev/null

echo "Job $JOB_ID finished."
