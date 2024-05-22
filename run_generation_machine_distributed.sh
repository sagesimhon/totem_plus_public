#!/bin/bash

# Dry run option (true to enable dry run)
DRY_RUN=false



# Remote machine details (not needed if keys already set in .ssh)
REMOTE_USER="simhon"
PRIVATE_KEY="tmp/key.pm"  # Path to private key file

ENV_PROFILE=/afs/csail.mit.edu/u/s/simhon/.bashrc

# Machines in the array
MACHINES=("visiongpu31" "visiongpu33" "visiongpu34" "visiongpu35" "visiongpu36" "visiongpu37")

MACHINES=("isola-titanrtx-1" "isola-2080ti-1" "isola-2080ti-3" "isola-3080-1" "torralba-titanxp-4" "torralba-titanxp-5" "torralba-titanxp-7")
# MACHINES=("isola-titanrtx-1" "isola-2080ti-1")
# MACHINES=("isola-titanrtx-1")
# "isola-ada6000-1"


# Define the range and other params
MIN_VALUE=-0.9
MAX_VALUE=-0.1
RESOLUTION=4096
EXP_FOLDER=r4096
SWEEP_DENSITY=1

LOG_PATH=/data/vision/billf/implicit_scenes/simhon/logs/${EXP_FOLDER}
mkdir "$LOG_PATH"

# Total range and number of machines
TOTAL_RANGE=$(awk "BEGIN { printf \"%.7f\", $MAX_VALUE - $MIN_VALUE}")
NUM_MACHINES=${#MACHINES[@]}

# Calculate segment size for each machine
SEGMENT_SIZE=$(awk "BEGIN { printf \"%.7f\", $TOTAL_RANGE / $NUM_MACHINES }")

# Run commands in parallel in the background
for ((i=0; i<NUM_MACHINES; i++)); do
    MACHINE="${MACHINES[i]}"
    START_RANGE=$(awk "BEGIN { printf \"%.7f\", $MIN_VALUE + ($i * $SEGMENT_SIZE) }")
    END_RANGE=$(awk "BEGIN { printf \"%.7f\", $START_RANGE + $SEGMENT_SIZE}")

    # SSH_COMMAND="ssh -i $PRIVATE_KEY ${REMOTE_USER}@${MACHINE}"  # not needed if the keys are already set in .ssh

    # Run the command in the background
    CMD="source ${ENV_PROFILE}; cd totem_plus; nohup ./run_command.sh python run_generation.py --res ${RESOLUTION} --exp_folder ${EXP_FOLDER}/${MACHINE} --sweep_density ${SWEEP_DENSITY} --continue_iter true  --p yes  --y_min  ${START_RANGE}  --y_max ${END_RANGE} --action generate &> ${LOG_PATH}/${MACHINE}.out"
    # ssh "$MACHINE" "source ${ENV_PROFILE}; cd totem_plus; export CUDA_VISIBLE_DEVICES=0,1,2,3; nohup ./run_command.sh python run_generation_v2.py --res ${RESOLUTION} --exp_folder ${EXP_FOLDER}_${MACHINE} --n 999 --continue_iter true  --p false  --y_min  ${START_RANGE}  --y_max ${END_RANGE} &> ${MACHINE}.out" &
  if [ "$DRY_RUN" = true ]; then
        echo "Dry run: ssh ${MACHINE} ${CMD}"
        ssh "${MACHINE}" "echo logged in to:; hostname"
  else
      # Run the command in the background
      if [ "$i" -eq 0 ]; then
        set -x    # this prints all commands executed in this script automatically, no need for echo, just do it once
      fi
      # echo "Running command on remote: $MACHINE $CMD"
      ssh "$MACHINE" "$CMD" &
  fi
done
# Wait for all background jobs to finish
wait




