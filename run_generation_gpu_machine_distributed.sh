#!/bin/bash

# Dry run option (true to enable dry run)
DRY_RUN=false

# Remote machine details (not needed if keys already set in .ssh)
REMOTE_USER="simhon"
PRIVATE_KEY="tmp/key.pm"  # Path to private key file

ENV_PROFILE=/afs/csail.mit.edu/u/s/simhon/.bashrc


declare -A machine_dict
# machine_dict["isola-3080-1"]=2
 machine_dict["isola-v100-1"]=(3,4,5,6,7)
# machine_dict["isola-v100-2"]=8

# machine_dict["torralba-3090-2"]=7
# machine_dict["torralba-3090-3"]=7
#machine_dict["torralba-v100-2"]=8
machine_dict["freeman-v100-1"]=(0,1,2,3,4,5,6,7)
machine_dict["freeman-v100-2"]=(0,1,2,3,4,5,6,7)
# machine_dict["agrawal-v100-1"]=8


# machine_dict["isola-3080-2"]=2       not functional
# machine_dict["isola-2080ti-1"]=4     too slow
# machine_dict["isola-2080ti-2"]=4     not functional
# machine_dict["isola-2080ti-3"]=4     too slow
# machine_dict["isola-2080ti-4"]=4     too slow
# machine_dict["isola-titanrtx-1"]=8   too slow on gpu fast on cpu
# machine_dict["isola-ada6000-1"]=0    no gpu lots of cpus

# machine_dict["torralba-titanxp-1"]=4
# machine_dict["torralba-titanxp-2"]=4
# machine_dict["torralba-titanxp-3"]=4
# machine_dict["torralba-titanxp-4"]=4
# machine_dict["torralba-titanxp-5"]=4
# machine_dict["torralba-titanxp-6"]=4
# machine_dict["torralba-titanxp-7"]=4

#machine_dict["freeman-titanrtx-1"]=(0,1,2,3,4,5,6,7)
#machine_dict["isola-titanrtx-1"]=(0,2,3,4,5,6,7)


# Define the range and other params
MIN_VALUE=-2
MAX_VALUE=2
RESOLUTION=1024
EXP_FOLDER=r1024
SWEEP_DENSITY=1

LOG_PATH=/data/vision/billf/implicit_scenes/simhon/logs/${EXP_FOLDER}
mkdir -p "$LOG_PATH"

# number of gpus (jobs to run)
total_gpus=0
for gpu_list in "${machine_dict[@]}"; do
  total_gpus=$(${#gpu_list[@]} + total_gpus)
done

# Total range and number of machines
TOTAL_RANGE=$(awk "BEGIN { printf \"%.7f\", $MAX_VALUE - $MIN_VALUE}")
# Calculate segment size for each execution
SEGMENT_SIZE=$(awk "BEGIN { printf \"%.7f\", $TOTAL_RANGE / $total_gpus }")

counter=0
for machine in "${!machine_dict[@]}"; do
  gpu_count=${machine_dict["$machine"]}
  echo "Executing on $machine with $gpu_count GPU(s)"

  # Loop through the GPUs for this machine
  for ((gpu=0; gpu<$gpu_count; gpu++)); do
    START_RANGE=$(awk "BEGIN { printf \"%.7f\", $MIN_VALUE + ($counter * $SEGMENT_SIZE) }")
    END_RANGE=$(awk "BEGIN { printf \"%.7f\", $START_RANGE + $SEGMENT_SIZE}")
    CMD="source ${ENV_PROFILE}; export CUDA_VISIBLE_DEVICES=${gpu}; cd totem_plus; nohup ./run_command.sh python run_generation.py --res ${RESOLUTION} --exp_folder ${EXP_FOLDER}/${machine}-gpu${gpu} --sweep_density ${SWEEP_DENSITY} --continue_iter true  --p false  --y_min  ${START_RANGE}  --y_max ${END_RANGE} --action generate &> ${LOG_PATH}/${machine}-gpu${gpu}.out"

    if [ "$DRY_RUN" = true ]; then
        echo "Dry run: ssh ${machine} ${CMD}"
    else
        echo "ssh $machine $CMD"
        # set -x
        ssh "$machine" "$CMD" &
        sleep 3
        # set +x
    fi
    counter=$((counter + 1))
  done
done

# Wait for all background jobs to finish
wait




