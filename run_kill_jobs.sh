#!/bin/bash

# Dry run option (true to enable dry run)
DRY_RUN=false


declare -A machine_dict
# machine_dict["isola-3080-1"]=2
# machine_dict["isola-v100-1"]=8
# machine_dict["isola-v100-2"]=8

# machine_dict["torralba-3090-2"]=7
# machine_dict["torralba-3090-3"]=7
machine_dict["torralba-v100-2"]=8
machine_dict["freeman-v100-1"]=8
machine_dict["freeman-v100-2"]=8
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

for machine in "${!machine_dict[@]}"; do

  CMD="pkill -u simhon"

  if [ "$DRY_RUN" = true ]; then
      echo "Dry run: ssh ${machine} ${CMD}"
  else
      echo "ssh $machine $CMD"
      # set -x
      ssh "$machine" "$CMD" &
      # set +x
  fi
done

# Wait for all background jobs to finish
wait




