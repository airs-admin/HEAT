#!/usr/bin/env bash
source $HOME/miniconda3/bin/activate
conda activate modular_rl

# Get current script path.
SCRIPT_PATH=$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")

# Echo json file for recovery.
CONFIG_FILE=$SCRIPT_PATH/config.json
cat "$CONFIG_FILE"

# Run training and analyzing script.
bash "$SCRIPT_PATH/eval_type_morphologies.sh" "$CONFIG_FILE"