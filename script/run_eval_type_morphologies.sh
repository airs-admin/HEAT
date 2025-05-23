#!/usr/bin/env bash
source $HOME/miniconda3/bin/activate
conda activate modular_rl

# cheetah, hopper, humanoid, walker.
MORPHLOGIES=cheetah
CUSTOM_XMLS=("xmls_${MORPHLOGIES}_10")
MAX_NUM_EXP=3
MAX_TIMESTAMPS=20000000
PARALLEL_COUNT=3

# Get current script path.
SCRIPT_PATH=$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")

# Create json file as input.
CONFIG_FILE=$SCRIPT_PATH/config.json
cat <<EOF > "$CONFIG_FILE"
{
  "MAX_NUM_EXP": $MAX_NUM_EXP,
  "MAX_TIMESTAMPS": $MAX_TIMESTAMPS,
  "MORPHLOGIES": "$MORPHLOGIES",
  "CUSTOM_XMLS": ["$(printf '%s","' "${CUSTOM_XMLS[@]}" | sed 's/","$//')"],
  "PARALLEL_COUNT": $PARALLEL_COUNT
}
EOF
cat "$CONFIG_FILE"

# Create a new workspace.
CURRENT_TIME=$(date +"%Y-%m-%d_%H_%M_%S")
WORKSPACE=$HOME/train_$CURRENT_TIME
echo "Create a new workspace $WORKSPACE"
mkdir -p $WORKSPACE
cp -r $SCRIPT_PATH/../../heat $WORKSPACE/.
# Update script path.
SCRIPT_PATH=$WORKSPACE/heat/script
# Clean up results directory.
TARGET_DIR=$SCRIPT_PATH/../src/results
TEMP_DIR=$(mktemp -d)
trap 'rm -rf "$TEMP_DIR"' EXIT
rm -rf $TARGET_DIR/*
# Run training and analyzing script.
bash "$SCRIPT_PATH/eval_type_morphologies.sh" "$CONFIG_FILE"