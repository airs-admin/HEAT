#!/usr/bin/env bash
CONFIG_FILE=$1
if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "Config file $CONFIG_FILE doesn't exist."
    exit 1
fi

# Parse JSON configuration
MAX_NUM_EXP=$(jq '.MAX_NUM_EXP' "$CONFIG_FILE")
MAX_TIMESTAMPS=$(jq '.MAX_TIMESTAMPS' "$CONFIG_FILE")
MORPHLOGIES=$(jq -r '.MORPHLOGIES' "$CONFIG_FILE")
CUSTOM_XMLS=($(jq -r '.CUSTOM_XMLS[]' "$CONFIG_FILE"))
PARALLEL_COUNT=$(jq '.PARALLEL_COUNT' "$CONFIG_FILE")

# Get the current script path
SCRIPT_PATH=$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")

# Print experiment details
echo "** Running morphologies: $MORPHLOGIES."
echo "** Custom XML folders: ${CUSTOM_XMLS[*]}."
echo "** Training $MAX_NUM_EXP models for each XML setting with max timesteps: $MAX_TIMESTAMPS."
NUM_FOLDERS=${#CUSTOM_XMLS[@]}
TOTAL=$((NUM_FOLDERS * MAX_NUM_EXP))
echo "** Total number of experiments: $TOTAL."

# Function to clean up background processes
cleanup() {
    echo "Cleaning up background processes..."
    for PID in "${PIDS[@]}"; do
        if kill -0 "$PID" 2>/dev/null; then
            pkill -TERM -P "$PID"
            kill -9 -- -"$PID"
            echo "Terminated process group $PID."
        fi
    done
    if [[ "$1" == "exit" ]]; then
        exit 0
    fi
}
trap 'cleanup exit' SIGINT

# Set file descriptor limit
MAX_FILE_LIMIT=$(cat /proc/sys/fs/file-max)
CURRENT_HARD_LIMIT=$(ulimit -Hn)
LIMIT_TO_SET=$((MAX_FILE_LIMIT < CURRENT_HARD_LIMIT ? MAX_FILE_LIMIT : CURRENT_HARD_LIMIT))
ulimit -n $LIMIT_TO_SET

# Training models
cd $SCRIPT_PATH/../src/
ID=1
PIDS=()
for CUSTOM_XML in "${CUSTOM_XMLS[@]}"; do
    RUNNING_COUNT=0
    for ((EXP_ID=1; EXP_ID<=MAX_NUM_EXP; EXP_ID++)); do
        echo "Running $CUSTOM_XML ($EXP_ID/$MAX_NUM_EXP), total ($ID/$TOTAL). Max timesteps: $MAX_TIMESTAMPS."
        setsid python train_context_based.py --expID $ID --td --bu --morphologies $MORPHLOGIES --max_timesteps $MAX_TIMESTAMPS --disable_fold --custom_xml ./environments/$CUSTOM_XML --save_checkpoint True --save_freq 10000 &
        PIDS+=($!)
        ((ID++))
        ((RUNNING_COUNT++))
        if (( RUNNING_COUNT >= PARALLEL_COUNT )); then
            wait || true
            RUNNING_COUNT=0
        fi
    done
    wait || true
done
cleanup

# Visualization
cd $SCRIPT_PATH/../src/
ID=1
for CUSTOM_XML in "${CUSTOM_XMLS[@]}"; do
    for ((EXP_ID=1; EXP_ID<=MAX_NUM_EXP; EXP_ID++)); do
        FORMATTED_ID=$(printf "%04d" "$ID")
        echo "Visualizing $CUSTOM_XML ($EXP_ID/$MAX_NUM_EXP), total ($FORMATTED_ID/$TOTAL)."
        python visualize.py --expID $FORMATTED_ID --td --bu --morphologies "$MORPHLOGIES" --custom_xml "./environments/xmls_${MORPHLOGIES}_10"
        ((ID++))
    done
done

# Extract data
DATA_DIR=$SCRIPT_PATH/../src/results/
OUTPUT_DIR=$SCRIPT_PATH/../src/results/
cd $SCRIPT_PATH/../src/helper/
for ((ID=1; ID<=TOTAL; ID++)); do
    FORMATTED_ID=$(printf "%04d" "$ID")
    echo "Extracting data for EXP_ID=$FORMATTED_ID"
    python tensorboard_extract.py --expID EXP_$FORMATTED_ID --data_dir $DATA_DIR --output_dir $OUTPUT_DIR
done

# Plot results
cd $SCRIPT_PATH/../src/helper/
python eval_type_morphologies.py --xml_ids "${CUSTOM_XMLS[@]}" --num_exp $MAX_NUM_EXP --data_dir $OUTPUT_DIR