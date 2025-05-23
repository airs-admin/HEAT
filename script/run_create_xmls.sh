#!/usr/bin/env bash
source $HOME/miniconda3/bin/activate
conda activate modular_rl

# Get current script path.
SCRIPT_PATH=$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")

# Check if any xmls_* files or directories exist before moving
DIR_ENV=$SCRIPT_PATH/../src/environments
if ls $DIR_ENV/xmls_* 1> /dev/null 2>&1; then
    CURRENT_TIME=$(date +"%Y-%m-%d_%H_%M_%S")
    DIR_BK_XMLS=$DIR_ENV/backup_xmls_$CURRENT_TIME
    mkdir -p $DIR_BK_XMLS
    mv $DIR_ENV/xmls_* $DIR_BK_XMLS
    echo "Backup completed: $DIR_BK_XMLS"
else
    echo "No xmls_* files found. Skipping backup."
fi

# Function to create xmls
create_xmls() {
    local type=$1
    local num_runs=$2
    local disable_prob=$3
    local range_len=$4
    local script_name=$5
    local folder_suffix=$6

    DIR_XML_TYPE=$DIR_ENV/xmls_${type}_${folder_suffix}
    mkdir -p $DIR_XML_TYPE
    python $script_name --num_runs $num_runs \
                        --disable_prob $disable_prob \
                        --range_len $range_len \
                        --output_dir $DIR_XML_TYPE \
                        --log_file $DIR_XML_TYPE/xmls_${type}.log
}

# Create xmls for different types
cd $SCRIPT_PATH/../src/environments/mutation

# Walker
create_xmls "walker" 1 0.05 0.1 "run_walker.py" "10"
create_xmls "walker" 8 0.05 0.1 "run_walker.py" "100"
create_xmls "walker" 83 0.05 0.1 "run_walker.py" "1000"

# Humanoid
create_xmls "humanoid" 1 0.05 0.1 "run_humanoid.py" "10"
create_xmls "humanoid" 12 0.05 0.1 "run_humanoid.py" "100"
create_xmls "humanoid" 125 0.05 0.1 "run_humanoid.py" "1000"

# Hopper
create_xmls "hopper" 3 0.05 0.1 "run_hopper.py" "10"
create_xmls "hopper" 33 0.05 0.1 "run_hopper.py" "100"
create_xmls "hopper" 333 0.05 0.1 "run_hopper.py" "1000"

# Cheetah
create_xmls "cheetah" 1 0.05 0.1 "run_cheetah.py" "10"
create_xmls "cheetah" 7 0.05 0.1 "run_cheetah.py" "100"
create_xmls "cheetah" 71 0.05 0.1 "run_cheetah.py" "1000"