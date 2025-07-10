#!/bin/bash

# Source shared parameters
source $WORK/ModelMerging/scripts/training_params.sh

# Set data selection parameters from command line arguments
echo "Setting up data selection parameters from command line arguments..."
DATA_SELECTION_STRATEGY=none
DATA_SELECTION_RATIO=1.0
DATA_SELECTION_SLICE=top
SUBSET_FILE_PATH=none

NUM_EPOCHS=${1:-1}
BATCH_SIZE_MULTIPLIER=${2:-1}
PROMPT_MULTIPLIER=${3:-1}
USE_CPPO=${4:-False}
CPPO_MULTIPLIER=${5:-1}
ONLY_POSITIVE_ADV=${6:-False}
LOG_PROB_MULTIPLIER=${7:-1}
DATASET=${8:-$DATASET}
BASE_MODEL=${9:-$BASE_MODEL}
PER_DEVICE_BATCH_SIZE=${10:-$PER_DEVICE_BATCH_SIZE}
NUM_GENERATIONS=${11:-$NUM_GENERATIONS}
RESUME_FROM_CHECKPOINT=${12:-""}

MAXIMIZE_THROUGHPUT=False






