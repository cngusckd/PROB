#!/usr/bin/env bash

# ========================================
#      PROB-DETR Multi-Task Runner
# ========================================

set -e  # Exit immediately on error

# Usage function
show_usage() {
    echo "========================================="
    echo "         PROB-DETR Multi-Task Runner"
    echo "========================================="
    echo "Usage: $0 <mode> <task_number>"
    echo ""
    echo "Mode:"
    echo "  train    - Training mode (save weights)"
    echo "  eval     - Evaluation mode (load weights)"
    echo ""
    echo "Task Number:"
    echo "  1        - Task 1 (PREV_CLS=0, CUR_CLS=3)"
    echo "  2        - Task 2 (PREV_CLS=3, CUR_CLS=2)"
    echo "  3        - Task 3 (PREV_CLS=5, CUR_CLS=1)"
    echo ""
    echo "Examples:"
    echo "  $0 train 1    # Train Task 1"
    echo "  $0 eval 1     # Evaluate Task 1"
    echo "  $0 train 2    # Train Task 2"
    echo "  $0 eval 2     # Evaluate Task 2"
    echo "========================================="
}

# Parameter validation
if [ $# -ne 2 ]; then
    show_usage
    exit 1
fi

MODE=$1
TASK=$2

# Mode validation
if [[ "$MODE" != "train" && "$MODE" != "eval" ]]; then
    echo "❌ ERROR: Mode must be 'train' or 'eval'"
    show_usage
    exit 1
fi

# Task validation
if [[ "$TASK" != "1" && "$TASK" != "2" && "$TASK" != "3" ]]; then
    echo "❌ ERROR: Task must be 1, 2, or 3"
    show_usage
    exit 1
fi

# Base configuration (matching modified_M_OWOD_BENCHMARK.sh)
EXP_DIR="exps/MOWODB/CLAD3"
PROJECT_NAME="CLAD_Logging"
WANDB_NAME="CLAD_HJ"
REPLAY_NAME="learned_clad_hj"
MODEL_TYPE="prob"
BATCH_SIZE=1
EPOCHS=1  # Set back to 21 for proper training

# Memory optimization settings
NUM_WORKERS=0  # Reduce CPU memory usage
EVAL_EVERY=999999  # Disable evaluation during training for memory efficiency

# Task-specific configuration (matching modified_M_OWOD_BENCHMARK.sh)
case $TASK in
    1)
        PREV_CLS=0
        CUR_CLS=3
        TRAIN_SET="clad_t1_train_2025ver"
        OUTPUT_DIR="${EXP_DIR}/t1"
        PRETRAIN_PATH=""
        WANDB_TASK_NAME="${WANDB_NAME}_t1"
        REPLAY_FILE="${REPLAY_NAME}_t1_ft.txt"
        ;;
    2)
        PREV_CLS=3
        CUR_CLS=2
        TRAIN_SET="clad_t2_train_2025ver"
        OUTPUT_DIR="${EXP_DIR}/t2"
        PRETRAIN_PATH="${EXP_DIR}/t1/task1_final.pth"
        WANDB_TASK_NAME="${WANDB_NAME}_t2"
        REPLAY_FILE="${REPLAY_NAME}_t2_ft.txt"
        REPLAY_PREV_FILE="${REPLAY_NAME}_t1_ft.txt"
        ;;
    3)
        PREV_CLS=5
        CUR_CLS=1
        TRAIN_SET="clad_t3_train_2025ver"
        OUTPUT_DIR="${EXP_DIR}/t3"
        PRETRAIN_PATH="${EXP_DIR}/t2/task2_final.pth"
        WANDB_TASK_NAME="${WANDB_NAME}_t3"
        REPLAY_FILE="${REPLAY_NAME}_t3_ft.txt"
        REPLAY_PREV_FILE="${REPLAY_NAME}_t2_ft.txt"
        ;;
esac

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "========================================="
echo "  MODE: $MODE | TASK: $TASK"
echo "  OUTPUT_DIR: $OUTPUT_DIR"
echo "  PREV_CLS: $PREV_CLS | CUR_CLS: $CUR_CLS"
echo "  WANDB_NAME: $WANDB_TASK_NAME"
echo "========================================="

# Execution branching
if [ "$MODE" = "train" ]; then
    echo "🏋️ Training mode execution..."
    
    # Check for previous checkpoint for Task 2,3
    if [[ "$TASK" != "1" && ! -f "$PRETRAIN_PATH" ]]; then
        echo "❌ ERROR: Previous task checkpoint not found!"
        echo "Required file: $PRETRAIN_PATH"
        echo "Please train the previous task first."
        exit 1
    fi
    
    # Prepare training arguments
    TRAIN_ARGS=""
    if [[ "$TASK" != "1" ]]; then
        TRAIN_ARGS="--pretrain $PRETRAIN_PATH"
        # Add previous exemplar replay file for Task 2,3
        if [[ -n "$REPLAY_PREV_FILE" ]]; then
            TRAIN_ARGS="$TRAIN_ARGS --exemplar_replay_prev_file $REPLAY_PREV_FILE"
        fi
    fi
    
    # Execute training
    python -u main_open_world.py \
        --output_dir "$OUTPUT_DIR" \
        --dataset CLAD \
        --PREV_INTRODUCED_CLS $PREV_CLS \
        --CUR_INTRODUCED_CLS $CUR_CLS \
        --train_set "$TRAIN_SET" \
        --test_set "clad_test" \
        --epochs $EPOCHS \
        --model_type "$MODEL_TYPE" \
        --obj_loss_coef 8e-4 \
        --obj_temp 1.3 \
        --batch_size $BATCH_SIZE \
        --num_workers $NUM_WORKERS \
        --wandb_project "$PROJECT_NAME" \
        --wandb_name "$WANDB_TASK_NAME" \
        --exemplar_replay_selection \
        --exemplar_replay_max_length 1000 \
        --exemplar_replay_dir "$WANDB_NAME" \
        --exemplar_replay_cur_file "$REPLAY_FILE" \
        --eval_every $EVAL_EVERY \
        $TRAIN_ARGS
    
    echo "✅ Task $TASK training completed!"
    echo "💾 Task-specific final weights saved: $OUTPUT_DIR/task${TASK}_final.pth"
    
elif [ "$MODE" = "eval" ]; then
    echo "🔍 Evaluation mode execution..."
    
    CHECKPOINT_PATH="$OUTPUT_DIR/task${TASK}_final.pth"
    
    # Check checkpoint existence
    if [ ! -f "$CHECKPOINT_PATH" ]; then
        echo "❌ ERROR: Task $TASK checkpoint not found!"
        echo "Required file: $CHECKPOINT_PATH"
        echo "Please run '$0 train $TASK' first to train the model."
        exit 1
    fi
    
    echo "✅ Checkpoint found: $CHECKPOINT_PATH"
    
    # Execute evaluation
    python -u main_open_world.py \
        --output_dir "${OUTPUT_DIR}_eval" \
        --dataset CLAD \
        --PREV_INTRODUCED_CLS $PREV_CLS \
        --CUR_INTRODUCED_CLS $CUR_CLS \
        --train_set "$TRAIN_SET" \
        --test_set "clad_test" \
        --model_type "$MODEL_TYPE" \
        --obj_loss_coef 8e-4 \
        --obj_temp 1.3 \
        --batch_size $BATCH_SIZE \
        --num_workers $NUM_WORKERS \
        --wandb_project "$PROJECT_NAME" \
        --wandb_name "${WANDB_TASK_NAME}_EVAL" \
        --pretrain "$CHECKPOINT_PATH" \
        --eval
    
    echo "✅ Task $TASK evaluation completed!"
fi

echo "========================================="
echo "           Task Completed!"
echo "========================================="
