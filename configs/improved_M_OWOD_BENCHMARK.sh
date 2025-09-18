#!/usr/bin/env bash

# ========================================
#    Improved PROB-DETR Multi-Task Runner
#    Based on modified_M_OWOD_BENCHMARK.sh
# ========================================

set -e  # Exit immediately on error

# Usage function
show_usage() {
    echo "========================================="
    echo "    Improved PROB-DETR Multi-Task Runner"
    echo "========================================="
    echo "Usage: $0 <mode> [task_step]"
    echo ""
    echo "Mode:"
    echo "  train    - Training mode (full pipeline or specific step)"
    echo "  eval     - Evaluation mode (full pipeline or specific step)"
    echo ""
    echo "Task Steps (optional):"
    echo "  1        - Task 1 only"
    echo "  2        - Task 2 only"
    echo "  2_ft     - Task 2 Fine-tune only"
    echo "  3        - Task 3 only"
    echo "  3_ft     - Task 3 Fine-tune only"
    echo ""
    echo "Examples:"
    echo "  $0 train           # Full training pipeline (1→2→2_ft→3→3_ft)"
    echo "  $0 eval            # Full evaluation pipeline (all steps)"
    echo "  $0 train 1         # Train Task 1 only"
    echo "  $0 train 2_ft      # Train Task 2 Fine-tune only"
    echo "  $0 eval 3_ft       # Evaluate Task 3 Fine-tune only"
    echo "========================================="
}

# Parameter validation
if [ $# -lt 1 ] || [ $# -gt 2 ]; then
    show_usage
    exit 1
fi

MODE=$1
TASK_STEP=${2:-"all"}  # Default to "all" if not specified

# Mode validation
if [[ "$MODE" != "train" && "$MODE" != "eval" ]]; then
    echo "❌ ERROR: Mode must be 'train' or 'eval'"
    show_usage
    exit 1
fi

# Task step validation
if [[ "$TASK_STEP" != "all" && "$TASK_STEP" != "1" && "$TASK_STEP" != "2" && "$TASK_STEP" != "2_ft" && "$TASK_STEP" != "3" && "$TASK_STEP" != "3_ft" ]]; then
    echo "❌ ERROR: Task step must be 'all', '1', '2', '2_ft', '3', or '3_ft'"
    show_usage
    exit 1
fi

# Base configuration (from modified_M_OWOD_BENCHMARK.sh)
EXP_DIR=exps/MOWODB/CLAD3
PROJECT_NAME=Freeze_HJ
WANDB_NAME=CLAD_HJ
REPLAY_NAME=learned_clad
MODEL_TYPE=prob
BATCH_SIZE=1
FREEZE_MODEL=backbone_transformer
EPOCHS=1

# Memory optimization settings
NUM_WORKERS=0
EVAL_EVERY=999999

echo "========================================="
echo "  MODE: $MODE | TASK_STEP: $TASK_STEP"
echo "  EXP_DIR: $EXP_DIR"
echo "  PROJECT_NAME: $PROJECT_NAME"
echo "  WANDB_NAME: $WANDB_NAME"
echo "========================================="

# Function to execute a single task step
execute_step() {
    local step=$1
    local mode=$2
    
    echo ""
    echo "🔄 Executing Step: $step (Mode: $mode)"
    echo "========================================="
    
    case $step in
        "1")
            # Task 1 Configuration
            PREV_CLS=0
            CUR_CLS=3
            TRAIN_SET="clad_t1_train_2025ver"
            OUTPUT_DIR="${EXP_DIR}/t1"
            WANDB_TASK_NAME="${WANDB_NAME}_t1"
            REPLAY_FILE="${REPLAY_NAME}_t1_ft.txt"
            TRANSFORMER_WEIGHTS="./checkpoints/PROB_transformer.pth"
            
            # Task 1 Command
            TASK_CMD="python -u main_open_world.py \
                --output_dir \"$OUTPUT_DIR\" \
                --dataset CLAD \
                --PREV_INTRODUCED_CLS $PREV_CLS \
                --CUR_INTRODUCED_CLS $CUR_CLS \
                --train_set \"$TRAIN_SET\" \
                --test_set \"clad_test\" \
                --epochs $EPOCHS \
                --model_type \"$MODEL_TYPE\" \
                --obj_loss_coef 8e-4 \
                --obj_temp 1.3 \
                --batch_size $BATCH_SIZE \
                --num_workers $NUM_WORKERS \
                --freeze_mode \"$FREEZE_MODEL\" \
                --wandb_project \"$PROJECT_NAME\" \
                --wandb_name \"$WANDB_TASK_NAME\" \
                --transformer_weights \"$TRANSFORMER_WEIGHTS\""
            
            if [[ "$mode" == "train" ]]; then
                TASK_CMD="$TASK_CMD --exemplar_replay_selection --exemplar_replay_max_length 1000 --exemplar_replay_dir \"$WANDB_NAME\" --exemplar_replay_cur_file \"$REPLAY_FILE\""
            elif [[ "$mode" == "eval" ]]; then
                CHECKPOINT_PATH="$OUTPUT_DIR/task1_final.pth"
                if [ ! -f "$CHECKPOINT_PATH" ]; then
                    CHECKPOINT_PATH="$OUTPUT_DIR/checkpoint00$((EPOCHS - 1)).pth"
                fi
                if [ ! -f "$CHECKPOINT_PATH" ]; then
                    echo "❌ ERROR: Task 1 checkpoint not found: $CHECKPOINT_PATH"
                    echo "Please run training first: $0 train 1"
                    exit 1
                fi
                TASK_CMD="$TASK_CMD --pretrain \"$CHECKPOINT_PATH\" --eval --output_dir \"${OUTPUT_DIR}_eval\" --wandb_name \"${WANDB_TASK_NAME}_EVAL\""
            fi
            ;;
            
        "2")
            # Task 2 Configuration
            PREV_CLS=3
            CUR_CLS=2
            TRAIN_SET="clad_t2_train_2025ver"
            OUTPUT_DIR="${EXP_DIR}/t2"
            WANDB_TASK_NAME="${WANDB_NAME}_t2"
            REPLAY_FILE="${REPLAY_NAME}_t2_ft.txt"
            REPLAY_PREV_FILE="${REPLAY_NAME}_t1_ft.txt"
            PRETRAIN_PATH="${EXP_DIR}/t1/checkpoint00$((EPOCHS - 1)).pth"
            
            # Check dependency
            if [ ! -f "$PRETRAIN_PATH" ]; then
                PRETRAIN_PATH="${EXP_DIR}/t1/task1_final.pth"
            fi
            if [ ! -f "$PRETRAIN_PATH" ]; then
                echo "❌ ERROR: Task 1 checkpoint not found: $PRETRAIN_PATH"
                echo "Please run Task 1 first: $0 train 1"
                exit 1
            fi
            
            # Task 2 Command
            TASK_CMD="python -u main_open_world.py \
                --output_dir \"$OUTPUT_DIR\" \
                --dataset CLAD \
                --PREV_INTRODUCED_CLS $PREV_CLS \
                --CUR_INTRODUCED_CLS $CUR_CLS \
                --train_set \"$TRAIN_SET\" \
                --test_set \"clad_test\" \
                --epochs $((EPOCHS + 20)) \
                --model_type \"$MODEL_TYPE\" \
                --obj_loss_coef 8e-4 \
                --obj_temp 1.3 \
                --batch_size $BATCH_SIZE \
                --num_workers $NUM_WORKERS \
                --freeze_mode \"$FREEZE_MODEL\" \
                --wandb_project \"$PROJECT_NAME\" \
                --wandb_name \"$WANDB_TASK_NAME\" \
                --pretrain \"$PRETRAIN_PATH\" \
                --freeze_prob_model \
                --lr 2e-5"
            
            if [[ "$mode" == "train" ]]; then
                TASK_CMD="$TASK_CMD --exemplar_replay_selection --exemplar_replay_max_length 1000 --exemplar_replay_dir \"$WANDB_NAME\" --exemplar_replay_prev_file \"$REPLAY_PREV_FILE\" --exemplar_replay_cur_file \"$REPLAY_FILE\""
            elif [[ "$mode" == "eval" ]]; then
                CHECKPOINT_PATH="$OUTPUT_DIR/task2_final.pth"
                if [ ! -f "$CHECKPOINT_PATH" ]; then
                    CHECKPOINT_PATH="$OUTPUT_DIR/checkpoint00$((EPOCHS + 19)).pth"
                fi
                if [ ! -f "$CHECKPOINT_PATH" ]; then
                    echo "❌ ERROR: Task 2 checkpoint not found: $CHECKPOINT_PATH"
                    echo "Please run training first: $0 train 2"
                    exit 1
                fi
                TASK_CMD="python -u main_open_world.py \
                    --output_dir \"${OUTPUT_DIR}_eval\" \
                    --dataset CLAD \
                    --PREV_INTRODUCED_CLS $PREV_CLS \
                    --CUR_INTRODUCED_CLS $CUR_CLS \
                    --train_set \"$TRAIN_SET\" \
                    --test_set \"clad_test\" \
                    --model_type \"$MODEL_TYPE\" \
                    --obj_loss_coef 8e-4 \
                    --obj_temp 1.3 \
                    --batch_size $BATCH_SIZE \
                    --num_workers $NUM_WORKERS \
                    --freeze_mode \"$FREEZE_MODEL\" \
                    --wandb_project \"$PROJECT_NAME\" \
                    --wandb_name \"${WANDB_TASK_NAME}_EVAL\" \
                    --pretrain \"$CHECKPOINT_PATH\" \
                    --eval"
            fi
            ;;
            
        "2_ft")
            # Task 2 Fine-tune Configuration
            PREV_CLS=3
            CUR_CLS=2
            TRAIN_SET="${WANDB_NAME}/${REPLAY_NAME}_t2_ft"
            OUTPUT_DIR="${EXP_DIR}/t2_ft"
            WANDB_TASK_NAME="${WANDB_NAME}_t2_ft"
            PRETRAIN_PATH="${EXP_DIR}/t2/checkpoint00$((EPOCHS + 19)).pth"
            
            # Check dependency
            if [ ! -f "$PRETRAIN_PATH" ]; then
                PRETRAIN_PATH="${EXP_DIR}/t2/task2_final.pth"
            fi
            if [ ! -f "$PRETRAIN_PATH" ]; then
                echo "❌ ERROR: Task 2 checkpoint not found: $PRETRAIN_PATH"
                echo "Please run Task 2 first: $0 train 2"
                exit 1
            fi
            
            # Task 2 Fine-tune Command
            TASK_CMD="python -u main_open_world.py \
                --output_dir \"$OUTPUT_DIR\" \
                --dataset CLAD \
                --PREV_INTRODUCED_CLS $PREV_CLS \
                --CUR_INTRODUCED_CLS $CUR_CLS \
                --train_set \"$TRAIN_SET\" \
                --test_set \"clad_test\" \
                --epochs $((EPOCHS + 40)) \
                --lr_drop 7 \
                --model_type \"$MODEL_TYPE\" \
                --obj_loss_coef 8e-4 \
                --obj_temp 1.3 \
                --batch_size $BATCH_SIZE \
                --num_workers $NUM_WORKERS \
                --freeze_mode \"$FREEZE_MODEL\" \
                --wandb_project \"$PROJECT_NAME\" \
                --wandb_name \"$WANDB_TASK_NAME\" \
                --pretrain \"$PRETRAIN_PATH\""
            
            if [[ "$mode" == "eval" ]]; then
                CHECKPOINT_PATH="$OUTPUT_DIR/task2_ft_final.pth"
                if [ ! -f "$CHECKPOINT_PATH" ]; then
                    CHECKPOINT_PATH="$OUTPUT_DIR/checkpoint00$((EPOCHS + 39)).pth"
                fi
                if [ ! -f "$CHECKPOINT_PATH" ]; then
                    echo "❌ ERROR: Task 2 Fine-tune checkpoint not found: $CHECKPOINT_PATH"
                    echo "Please run training first: $0 train 2_ft"
                    exit 1
                fi
                TASK_CMD="python -u main_open_world.py \
                    --output_dir \"${OUTPUT_DIR}_eval\" \
                    --dataset CLAD \
                    --PREV_INTRODUCED_CLS $PREV_CLS \
                    --CUR_INTRODUCED_CLS $CUR_CLS \
                    --train_set \"$TRAIN_SET\" \
                    --test_set \"clad_test\" \
                    --model_type \"$MODEL_TYPE\" \
                    --obj_loss_coef 8e-4 \
                    --obj_temp 1.3 \
                    --batch_size $BATCH_SIZE \
                    --num_workers $NUM_WORKERS \
                    --freeze_mode \"$FREEZE_MODEL\" \
                    --wandb_project \"$PROJECT_NAME\" \
                    --wandb_name \"${WANDB_TASK_NAME}_EVAL\" \
                    --pretrain \"$CHECKPOINT_PATH\" \
                    --eval"
            fi
            ;;
            
        "3")
            # Task 3 Configuration
            PREV_CLS=5
            CUR_CLS=1
            TRAIN_SET="clad_t3_train_2025ver"
            OUTPUT_DIR="${EXP_DIR}/t3"
            WANDB_TASK_NAME="${WANDB_NAME}_t3"
            REPLAY_FILE="${REPLAY_NAME}_t3_ft.txt"
            REPLAY_PREV_FILE="${REPLAY_NAME}_t2_ft.txt"
            PRETRAIN_PATH="${EXP_DIR}/t2/checkpoint00$((EPOCHS + 19)).pth"  # Task 2 결과 사용
            
            # Check dependency
            if [ ! -f "$PRETRAIN_PATH" ]; then
                PRETRAIN_PATH="${EXP_DIR}/t2/task2_final.pth"
            fi
            if [ ! -f "$PRETRAIN_PATH" ]; then
                echo "❌ ERROR: Task 2 checkpoint not found: $PRETRAIN_PATH"
                echo "Please run Task 2 first: $0 train 2"
                exit 1
            fi
            
            # Task 3 Command
            TASK_CMD="python -u main_open_world.py \
                --output_dir \"$OUTPUT_DIR\" \
                --dataset CLAD \
                --PREV_INTRODUCED_CLS $PREV_CLS \
                --CUR_INTRODUCED_CLS $CUR_CLS \
                --train_set \"$TRAIN_SET\" \
                --test_set \"clad_test\" \
                --epochs $((EPOCHS + 40)) \
                --model_type \"$MODEL_TYPE\" \
                --obj_loss_coef 8e-4 \
                --obj_temp 1.3 \
                --batch_size $BATCH_SIZE \
                --num_workers $NUM_WORKERS \
                --freeze_mode \"$FREEZE_MODEL\" \
                --wandb_project \"$PROJECT_NAME\" \
                --wandb_name \"$WANDB_TASK_NAME\" \
                --pretrain \"$PRETRAIN_PATH\" \
                --freeze_prob_model \
                --lr 2e-5"
            
            if [[ "$mode" == "train" ]]; then
                TASK_CMD="$TASK_CMD --exemplar_replay_selection --exemplar_replay_max_length 1000 --exemplar_replay_dir \"$WANDB_NAME\" --exemplar_replay_prev_file \"$REPLAY_PREV_FILE\" --exemplar_replay_cur_file \"$REPLAY_FILE\""
            elif [[ "$mode" == "eval" ]]; then
                CHECKPOINT_PATH="$OUTPUT_DIR/task3_final.pth"
                if [ ! -f "$CHECKPOINT_PATH" ]; then
                    CHECKPOINT_PATH="$OUTPUT_DIR/checkpoint00$((EPOCHS + 39)).pth"
                fi
                if [ ! -f "$CHECKPOINT_PATH" ]; then
                    echo "❌ ERROR: Task 3 checkpoint not found: $CHECKPOINT_PATH"
                    echo "Please run training first: $0 train 3"
                    exit 1
                fi
                TASK_CMD="python -u main_open_world.py \
                    --output_dir \"${OUTPUT_DIR}_eval\" \
                    --dataset CLAD \
                    --PREV_INTRODUCED_CLS $PREV_CLS \
                    --CUR_INTRODUCED_CLS $CUR_CLS \
                    --train_set \"$TRAIN_SET\" \
                    --test_set \"clad_test\" \
                    --model_type \"$MODEL_TYPE\" \
                    --obj_loss_coef 8e-4 \
                    --obj_temp 1.3 \
                    --batch_size $BATCH_SIZE \
                    --num_workers $NUM_WORKERS \
                    --freeze_mode \"$FREEZE_MODEL\" \
                    --wandb_project \"$PROJECT_NAME\" \
                    --wandb_name \"${WANDB_TASK_NAME}_EVAL\" \
                    --pretrain \"$CHECKPOINT_PATH\" \
                    --eval"
            fi
            ;;
            
        "3_ft")
            # Task 3 Fine-tune Configuration
            PREV_CLS=5
            CUR_CLS=1
            TRAIN_SET="${WANDB_NAME}/${REPLAY_NAME}_t3_ft"
            OUTPUT_DIR="${EXP_DIR}/t3_ft"
            WANDB_TASK_NAME="${WANDB_NAME}_t3_ft"
            PRETRAIN_PATH="${EXP_DIR}/t2/checkpoint00$((EPOCHS - 1)).pth"  # Task 2 결과 사용 (주목!)
            
            # Check dependency
            if [ ! -f "$PRETRAIN_PATH" ]; then
                PRETRAIN_PATH="${EXP_DIR}/t2/task2_final.pth"
            fi
            if [ ! -f "$PRETRAIN_PATH" ]; then
                echo "❌ ERROR: Task 2 checkpoint not found: $PRETRAIN_PATH"
                echo "Please run Task 2 first: $0 train 2"
                exit 1
            fi
            
            # Task 3 Fine-tune Command
            TASK_CMD="python -u main_open_world.py \
                --output_dir \"$OUTPUT_DIR\" \
                --dataset CLAD \
                --PREV_INTRODUCED_CLS $PREV_CLS \
                --CUR_INTRODUCED_CLS $CUR_CLS \
                --train_set \"$TRAIN_SET\" \
                --test_set \"clad_test\" \
                --epochs $EPOCHS \
                --lr_drop 7 \
                --model_type \"$MODEL_TYPE\" \
                --obj_loss_coef 8e-4 \
                --obj_temp 1.3 \
                --batch_size $BATCH_SIZE \
                --num_workers $NUM_WORKERS \
                --freeze_mode \"$FREEZE_MODEL\" \
                --wandb_project \"$PROJECT_NAME\" \
                --wandb_name \"$WANDB_TASK_NAME\" \
                --pretrain \"$PRETRAIN_PATH\""
            
            if [[ "$mode" == "eval" ]]; then
                CHECKPOINT_PATH="$OUTPUT_DIR/task3_ft_final.pth"
                if [ ! -f "$CHECKPOINT_PATH" ]; then
                    CHECKPOINT_PATH="$OUTPUT_DIR/checkpoint00$((EPOCHS - 1)).pth"
                fi
                if [ ! -f "$CHECKPOINT_PATH" ]; then
                    echo "❌ ERROR: Task 3 Fine-tune checkpoint not found: $CHECKPOINT_PATH"
                    echo "Please run training first: $0 train 3_ft"
                    exit 1
                fi
                TASK_CMD="python -u main_open_world.py \
                    --output_dir \"${OUTPUT_DIR}_eval\" \
                    --dataset CLAD \
                    --PREV_INTRODUCED_CLS $PREV_CLS \
                    --CUR_INTRODUCED_CLS $CUR_CLS \
                    --train_set \"$TRAIN_SET\" \
                    --test_set \"clad_test\" \
                    --model_type \"$MODEL_TYPE\" \
                    --obj_loss_coef 8e-4 \
                    --obj_temp 1.3 \
                    --batch_size $BATCH_SIZE \
                    --num_workers $NUM_WORKERS \
                    --freeze_mode \"$FREEZE_MODEL\" \
                    --wandb_project \"$PROJECT_NAME\" \
                    --wandb_name \"${WANDB_TASK_NAME}_EVAL\" \
                    --pretrain \"$CHECKPOINT_PATH\" \
                    --eval"
            fi
            ;;
    esac
    
    # Execute the command
    echo "🚀 Executing: $TASK_CMD"
    eval $TASK_CMD
    
    echo "✅ Step $step ($mode) completed!"
}

# Main execution logic
if [[ "$TASK_STEP" == "all" ]]; then
    # Execute full pipeline
    if [[ "$MODE" == "train" ]]; then
        echo "🏋️ Starting FULL TRAINING PIPELINE..."
        echo "📋 Pipeline: Task 1 → Task 2 → Task 2_ft → Task 3 → Task 3_ft"
        
        execute_step "1" "train"
        execute_step "2" "train"
        execute_step "2_ft" "train"
        execute_step "3" "train"
        execute_step "3_ft" "train"
        
        echo ""
        echo "🎉 FULL TRAINING PIPELINE COMPLETED!"
        echo "📁 All checkpoints saved in: $EXP_DIR"
        
    elif [[ "$MODE" == "eval" ]]; then
        echo "🔍 Starting FULL EVALUATION PIPELINE..."
        echo "📋 Pipeline: Task 1 → Task 2 → Task 2_ft → Task 3 → Task 3_ft"
        
        execute_step "1" "eval"
        execute_step "2" "eval"
        execute_step "2_ft" "eval"
        execute_step "3" "eval"
        execute_step "3_ft" "eval"
        
        echo ""
        echo "🎉 FULL EVALUATION PIPELINE COMPLETED!"
        echo "📁 All evaluation results saved in: ${EXP_DIR}/*_eval"
    fi
else
    # Execute single step
    execute_step "$TASK_STEP" "$MODE"
fi

echo ""
echo "========================================="
echo "           Task Completed!"
echo "========================================="
