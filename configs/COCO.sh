#!/usr/bin/env bash

echo running training of prob-detr, CLAD dataset

set -x

EXP_DIR=exps/COCO
PY_ARGS=${@:1}
PROJECT_NAME=COCO
WANDB_NAME=CLAD_SW
REPLAY_NAME=learned_coco
MODEL_TYPE=prob
BATCH_SIZE=2
FREEZE_MODEL=none

EPOCHS=50

# train task 1
python -m memory_profiler main_open_world.py \
    --output_dir "${EXP_DIR}/t1" --dataset TOWOD --PREV_INTRODUCED_CLS 0 --CUR_INTRODUCED_CLS 80 --num_classes 81\
    --train_set 'train_all' --test_set 'owod_all_task_test' --epochs ${EPOCHS} --transformer_weights 'none'\
    --model_type "${MODEL_TYPE}" --obj_loss_coef 8e-4 --obj_temp 1.3 --batch_size ${BATCH_SIZE} --freeze_mode ${FREEZE_MODEL}\
    --wandb_project ${PROJECT_NAME} --wandb_name "${WANDB_NAME}_t1" --exemplar_replay_selection --exemplar_replay_max_length 1000\
    --exemplar_replay_dir ${WANDB_NAME} --exemplar_replay_cur_file "${REPLAY_NAME}_t1_ft.txt" \
    ${PY_ARGS}