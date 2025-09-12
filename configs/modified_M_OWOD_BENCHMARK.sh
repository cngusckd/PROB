#!/usr/bin/env bash

echo running training of prob-detr, CLAD dataset

set -x

EXP_DIR=exps/MOWODB/CLAD3
PY_ARGS=${@:1}
PROJECT_NAME=CLAD_OnlyTrain
WANDB_NAME=CLAD_OnlyTrain
REPLAY_NAME=learned_clad
MODEL_TYPE=prob
BATCH_SIZE=1

EPOCHS=21

# train task 1
python -u main_open_world.py \
    --output_dir "${EXP_DIR}/t1" --dataset CLAD --PREV_INTRODUCED_CLS 0 --CUR_INTRODUCED_CLS 3\
    --train_set 'clad_t1_train_2025ver' --test_set 'clad_test' --epochs ${EPOCHS}\
    --model_type "${MODEL_TYPE}" --obj_loss_coef 8e-4 --obj_temp 1.3 --batch_size ${BATCH_SIZE}\
    --wandb_project ${PROJECT_NAME} --wandb_name "${WANDB_NAME}_t1" --exemplar_replay_selection --exemplar_replay_max_length 1000\
    --exemplar_replay_dir ${WANDB_NAME} --exemplar_replay_cur_file "${REPLAY_NAME}_t1_ft.txt" \
    ${PY_ARGS}
    

# # train task 2
# # PY_ARGS=${@:1}
# python -u main_open_world.py \
#     --output_dir "${EXP_DIR}/t2" --dataset CLAD --PREV_INTRODUCED_CLS 3 --CUR_INTRODUCED_CLS 2\
#     --train_set 'clad_t2_train_2025ver' --test_set 'clad_test' --epochs $(( EPOCHS + 20 ))\
#     --model_type 'prob' --obj_loss_coef 8e-4 --obj_temp 1.3 --freeze_prob_model \
#     --wandb_project ${PROJECT_NAME} --wandb_name "${WANDB_NAME}_t2" --batch_size ${BATCH_SIZE}\
#     --exemplar_replay_selection --exemplar_replay_max_length 1000 --exemplar_replay_dir ${WANDB_NAME}\
#     --exemplar_replay_prev_file "${REPLAY_NAME}_t1_ft.txt" --exemplar_replay_cur_file "${REPLAY_NAME}_t2_ft.txt"\
#     --pretrain "${EXP_DIR}/t1/checkpoint00$(( EPOCHS - 1 )).pth" --lr 2e-5 --lite_model "${LITE_MODEL}"\
#     ${PY_ARGS}

# # fine-tune with t1 replay
# PY_ARGS=${@:1}
# python -u main_open_world.py \
#     --output_dir "${EXP_DIR}/t2_ft" --dataset CLAD --PREV_INTRODUCED_CLS 3 --CUR_INTRODUCED_CLS 2\
#     --train_set "${WANDB_NAME}/${REPLAY_NAME}_t2_ft" --test_set 'clad_test' --epochs $(( EPOCHS + 40 )) --lr_drop 7\
#     --wandb_project ${PROJECT_NAME} --wandb_name "${WANDB_NAME}_t2_ft"\
#     --model_type 'prob' --batch_size ${BATCH_SIZE} --obj_loss_coef 8e-4 --obj_temp 1.3\
#     --pretrain "${EXP_DIR}/t2/checkpoint00$(( EPOCHS + 19 )).pth" --lite_model "${LITE_MODEL}"\
#     ${PY_ARGS}

# # train task 3
# PY_ARGS=${@:1}
# python -u main_open_world.py \
#     --output_dir "${EXP_DIR}/t3" --dataset CLAD --PREV_INTRODUCED_CLS 5 --CUR_INTRODUCED_CLS 1\
#     --train_set 'clad_t3_train_2025ver' --test_set 'clad_test' --epochs $(( EPOCHS + 40 ))\
#     --model_type 'prob' --obj_loss_coef 8e-4 --obj_temp 1.3 --freeze_prob_model \
#     --wandb_project "${PROJECT_NAME}" --wandb_name "${WANDB_NAME}_t3" --batch_size ${BATCH_SIZE}\
#     --exemplar_replay_selection --exemplar_replay_max_length 1000 --exemplar_replay_dir ${WANDB_NAME}\
#     --exemplar_replay_prev_file "${REPLAY_NAME}_t1_ft.txt" --exemplar_replay_cur_file "${REPLAY_NAME}_t2_ft.txt"\
#     --pretrain "${EXP_DIR}/t2/checkpoint00$((  EPOCHS + 19 )).pth" --lr 2e-5 --lite_model "${LITE_MODEL}"\
#     ${PY_ARGS}

# # fine-tune with t2 replay
# PY_ARGS=${@:1}
# python -u main_open_world.py \
#     --output_dir "${EXP_DIR}/t3_ft" --dataset CLAD --PREV_INTRODUCED_CLS 5 --CUR_INTRODUCED_CLS 1 \
#     --train_set "${WANDB_NAME}/${REPLAY_NAME}_t3_ft" --test_set 'clad_test' --epochs ${EPOCHS} --lr_drop 7\
#     --model_type 'prob' --batch_size ${BATCH_SIZE} --obj_loss_coef 8e-4 --obj_temp 1.3\
#     --wandb_project ${PROJECT_NAME} --wandb_name "${WANDB_NAME}_t3_ft"\
#     --pretrain "${EXP_DIR}/t2/checkpoint00$(( EPOCHS - 1 )).pth" --lite_model "${LITE_MODEL}"\
#     ${PY_ARGS}