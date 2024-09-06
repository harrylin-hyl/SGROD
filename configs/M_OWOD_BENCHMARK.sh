#!/usr/bin/env bash

echo running training of SGROD, M-OWODB dataset

set -x

EXP_DIR=exps/MOWODB/SGROD
PY_ARGS=${@:1}
WANDB_NAME=M_SGROD


python -u main_open_world.py \
    --output_dir "${EXP_DIR}/t1" --dataset TOWOD --PREV_INTRODUCED_CLS 0 --CUR_INTRODUCED_CLS 20\
    --train_set 'owod_t1_train' --test_set 'owod_all_task_test' --epochs 41\
    --model_type 'sgrod' --obj_loss_coef 0.1 --pseudo_obj_loss_coef 0.1 --obj_temp 1.0 --pseudo_obj_temp 1.0\
    --pseudo_bbox_loss_coef 5. --pseudo_giou_loss_coef 0. --geometirc_mean_alpha 0.5 \
    --wandb_name "${WANDB_NAME}_t1" --exemplar_replay_selection --exemplar_replay_max_length 850\
    --exemplar_replay_dir ${WANDB_NAME} --exemplar_replay_cur_file "learned_owod_t1_ft.txt"\
    --iou_thre 0.4 --obj_thre 0.7 --sns_thre 0.3 --lr_decline_p 0.5 --unmatched_boxes\
    ${PY_ARGS}

python -u main_open_world.py \
    --output_dir "${EXP_DIR}/t2" --dataset TOWOD --PREV_INTRODUCED_CLS 20 --CUR_INTRODUCED_CLS 20\
    --train_set 'owod_t2_train' --test_set 'owod_all_task_test' --epochs 51\
    --model_type 'sgrod' --obj_loss_coef 0.1 --pseudo_obj_loss_coef 0.1 --obj_temp 1.0 --pseudo_obj_temp 1.0 --freeze_prob_model\
    --pseudo_bbox_loss_coef 5 --pseudo_giou_loss_coef 0. --geometirc_mean_alpha 0.5 \
    --wandb_name "${WANDB_NAME}_t2"\
    --exemplar_replay_selection --exemplar_replay_max_length 1743 --exemplar_replay_dir ${WANDB_NAME}\
    --exemplar_replay_prev_file "learned_owod_t1_ft.txt" --exemplar_replay_cur_file "learned_owod_t2_ft.txt"\
    --pretrain "${EXP_DIR}/t1/checkpoint0040.pth" --lr 2e-5\
    --iou_thre 0.4 --obj_thre 0.7 --sns_thre 0.3 --lr_decline_p 0.5 --unmatched_boxes\
    ${PY_ARGS}
    

python -u main_open_world.py \
    --output_dir "${EXP_DIR}/t2_ft" --dataset TOWOD --PREV_INTRODUCED_CLS 20 --CUR_INTRODUCED_CLS 20 \
    --train_set "${WANDB_NAME}/learned_owod_t2_ft" --test_set 'owod_all_task_test' --epochs 111 --lr_drop 40\
    --model_type 'sgrod' --obj_loss_coef 0.1 --pseudo_obj_loss_coef 0.1 --obj_temp 1.0 --pseudo_obj_temp 1.0\
    --pseudo_bbox_loss_coef 5 --pseudo_giou_loss_coef 0. --geometirc_mean_alpha 0.5 \
    --wandb_name "${WANDB_NAME}_t2_ft"\
    --pretrain "${EXP_DIR}/t2/checkpoint0050.pth"\
    --iou_thre 0.4 --obj_thre 0.7 --sns_thre 0.3 --lr_decline_p 0.5 --unmatched_boxes\
    ${PY_ARGS}
    
    
python -u main_open_world.py \
    --output_dir "${EXP_DIR}/t3" --dataset TOWOD --PREV_INTRODUCED_CLS 40 --CUR_INTRODUCED_CLS 20\
    --train_set 'owod_t3_train' --test_set 'owod_all_task_test' --epochs 121\
    --model_type 'sgrod' --obj_loss_coef 0.1 --pseudo_obj_loss_coef 0.1 --obj_temp 1.0 --pseudo_obj_temp 1.0 --freeze_prob_model\
    --pseudo_bbox_loss_coef 5 --pseudo_giou_loss_coef 0. --geometirc_mean_alpha 0.5 \
    --wandb_name "${WANDB_NAME}_t3"\
    --exemplar_replay_selection --exemplar_replay_max_length 2361 --exemplar_replay_dir ${WANDB_NAME}\
    --exemplar_replay_prev_file "learned_owod_t2_ft.txt" --exemplar_replay_cur_file "learned_owod_t3_ft.txt"\
    --pretrain "${EXP_DIR}/t2_ft/checkpoint0110.pth" --lr 2e-5 \
    --iou_thre 0.4 --obj_thre 0.7 --sns_thre 0.3 --lr_decline_p 0.5  --unmatched_boxes
    ${PY_ARGS}
    
    
python -u main_open_world.py \
    --output_dir "${EXP_DIR}/t3_ft" --dataset TOWOD --PREV_INTRODUCED_CLS 40 --CUR_INTRODUCED_CLS 20 \
    --train_set "${WANDB_NAME}/learned_owod_t3_ft" --test_set 'owod_all_task_test' --epochs 181 --lr_drop 35\
    --model_type 'sgrod' --obj_loss_coef 0.1 --pseudo_obj_loss_coef 0.1 --obj_temp 1.0 --pseudo_obj_temp 1.0\
    --pseudo_bbox_loss_coef 5 --pseudo_giou_loss_coef 0. --geometirc_mean_alpha 0.5 \
    --wandb_name "${WANDB_NAME}_t3_ft"\
    --pretrain "${EXP_DIR}/t3/checkpoint0120.pth"\
    --iou_thre 0.4 --obj_thre 0.7 --sns_thre 0.3 --lr_decline_p 0.5  --unmatched_boxes
    ${PY_ARGS}
    
    

python -u main_open_world.py \
    --output_dir "${EXP_DIR}/t4" --dataset TOWOD --PREV_INTRODUCED_CLS 60 --CUR_INTRODUCED_CLS 20\
    --train_set 'owod_t4_train' --test_set 'owod_all_task_test' --epochs 191 \
    --model_type 'sgrod' --obj_loss_coef 0.1 --pseudo_obj_loss_coef 0.1 --obj_temp 1.0 --pseudo_obj_temp 1.0 --freeze_prob_model\
    --pseudo_bbox_loss_coef 5 --pseudo_giou_loss_coef 0. --geometirc_mean_alpha 0.5 \
    --wandb_name "${WANDB_NAME}_t4"\
    --exemplar_replay_selection --exemplar_replay_max_length 2749 --exemplar_replay_dir ${WANDB_NAME}\
    --exemplar_replay_prev_file "learned_owod_t3_ft.txt" --exemplar_replay_cur_file "learned_owod_t4_ft.txt"\
    --num_inst_per_class 40\
    --pretrain "${EXP_DIR}/t3_ft/checkpoint0180.pth" --lr 2e-5\
    --iou_thre 0.4 --obj_thre 0.7  --sns_thre 0.3 --lr_decline_p 0.5 --unmatched_boxes \
    ${PY_ARGS}
    
    
python -u main_open_world.py \
    --output_dir "${EXP_DIR}/t4_ft" --dataset TOWOD --PREV_INTRODUCED_CLS 60 --CUR_INTRODUCED_CLS 20\
    --train_set "${WANDB_NAME}/learned_owod_t4_ft" --test_set 'owod_all_task_test' --epochs 261 --lr_drop 50\
    --model_type 'sgrod' --obj_loss_coef 0.1 --pseudo_obj_loss_coef 0.1 --obj_temp 1.0 --pseudo_obj_temp 1.0 \
    --pseudo_bbox_loss_coef 5 --pseudo_giou_loss_coef 0. --geometirc_mean_alpha 0.5 \
    --wandb_name "${WANDB_NAME}_t4_ft"\
    --pretrain "${EXP_DIR}/t4/checkpoint0190.pth" \
    --iou_thre 0.4 --obj_thre 0.7 --sns_thre 0.3 --lr_decline_p 0.5 --unmatched_boxes \
    ${PY_ARGS}

    
