#!/bin/bash
python train_2q_module_cleaned.py \
        --code_distance 3 \
        --logical_circuit_index 4 \
        --if_final_round_syndrome \
        --batch_size 1024 \
        --run_index 0 \
        --model_save_path c3_d4 \
        --train_data_dir cached_qec_data_small/train \
        --val_data_dir cached_qec_data_small/val \
        --train_depth_list 1 2 3 4 5 \
        --val_depth_list 1 2 3 4 5 6 7 8 9 10 \
        --load_model trained_models/c3_d3
