
torchrun qchem_30B.py \
    --model_name_or_path ./cpt_HF_30B/ \
    --data_path "./data/pyinstructions.json" \
    --fp16 True \
    --output_dir ./output4/ \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --deepspeed ds_config_own_stage3.json \
    --tf32 False
