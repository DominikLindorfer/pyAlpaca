
torchrun --master_port=1234 qchem_7B_5ep.py \
    --model_name_or_path ./cpt_HF/ll7b/ \
    --data_path "./data/pyinstructions.json" \
    --fp16 True \
    --output_dir ./output_5ep/ \
    --num_train_epochs 5 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --deepspeed ds_config_own_stage3.json \
    --tf32 False
