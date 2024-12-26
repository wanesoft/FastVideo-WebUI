export WANDB_BASE_URL="https://api.wandb.ai"
export WANDB_MODE=online

torchrun --nnodes 1 --nproc_per_node 8 \
    fastvideo/train.py \
    --seed 42 \
    --pretrained_model_name_or_path data/hunyuan \
    --dit_model_name_or_path data/hunyuan/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt\
    --model_type "hunyuan" \
    --cache_dir data/.cache \
    --data_json_path data/Image-Vid-Finetune-HunYuan/videos2caption.json \
    --validation_prompt_dir data/Image-Vid-Finetune-HunYuan/validation \
    --gradient_checkpointing \
    --train_batch_size=1 \
    --num_latent_t 24 \
    --sp_size 4 \
    --train_sp_batch_size 1 \
    --dataloader_num_workers 4 \
    --gradient_accumulation_steps=1 \
    --max_train_steps=2000 \
    --learning_rate=5e-6 \
    --mixed_precision=bf16 \
    --checkpointing_steps=200 \
    --validation_steps 100 \
    --validation_sampling_steps 64 \
    --checkpoints_total_limit 3 \
    --allow_tf32 \
    --ema_start_step 0 \
    --cfg 0.0 \
    --ema_decay 0.999 \
    --log_validation \
    --output_dir=data/outputs/HSH-Taylor-Finetune-Hunyuan \
    --tracker_project_name HSH-Taylor-Finetune-Hunyuan \
    --num_frames 93 \
    --validation_guidance_scale "1.0" \
    --group_frame