PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"
    
torchrun --nnodes 1 --nproc_per_node 8 --master-port 29516\
    main/train.py \
    --visual \
    --shift 5 \
    --gradient_accumulation_steps 1 \
    --generator_lr 1e-6 \
    --guidance_lr 5e-6 \
    --warmup_step 10 \
    --dfake_gen_update_ratio 2 \
    --sp_size 8 \
    --max_seq_len 32760 \
    --diffusion_gan_max_timestep 10 \
    --gen_cls_loss \
    --gen_cls_loss_weight 0.1 \
    --denoising_timestep 50 \
    --num_denoising_step 5 \
    --model_id /vepfs-zulution/zhangpengpeng/cv/video_generation/Wan2.1/data/outputs/exp15_distill_cfg_sequence_parallel/checkpoint-200 \
    --data_json_path /cv/zhangpengpeng/cv/video_generation/Wan2.1/data/processed/mixkit/merge_480/videos2caption.json \
    --log_path /vepfs-zulution/zhangpengpeng/cv/video_generation/DMD2/outputs/debug/exp3 \
    --seed 42 \
    --initialie_generator \
    --fsdp \
    --output_path /vepfs-zulution/zhangpengpeng/cv/video_generation/DMD2/outputs/debug1 \
    --cache_dir /vepfs-zulution/zhangpengpeng/cv/video_generation/DMD2/outputs/cache \
    --use_fp16 \
    --diffusion_gan \
    --cls_on_clean_image \
    --use_cpu_offload \
    --gradient_checkpointing \
    # --ckpt_only_path "/vepfs-zulution/zhangpengpeng/cv/video_generation/DMD2/outputs/exp1.5/time_0327_1702|49_seed42/checkpoint_model_000049" \