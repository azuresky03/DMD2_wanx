source ~/.bashrc
source activate
conda activate /vepfs-zulution/zhangpengpeng/conda/envs/fastvideo
cd /vepfs-zulution/zhangpengpeng/cv/video_generation/DMD2

PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

# 修改后的torchrun命令
torchrun --nproc_per_node $MLP_WORKER_GPU --master_addr $MLP_WORKER_0_HOST --node_rank $MLP_ROLE_INDEX --master_port $MLP_WORKER_0_PORT --nnodes $MLP_WORKER_NUM \
    main/train.py \
    --visual \
    --shift 5 \
    --gen_shift 7 \
    --min_step_percent 0.04 \
    --log_iters 100 \
    --lora_rank 128 \
    --lora_scale 0.5 \
    --generator_lr 1e-6 \
    --guidance_lr 2e-5 \
    --warmup_step 10 \
    --dfake_gen_update_ratio 4 \
    --sp_size 8 \
    --max_seq_len 32760 \
    --diffusion_gan_max_timestep 10 \
    --denoising_timestep 50 \
    --num_denoising_step 5 \
    --model_id /vepfs-zulution/zhangpengpeng/cv/video_generation/Wan2.1/data/outputs/exp15_distill_cfg_sequence_parallel/checkpoint-200 \
    --data_json_path /cv/zhangpengpeng/cv/video_generation/Wan2.1/data/processed/mixkit/merge_480/videos2caption.json \
    --log_path "/cv/zhangpengpeng/cv/video_generation/DMD2_wanx/outputs/log/exp6.5" \
    --seed 42 \
    --initialie_generator \
    --fsdp \
    --output_path "/cv/zhangpengpeng/cv/video_generation/DMD2_wanx/outputs/exp6.5" \
    --cache_dir /cv/zhangpengpeng/cv/video_generation/DMD2_wanx/outputs/cache \
    --use_fp16 \
    --diffusion_gan \
    --use_cpu_offload \
    --gradient_checkpointing \
    --dmd_loss \
    --gen_cls_loss \
    --gen_cls_loss_weight 0.1 \
    --cls_on_clean_image \
    --guidance_cls_loss_weight 0.002