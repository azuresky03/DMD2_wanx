PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

accelerate launch --config_file  /vepfs-zulution/zhangpengpeng/cv/video_generation/DMD2/main/sh/acc_debug.yaml \
    main/train_wan.py \
    --model_id /vepfs-zulution/zhangpengpeng/cv/video_generation/Wan2.1/data/outputs/exp10_distill_cfg/checkpoint-200 \
    --data_json_path /vepfs-zulution/zhangpengpeng/cv/video_generation/Wan2.1/data/mixkit/processed_480/videos2caption.json \
    --log_path /vepfs-zulution/zhangpengpeng/cv/video_generation/DMD2/outputs/debug \
    --seed 42 \
    --initialie_generator \
    --generator_lr 1e-6 \
    --guidance_lr 1e-6 \
    --fsdp \
    --output_path /vepfs-zulution/zhangpengpeng/cv/video_generation/DMD2/outputs/debug \