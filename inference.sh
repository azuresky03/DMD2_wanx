PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

torchrun --nproc_per_node=8 --master-port 29517 main/test/generate.py --task t2v-14B --size 832*480 --sample_steps 20 --ckpt_dir /vepfs-zulution/models/Wan2.1-T2V-14B --dit_fsdp --t5_fsdp --ulysses_size 4 --ring_size 2 --sample_shift 5\
 --sample_guide_scale 5 --prompt "A cat walks on the grass, realistic style." --base_seed 42