from main.wan.modules.model_cfg import WanModelCFG
from main.wan.modules.lora import apply_lora_to_wanx, set_lora_state, LoRALayer
import torch
import argparse
from main.wan_guidance import WanGuidance
import os

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--output_path", type=str, default="/mnt/localssd/test_stable_diffusion_coco")
    parser.add_argument("--log_path", type=str, default="/mnt/localssd/log_stable_diffusion_coco")
    parser.add_argument("--train_iters", type=int, default=1000000)
    parser.add_argument("--log_iters", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resolution", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--initialie_generator", action="store_true")
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--wandb_entity", type=str)
    parser.add_argument("--wandb_project", type=str)
    parser.add_argument("--wandb_iters", type=int, default=100)
    parser.add_argument("--wandb_name", type=str, default=None)
    parser.add_argument("--max_grad_norm", type=float, default=10.0, help="max grad norm for network")
    parser.add_argument("--warmup_step", type=int, default=500, help="warmup step for network")
    parser.add_argument("--min_step_percent", type=float, default=0.02, help="minimum step percent for training")
    parser.add_argument("--max_step_percent", type=float, default=0.98, help="maximum step percent for training")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--use_fp16", action="store_true")
    parser.add_argument("--num_train_timesteps", type=int, default=1000)
    parser.add_argument("--ckpt_only_path", type=str, default=None, help="checkpoint (no optimizer state) only path")
    parser.add_argument("--train_prompt_path", type=str)
    parser.add_argument("--latent_resolution", type=int, default=64)
    parser.add_argument("--real_guidance_scale", type=float, default=6.0)
    parser.add_argument("--fake_guidance_scale", type=float, default=1.0)
    parser.add_argument("--grid_size", type=int, default=2)
    parser.add_argument("--no_save", action="store_true", help="don't save ckpt for debugging only")
    parser.add_argument("--cache_dir", type=str, default="/mnt/localssd/cache")
    parser.add_argument("--log_loss", action="store_true", help="log loss at every iteration")
    parser.add_argument("--num_workers", type=int, default=12)
    parser.add_argument("--latent_channel", type=int, default=4)
    parser.add_argument("--max_checkpoint", type=int, default=150)
    parser.add_argument("--dfake_gen_update_ratio", type=int, default=1)
    parser.add_argument("--generator_lr", type=float)
    parser.add_argument("--guidance_lr", type=float)
    parser.add_argument("--cls_on_clean_image", action="store_true")
    parser.add_argument("--gen_cls_loss", action="store_true")
    parser.add_argument("--gen_cls_loss_weight", type=float, default=0)
    parser.add_argument("--guidance_cls_loss_weight", type=float, default=0)
    parser.add_argument("--sdxl", action="store_true")
    parser.add_argument("--fsdp", action="store_true")
    parser.add_argument("--generator_ckpt_path", type=str)
    parser.add_argument("--conditioning_timestep", type=int, default=999)
    parser.add_argument("--tiny_vae", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="apply gradient checkpointing for dfake and generator. this might be a better option than FSDP")
    parser.add_argument("--dm_loss_weight", type=float, default=1.0)

    parser.add_argument("--denoising", action="store_true", help="train the generator for denoising")
    parser.add_argument("--denoising_timestep", type=int, default=1000)
    parser.add_argument("--num_denoising_step", type=int, default=1)
    parser.add_argument("--denoising_loss_weight", type=float, default=1.0)

    parser.add_argument("--diffusion_gan", action="store_true")
    parser.add_argument("--diffusion_gan_max_timestep", type=int, default=0)
    parser.add_argument("--revision", type=str)

    parser.add_argument("--real_image_path", type=str)
    parser.add_argument("--gan_alone", action="store_true", help="only use the gan loss without dmd")
    parser.add_argument("--backward_simulation", action="store_true")

    parser.add_argument("--generator_lora", action="store_true")
    parser.add_argument("--lora_rank", type=int, default=64)
    parser.add_argument("--lora_alpha", type=float, default=8)
    parser.add_argument("--lora_dropout", type=float, default=0.0)

    parser.add_argument("--lora_scale", type=float, default=1)
    parser.add_argument("--shift", type=float, default=5)
    parser.add_argument("--data_json_path", type=str, required=True)
    parser.add_argument("--num_latent_t", type=int, default=21, help="Number of latent timesteps.")
    parser.add_argument("--null_encoded_path",type=str,default="/vepfs-zulution/zhangpengpeng/cv/video_generation/Wan2.1/data/mixkit/meta/null.pt",help="The path to the null encoded path.")
    parser.add_argument(
        "--use_cpu_offload",
        action="store_true",
        help="Whether to use CPU offload for param & gradient & optimizer states.",
    )
    parser.add_argument("--sp_size", type=int, default=1, help="For sequence parallel")
    parser.add_argument("--cfg", type=int, default=5)
    parser.add_argument("--selective_checkpointing", type=float, default=1.0)
    parser.add_argument("--max_seq_len", type=int, default=32760)
    parser.add_argument("--visual", action="store_true")

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # assert args.gradient_accumulation_steps == 1, "grad accumulation not supported yet"

    # assert not (args.fsdp and args.gradient_checkpointing), "currently, we don't support both options. open an issue for details."

    assert args.wandb_iters % args.dfake_gen_update_ratio == 0, "wandb_iters should be a multiple of dfake_gen_update_ratio"

    return args 

if __name__ == "__main__":
    args = parse_args()

    model = WanGuidance(args, device="cuda")

    ckp_dir = "/vepfs-zulution/zhangpengpeng/cv/video_generation/DMD2/outputs/exp2/time_0331_1234|34_seed42/checkpoint_model_000199/guidance.bin"
    model.load_state_dict(torch.load(ckp_dir, map_location="cpu",weights_only=True), strict=True)
    count = 0
    for name, module in model.wan.named_modules():
        if isinstance(module,LoRALayer):
            print("count",count)
            count += 1
            print("lora_A",module.lora_A)
            print("lora_B",module.lora_B)
            print("product",module.lora_A @ module.lora_B)
            print()
            if count == 5:
                break