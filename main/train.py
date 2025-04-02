import matplotlib
matplotlib.use('Agg')
from main.utils import prepare_images_for_saving, draw_valued_array, draw_probability_histogram
from accelerate.utils import ProjectConfiguration
from diffusers.optimization import get_scheduler
from main.utils import cycle 
from main.wan_unified_model import WanUniModel
from main.wan.modules.lora import set_lora_state
from fastvideo.dataset.latent_datasets import (LatentDataset, latent_collate_function)
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from accelerate.utils import set_seed
from accelerate import Accelerator
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    FullStateDictConfig,
    StateDictType
)
import argparse 
import shutil 
import wandb 
import torch 
import time 
import os
from time import sleep
import random

import torch.distributed as dist
from fastvideo.utils.parallel_states import initialize_sequence_parallel_state
from main.dataset.hidden_datasets import LatentDataset, latent_collate_function
from main.util.hidden_communication_data_wrapper import sp_parallel_dataloader_wrapper
from fastvideo.utils.fsdp_util import (apply_fsdp_checkpointing, get_dit_fsdp_kwargs)
from fastvideo.utils.communications import broadcast
# from main.wan.modules.model import WanAttentionBlock
from main.wan.modules.model_cfg import WanAttentionBlock
from main.wan.utils.utils import cache_video
from torch.nn.utils import clip_grad_norm_
from datetime import datetime, timezone, timedelta

def main_print(content):
    if int(os.environ["LOCAL_RANK"]) <= 0:
        print(content)

import torch
import torch.distributed as dist

def gather_and_average_loss(loss_dict: dict, key: str) -> torch.Tensor:
    """
    使用 PyTorch 原生分布式收集所有设备上 loss_dict 中指定 key 的值并计算平均
    
    Args:
        loss_dict (dict): 包含损失值的字典，每个设备可能不同
        key (str): 需要收集和平均的键
    
    Returns:
        averaged_value (torch.Tensor): 所有设备上该 key 对应值的平均值
    """
    # 获取当前设备的 value
    value = loss_dict[key]
    
    # 确保所有设备上的 value 是张量
    if not isinstance(value, torch.Tensor):
        value = torch.tensor(value, device=torch.cuda.current_device())
    
    # 收集所有设备的值到列表
    world_size = dist.get_world_size()
    gathered_values = [torch.zeros_like(value) for _ in range(world_size)]
    dist.all_gather(gathered_values, value)
    
    # 拼接并计算平均
    averaged_value = torch.mean(torch.stack(gathered_values))
    
    return averaged_value

class Trainer:
    def __init__(self, args):
        self.args = args

        torch.backends.cuda.matmul.allow_tf32 = True

        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)
        device = torch.cuda.current_device()
        initialize_sequence_parallel_state(args.sp_size)
        self.sp_size = args.sp_size

        set_seed(args.seed + rank)
        # We use different seeds for the noise generation in each process to ensure that the noise is different in a batch.
        noise_random_generator = None

        null_encoded = torch.load(args.null_encoded_path, map_location="cpu",weights_only=True)
        self.null_encoded = null_encoded.to(device)
        self.fsdp = args.fsdp
        self.gradient_accumulation_steps = args.gradient_accumulation_steps

        if rank <= 0:
            utc_plus_8 = timezone(timedelta(hours=8))
            current_time = datetime.now(utc_plus_8).strftime("%m%d_%H%M|%S")
            output_path = os.path.join(args.output_path, f"time_{current_time}")
            os.makedirs(output_path, exist_ok=False)

            self.cache_dir = os.path.join(args.cache_dir, f"time_{current_time}")
            os.makedirs(self.cache_dir, exist_ok=False)

            self.output_path = output_path

            os.makedirs(args.log_path, exist_ok=True)
        self.log_path = args.log_path

        self.model = WanUniModel(args,device)
        main_print("WanUniModel loaded !!!")
        self.max_grad_norm = args.max_grad_norm
        self.denoising = args.denoising
        self.step = 0 
        self.visual = args.visual
        if self.visual:
            from main.wan.modules.vae import WanVAE
            vae = WanVAE(vae_pth="/vepfs-zulution/models/Wan2.1-T2V-14B/Wan2.1_VAE.pth")
            # autocast_type = next(vae.model.parameters()).dtype
            autocast_type = torch.bfloat16
            vae.model = vae.model.to(device).to(autocast_type)
            vae.model.eval()
            vae.model.requires_grad_(False)
            self.vae = vae
            main_print(f"VAE loaded, {torch.cuda.max_memory_allocated() / (1024**3):02f}GB")
        else:
            self.vae = None

        if args.ckpt_only_path:
            fw = torch.load(os.path.join(args.ckpt_only_path, "feedforward.bin"),weights_only=True)
            self.model.feedforward_model.load_state_dict(fw)
            gm = torch.load(os.path.join(args.ckpt_only_path, "guidance.bin"),weights_only=True)
            self.model.guidance_model.load_state_dict(gm)
            # self.model.feedforward_model.train()
            # self.model.guidance_model.train()
            del fw, gm
            set_lora_state(self.model.guidance_model,True)
            main_print(f"load checkpoint from {args.ckpt_only_path}")
            dist.barrier()
        else:
            if args.cls_on_clean_image:
                for param in self.model.guidance_model.cls_pred_branch.parameters():
                    dist.broadcast(param.data, src=0)  # 从 rank 0 广播到其他进程

        dataset = LatentDataset(args.data_json_path)
        sampler = DistributedSampler(dataset, rank=rank, num_replicas=world_size, shuffle=False)
        dataloader = DataLoader(
            dataset,
            sampler=sampler,
            collate_fn=latent_collate_function,
            pin_memory=True,
            batch_size=1,
            num_workers=4,
            drop_last=True,
        )
        self.dataloader = sp_parallel_dataloader_wrapper(dataloader,device,1,args.sp_size,1)

        # use two dataloader 
        # as the generator and guidance model are trained at different paces 
        guide_sampler = DistributedSampler(dataset, rank=rank, num_replicas=world_size, shuffle=True)
        guidance_dataloader = DataLoader(
            dataset,
            sampler=guide_sampler,
            collate_fn=latent_collate_function,
            pin_memory=True,
            batch_size=1,
            num_workers=4,
            drop_last=True,
        )
        self.guidance_dataloader = sp_parallel_dataloader_wrapper(guidance_dataloader,device,1,args.sp_size,1)

        cls_sampler = DistributedSampler(dataset, rank=rank, num_replicas=world_size, shuffle=True)
        cls_dataloader = DataLoader(
            dataset,
            sampler=cls_sampler,
            collate_fn=latent_collate_function,
            pin_memory=True,
            batch_size=1,
            num_workers=4,
            drop_last=True,
        )
        self.real_dataloader = sp_parallel_dataloader_wrapper(cls_dataloader,device,1,args.sp_size,1)


        self.guidance_cls_loss_weight = args.guidance_cls_loss_weight 

        self.cls_on_clean_image = args.cls_on_clean_image 
        self.gen_cls_loss = args.gen_cls_loss 
        self.gen_cls_loss_weight = args.gen_cls_loss_weight 
        self.previous_time = None 

        feedforward_model_fsdp_kwargs, feedforward_model_no_split_modules = get_dit_fsdp_kwargs(
        self.model.feedforward_model,
        "full",
        False,
        args.use_cpu_offload,
        "bf16",
        (WanAttentionBlock,)
        )

        self.model.feedforward_model = FSDP(
        self.model.feedforward_model,
        **feedforward_model_fsdp_kwargs,
        )


        self.model.guidance_model.wan = FSDP(
        self.model.guidance_model.wan,
        use_orig_params=True,
        **feedforward_model_fsdp_kwargs,
        )

        main_print("FSDP loaded !!!")
        dist.barrier()

        if args.gradient_checkpointing:
            apply_fsdp_checkpointing(self.model.feedforward_model, feedforward_model_no_split_modules, args.selective_checkpointing)
            apply_fsdp_checkpointing(self.model.guidance_model.wan, feedforward_model_no_split_modules, args.selective_checkpointing)

        self.optimizer_generator = torch.optim.AdamW(
            [param for param in self.model.feedforward_model.parameters() if param.requires_grad], 
            lr=args.generator_lr, 
            betas=(0.9, 0.999),  # pytorch's default 
            weight_decay=0.01  # pytorch's default 
        )
        self.optimizer_guidance = torch.optim.AdamW(
            [param for param in self.model.guidance_model.parameters() if param.requires_grad], 
            lr=args.guidance_lr, 
            betas=(0.9, 0.999),  # pytorch's default 
            weight_decay=0.01  # pytorch's default 
        )

        # actually this scheduler is not very useful (it warms up from 0 to max_lr in 500 / num_gpu steps), but we keep it here for consistency
        self.scheduler_generator = get_scheduler(
            "constant_with_warmup",
            optimizer=self.optimizer_generator,
            num_warmup_steps=args.warmup_step,
            num_training_steps=args.train_iters 
        )
        self.scheduler_guidance = get_scheduler(
            "constant_with_warmup",
            optimizer=self.optimizer_guidance,
            num_warmup_steps=args.warmup_step,
            num_training_steps=args.train_iters 
        )

        self.train_iters = args.train_iters
        self.batch_size = args.batch_size
        self.resolution = args.resolution 
        self.log_iters = args.log_iters
        self.wandb_iters = args.wandb_iters
        self.latent_resolution = args.latent_resolution
        self.grid_size = args.grid_size
        self.log_loss = args.log_loss
        self.latent_channel = args.latent_channel

        self.no_save = args.no_save
        self.max_checkpoint = args.max_checkpoint

        self.dfake_gen_update_ratio = args.dfake_gen_update_ratio

        if args.checkpoint_path is not None:
            self.load(args.checkpoint_path)

        self.device = device
        self.rank = rank
        self.debug = args.debug

        dist.barrier()


    def fsdp_state_dict(self, model):
        fsdp_fullstate_save_policy = FullStateDictConfig(
            offload_to_cpu=True, rank0_only=True
        )
        with FSDP.state_dict_type(
            model, StateDictType.FULL_STATE_DICT, fsdp_fullstate_save_policy
        ):
            checkpoint = model.state_dict()

        return checkpoint 

    def save(self):
        # NOTE: we save the checkpoints to two places 
        # 1. output_path: save the latest one, this is assumed to be a permanent storage
        # 2. cache_dir: save all checkpoints, this is assumed to be a temporary storage
        # training states 
        # If FSDP is used, we only save the model parameter as I haven't figured out how to save the optimizer state without oom yet, help is appreciated.
        # Otherwise, we use the default accelerate save_state function 
        
        if self.fsdp:
            feedforward_state_dict = self.fsdp_state_dict(self.model.feedforward_model)
            guidance_model_state_dict = self.fsdp_state_dict(self.model.guidance_model)

        if int(os.environ["RANK"]) <= 0:
            output_path = os.path.join(self.output_path, f"checkpoint_model_{self.step:06d}")
            os.makedirs(output_path, exist_ok=True)
            print(f"start saving checkpoint to {output_path}")

            if self.fsdp: 
                torch.save(feedforward_state_dict, os.path.join(output_path, f"feedforward.bin"))
                del feedforward_state_dict
                torch.save(guidance_model_state_dict, os.path.join(output_path, f"guidance.bin"))
                del guidance_model_state_dict
            else:
                self.accelerator.save_state(output_path) 

            # remove previous checkpoints 
            for folder in os.listdir(self.output_path):
                if folder.startswith("checkpoint_model") and folder != f"checkpoint_model_{self.step:06d}":
                    shutil.rmtree(os.path.join(self.output_path, folder))

            # copy checkpoints to cache 
            # overwrite the cache
            if os.path.exists(os.path.join(self.cache_dir, f"checkpoint_model_{self.step:06d}")):
                shutil.rmtree(os.path.join(self.cache_dir, f"checkpoint_model_{self.step:06d}"))

            shutil.copytree(
                os.path.join(self.output_path, f"checkpoint_model_{self.step:06d}"),
                os.path.join(self.cache_dir, f"checkpoint_model_{self.step:06d}")
            )

            checkpoints = sorted(
                [folder for folder in os.listdir(self.cache_dir) if folder.startswith("checkpoint_model")]
            )

            if len(checkpoints) > self.max_checkpoint:
                for folder in checkpoints[:-self.max_checkpoint]:
                    shutil.rmtree(os.path.join(self.cache_dir, folder))
            print("done saving")
        torch.cuda.empty_cache()

    def train_one_step(self):
        self.model.train()

        device = self.device

        # 4 channel for SD-VAE, please adapt for other autoencoders 
        COMPUTE_GENERATOR_GRADIENT = self.step % self.dfake_gen_update_ratio == 0
        main_print(f"STEP: {self.step}, COMPUTE_GENERATOR_GRADIENT: {COMPUTE_GENERATOR_GRADIENT}")

        # if self.rank == 0:
        #     mid = args.cfg
        #     a = random.randint(mid-2,mid+3)
        #     a_tensor = torch.tensor([a], dtype=torch.float32,device=device)
        # else:
        #     a_tensor = torch.tensor([-1], dtype=torch.float32,device=device)

        # # 主进程广播 a_tensor 到所有其他进程
        # dist.broadcast(a_tensor, src=0)
        # assert a_tensor[0].item() >0
        a_tensor = torch.tensor([args.cfg], dtype=torch.float32,device=device)
        guidance_cfg = a_tensor[0].item()
        uncond_embedding = self.null_encoded.to(device)


        if COMPUTE_GENERATOR_GRADIENT:
            latents, encoder_hidden_states, latents_attention_mask, encoder_attention_mask, y, clip_feature = next(self.dataloader)
        else:
            latents, encoder_hidden_states, latents_attention_mask, encoder_attention_mask, y, clip_feature = next(self.guidance_dataloader) 
        
        if self.cls_on_clean_image:
            real_train_dict = dict()
            real_latents, real_encoder_hidden_states, real_latents_attention_mask, real_encoder_attention_mask, real_y, real_clip_feature = next(self.real_dataloader) 
            real_train_dict["latents"] = real_latents.to(self.device)
            real_train_dict["encoder_hidden_states"] = real_encoder_hidden_states.to(self.device)
        else:
            real_train_dict = None

        noise = torch.randn_like(latents, device=device)
        if self.sp_size>1:
            broadcast(noise)
        latents = latents.to(device)
        encoder_hidden_states = encoder_hidden_states.to(device)

        # generate images and optionaly compute the generator gradient
        generator_loss_dict, generator_log_dict = self.model(
            noise, encoder_hidden_states, uncond_embedding, latents,
            compute_generator_gradient=COMPUTE_GENERATOR_GRADIENT,
            generator_turn=True,
            guidance_turn=False,
            guidance_cfg=guidance_cfg,
            real_train_dict=real_train_dict
        )

        # first update the generator if the current step is a multiple of dfake_gen_update_ratio
        generator_loss = 0.0 

        if COMPUTE_GENERATOR_GRADIENT:
            generator_loss += generator_loss_dict["loss_dm"] * self.args.dm_loss_weight

            if self.cls_on_clean_image and self.gen_cls_loss:
                generator_loss += generator_loss_dict["gen_cls_loss"] * self.gen_cls_loss_weight

            if self.args.diffusion_loss:
                generator_loss += generator_loss_dict["diffusion_loss"] * self.args.diffusion_loss_weight

            generator_loss.backward()
            if self.debug: main_print(f"backward generator_loss {torch.cuda.max_memory_reserved()/1024**3:.2f} GB")
            generator_grad_norm = clip_grad_norm_(self.model.feedforward_model.parameters(), self.max_grad_norm)

            self.optimizer_generator.step()
            
            # if we also compute gan loss, the classifier may also receive gradient 
            # zero out guidance model's gradient avoids undesired gradient accumulation
            self.optimizer_generator.zero_grad() 
            self.optimizer_guidance.zero_grad()

        self.scheduler_generator.step()



        # update the guidance model (dfake and classifier)
        guidance_loss_dict, guidance_log_dict = self.model(
            noise, encoder_hidden_states, uncond_embedding, latents, 
            compute_generator_gradient=COMPUTE_GENERATOR_GRADIENT,
            generator_turn=False,
            guidance_turn=True,
            guidance_data_dict=generator_log_dict['guidance_data_dict']
        )

        guidance_loss = 0 

        guidance_loss += guidance_loss_dict["loss_fake_mean"]

        if self.cls_on_clean_image:
            guidance_loss += guidance_loss_dict["guidance_cls_loss"] * self.guidance_cls_loss_weight

        guidance_loss.backward()
        guidance_grad_norm = clip_grad_norm_(self.model.guidance_model.parameters(), self.max_grad_norm)


        self.optimizer_guidance.step()
        self.optimizer_guidance.zero_grad()
        self.optimizer_generator.zero_grad() # zero out the generator's gradient as well
        self.scheduler_guidance.step()
        if self.debug: main_print(f"backward guidance_loss {torch.cuda.max_memory_reserved()/1024**3:.2f} GB")

            # combine the two dictionaries 
        loss_dict = {**generator_loss_dict, **guidance_loss_dict}
        log_dict = {**generator_log_dict, **guidance_log_dict}
        if self.debug: main_print(f"log_dict: {log_dict.keys()} {log_dict['guidance_data_dict'].keys()}")

        loss_str = ""
        for key in ["loss_fake_mean", "guidance_cls_loss", "loss_dm", "gen_cls_loss","diffusion_loss"]:
            if key in loss_dict:
                loss_dict[key] = gather_and_average_loss(loss_dict, key)
                loss_str += f"{key}: {loss_dict[key].item():.4f} "
        main_print(loss_str)

        dist.barrier()
        torch.cuda.empty_cache()

        if self.visual and self.rank == 0 and self.step % 10 == 0:
            debug_save_dir = os.path.join(self.log_path, f"step_{self.step}")
            os.makedirs(debug_save_dir, exist_ok=True)
            log_dict["input"] = latents
            for key in ["noisy_video","generated_noise","video"]:
                log_dict[key] = log_dict["guidance_data_dict"][key].detach()
            for key in log_dict:
                # if key not in [ 'faketrain_latents', 'faketrain_noisy_latents', 'faketrain_x0_pred']:
                #     continue
                if key in ["faketrain_latents","faketrain_x0_pred","real_video_for_cls","input","noisy_video","dmtrain_pred_real_video","dmtrain_pred_fake_video"]:
                    value = log_dict[key]
                    with torch.no_grad():
                        video = self.vae.decode([value.squeeze(0).float()])[0]
                        cache_video(tensor=video[None],save_file=f"{debug_save_dir}/rank{dist.get_rank()}_{key}.mp4",fps=16,nrow=1,normalize=True,value_range=(-1, 1))
            print(f"save video to {debug_save_dir}")
        dist.barrier()
                

    def train(self):
        main_print(f"start training {torch.cuda.max_memory_reserved()/1024**3:.2f} GB")
        for index in range(self.step, self.train_iters):                
            self.train_one_step()
            if (not self.no_save)  and (self.step+1) % self.log_iters == 0:
                self.save()

            dist.barrier()
            if int(os.environ["LOCAL_RANK"]) <= 0:
                current_time = time.time()
                if self.previous_time is None:
                    self.previous_time = current_time
                else:
                    print(f"per iteration time {current_time-self.previous_time:.4f}")
                    self.previous_time = current_time

            self.step += 1

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
    parser.add_argument("--backward_simulation", action="store_true")

    parser.add_argument("--generator_lora", action="store_true")
    parser.add_argument("--lora_rank", type=int, default=64)

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
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--gen_shift", type=float, default=3)
    parser.add_argument("--diffusion_loss", action="store_true")
    parser.add_argument("--diffusion_loss_weight", type=float, default=1)
    parser.add_argument("--uncond_for_fake", action="store_true")

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

    trainer = Trainer(args)

    trainer.train()