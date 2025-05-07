# A single unified model that wraps both the generator and discriminator
from diffusers import UNet2DConditionModel, AutoencoderKL, AutoencoderTiny
# from main.wan.modules.model import WanModel
from main.wan.modules.model_cfg import WanModelCFG as WanModel
from main.fm_solvers_unipc import FlowUniPCMultistepScheduler
from main.utils import get_x0_from_noise, NoOpContext
from main.sdxl.sdxl_text_encoder import SDXLTextEncoder
from main.wan_guidance import WanGuidance
from transformers import CLIPTextModel
from peft import LoraConfig
from torch import nn
import torch 
from fastvideo.utils.communications import broadcast
import torch.nn.functional as F

from main.wan.modules.lora import  set_lora_state, apply_lora_to_wanx

import os
def main_print(content):
    if int(os.environ["LOCAL_RANK"]) <= 0:
        print(content)

import math
def sigmoid(x):
  return 1 / (1 + math.exp(-x))

class WanUniModel(nn.Module):
    def __init__(self, args, device):
        super().__init__()

        self.args = args
        self.guidance_model = WanGuidance(args, device)
        self.num_train_timesteps = self.guidance_model.num_train_timesteps
        self.num_visuals = args.grid_size * args.grid_size
        self.conditioning_timestep = args.conditioning_timestep 
        self.use_fp16 = args.use_fp16 
        self.gradient_checkpointing = args.gradient_checkpointing 
        self.backward_simulation = args.backward_simulation 
        self.device = device

        self.cls_on_clean_image = args.cls_on_clean_image 
        self.denoising = args.denoising
        self.denoising_timestep = args.denoising_timestep 
        self.noise_scheduler = self.guidance_model.scheduler

        self.num_denoising_step = args.num_denoising_step 
        self.gen_scheduler = FlowUniPCMultistepScheduler()
        self.gen_scheduler.set_timesteps(self.num_denoising_step,device=device, shift=args.gen_shift)
        self.denoising_step_list = self.gen_scheduler.timesteps.to(device)

        if args.initialie_generator:
            self.feedforward_model = WanModel.from_pretrained(args.model_id)

            if args.generator_lora:
                self.feedforward_model.requires_grad_(False)
                apply_lora_to_wanx(self.feedforward_model,args.lora_rank, args.lora_scale)
                set_lora_state(self.feedforward_model, enabled=True)
            else:
                self.feedforward_model.requires_grad_(True)

            # if self.gradient_checkpointing:
            #     self.feedforward_model.enable_gradient_checkpointing()
        else:
            raise NotImplementedError()

        self.network_context_manager = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if self.use_fp16 else NoOpContext()

        self.max_seq_len = args.max_seq_len

    @torch.no_grad()
    def prepare_denoising_data(self, latents, noise):

        if self.num_denoising_step==1:
            indices = torch.tensor([0], device=noise.device, dtype=torch.long)
        else:
            indices = torch.randint(
                0, self.num_denoising_step-1, (noise.shape[0],), device=noise.device, dtype=torch.long
            )

        torch.distributed.broadcast(indices, src=0)

        timesteps = self.denoising_step_list.to(noise.device)[indices]
        # print(f"timesteps: {timesteps} self.denoising_step_list: {self.denoising_step_list}")

        noisy_image = self.gen_scheduler.add_noise(
            latents, noise, timesteps
        )

        # set last timestep to pure noise
        pure_noise_mask = (timesteps == (self.num_train_timesteps-1))
        # print(f"pure_noise_mask: {pure_noise_mask} timesteps: {timesteps}")
        noisy_image[pure_noise_mask] = noise[pure_noise_mask]

        return timesteps, noisy_image, indices

    def forward(self, noise, encoder_hidden_states, uncond_embedding, latents,
        compute_generator_gradient=True,
        generator_turn=False,
        guidance_turn=False,
        guidance_cfg=None,
        visual=False,
        guidance_data_dict=None,
        real_train_dict=None,
        step = None,
        y=None,
        clip_feature=None,
    ):
        assert (generator_turn and not guidance_turn) or (guidance_turn and not generator_turn) 

        if generator_turn:
            timesteps, noisy_video, indices = self.prepare_denoising_data(
                latents, noise
            )
            # print(f"guidance_cfg: {guidance_cfg} indices: {indices} timestep: {timesteps}")
            guidance_tensor = torch.tensor([guidance_cfg*1000],
                                            device=noisy_video.device,
                                            dtype=torch.bfloat16)
            x = [noisy_video[i] for i in range(noisy_video.size(0))]
            with self.network_context_manager:
                if compute_generator_gradient:
                    if self.args.generator_lora:
                        set_lora_state(self.feedforward_model,requires_grad=True)
                    else:
                        self.feedforward_model.requires_grad_(True)
                    generated_noise = self.feedforward_model(x,timesteps,None,self.max_seq_len,batch_context=encoder_hidden_states,guidance=guidance_tensor,y=y,clip_fea=clip_feature)[0]
                else:
                    # if self.gradient_checkpointing:
                    #     self.accelerator.unwrap_model(self.feedforward_model).disable_gradient_checkpointing()
                    self.feedforward_model.requires_grad_(False)
                    with torch.no_grad():
                        generated_noise = self.feedforward_model(x,timesteps,None,self.max_seq_len,batch_context=encoder_hidden_states,guidance=guidance_tensor,y=y,clip_fea=clip_feature)[0]

            self.gen_scheduler.lower_order_nums = 0
            self.gen_scheduler.model_outputs = [None] * self.gen_scheduler.config.solver_order
            self.gen_scheduler._step_index = int(indices.item())
            self.gen_scheduler.last_sample = None
            generated_video = self.gen_scheduler.step(sample=noisy_video, model_output=generated_noise, timestep=timesteps,return_dict=False)[1]
            
            input_noisy = noisy_video.detach()           

            # print(f"finished generation of fake video by generator, {torch.cuda.max_memory_reserved()/1024**3:.2f} GB")

            if compute_generator_gradient:

                if self.args.dmd_loss:
                    generator_data_dict = {
                        "video": generated_video,
                        "encoder_hidden_states": encoder_hidden_states,
                        "uncond_embedding": uncond_embedding,
                        "guidance_tensor": guidance_tensor,
                        "y": y,
                        "clip_feature": clip_feature,
                    } 

                    # avoid any side effects of gradient accumulation
                    self.guidance_model.requires_grad_(False)
                    loss_dict, log_dict = self.guidance_model(
                        generator_turn=True,
                        guidance_turn=False,
                        generator_data_dict=generator_data_dict
                    )
                    # self.guidance_model.requires_grad_(True)
                else:
                    loss_dict = {}
                    log_dict = {}

                if self.args.diffusion_loss:
                    mult = (self.denoising_timestep // self.num_denoising_step)
                    indice = int(indices.item()) * mult
                    # print(f"indice: {indice} self.denoising_timestep: {self.denoising_timestep} self.num_denoising_step: {self.num_denoising_step}")

                    scheduler = FlowUniPCMultistepScheduler()
                    scheduler.set_timesteps(self.denoising_timestep,device=self.device, shift=self.args.shift)
                    set_lora_state(self.guidance_model.wan,enabled=False)
                    # main_print(f"indice: {indice}")
                    with self.network_context_manager:
                        for ind in range(indice,self.denoising_timestep if self.args.to_x0 else (indice+mult) ):
                            timesteps = scheduler.timesteps[torch.tensor([ind,],device=self.device,dtype=torch.long)]
                            # main_print(f"timesteps {timesteps}")
                            with torch.no_grad():
                                x = [input_noisy[i] for i in range(input_noisy.size(0))]
                                model_pred = self.guidance_model.wan(x,timesteps,None,self.max_seq_len,batch_context=encoder_hidden_states,guidance=guidance_tensor,y=y,clip_fea=clip_feature)[0]
                                input_noisy = scheduler.step(sample=input_noisy, model_output=model_pred, timestep=timesteps,return_dict=False)[1 if (not self.args.to_x0 and ind==(indice+mult-1)) else 0]
                    if self.args.to_x0:
                        scale = (sigmoid((1.1-(self.denoising_timestep - indice) / self.denoising_timestep) * ( (step/self.args.dfake_gen_update_ratio) if step else 1)) - 0.5) * 2
                        main_print(f"scale {scale} indice {indice}")
                    else:
                        scale = 1
                    diffusion_loss = F.mse_loss(generated_video.float(), input_noisy.float(),reduction="mean") * scale
                    log_dict["regression_video"] = input_noisy

                    loss_dict["diffusion_loss"] = diffusion_loss
                    torch.distributed.barrier()
            else:
                loss_dict = {}
                log_dict = {} 

            log_dict["guidance_data_dict"] = {
                "video": generated_video.detach(),
                "encoder_hidden_states": encoder_hidden_states.detach(),
                "uncond_embedding": uncond_embedding.detach(),
                "guidance_tensor": guidance_tensor.detach(),
                "real_train_dict": real_train_dict,
                "noisy_video":noisy_video.detach(),
                "generated_noise":generated_noise.detach(),
                "y": [yi.detach() for yi in y] if self.args.i2v else None,
                "clip_feature": clip_feature.detach() if self.args.i2v else None, 
            }

            log_dict['denoising_timestep'] = timesteps

        elif guidance_turn:
            assert guidance_data_dict is not None 
            set_lora_state(self.guidance_model,requires_grad=True)
            if self.args.cls_on_clean_image:
                self.guidance_model.cls_pred_branch.requires_grad_(True)
            loss_dict, log_dict = self.guidance_model(
                generator_turn=False,
                guidance_turn=True,
                guidance_data_dict=guidance_data_dict
            )    
        return loss_dict, log_dict




