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

from main.wan.modules.lora import  set_lora_state

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
        self.denoising_step_list = self.gen_scheduler.timesteps

        if args.initialie_generator:
            self.feedforward_model = WanModel.from_pretrained(args.model_id)

            if args.generator_lora:
                raise NotImplementedError()
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

        indices = torch.randint(
            0, self.num_denoising_step-1, (noise.shape[0],), device=noise.device, dtype=torch.long
        )

        if self.args.sp_size > 1:
            broadcast(indices)

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
                    self.feedforward_model.requires_grad_(True)
                    generated_noise = self.feedforward_model(x,timesteps,None,self.max_seq_len,batch_context=encoder_hidden_states,guidance=guidance_tensor)[0]
                else:
                    # if self.gradient_checkpointing:
                    #     self.accelerator.unwrap_model(self.feedforward_model).disable_gradient_checkpointing()
                    self.feedforward_model.requires_grad_(False)
                    with torch.no_grad():
                        generated_noise = self.feedforward_model(x,timesteps,None,self.max_seq_len,batch_context=encoder_hidden_states,guidance=guidance_tensor)[0]

            self.gen_scheduler.lower_order_nums = 0
            self.gen_scheduler.model_outputs = [None] * self.gen_scheduler.config.solver_order
            self.gen_scheduler._step_index = int(indices.item())
            self.gen_scheduler.last_sample = None
            generated_video = self.gen_scheduler.step(sample=noisy_video, model_output=generated_noise, timestep=timesteps,return_dict=False)[1]

            # print(f"finished generation of fake video by generator, {torch.cuda.max_memory_reserved()/1024**3:.2f} GB")

            if compute_generator_gradient:
                generator_data_dict = {
                    "video": generated_video,
                    "encoder_hidden_states": encoder_hidden_states,
                    "uncond_embedding": uncond_embedding,
                    "guidance_tensor": guidance_tensor,
                } 

                # avoid any side effects of gradient accumulation
                self.guidance_model.requires_grad_(False)
                loss_dict, log_dict = self.guidance_model(
                    generator_turn=True,
                    guidance_turn=False,
                    generator_data_dict=generator_data_dict
                )
                # self.guidance_model.requires_grad_(True)

                if self.args.diffusion_loss:
                    diffusion_loss = F.mse_loss(generated_noise.unsqueeze(0).float(), noise.float(),reduction="mean")
                    # print(f"diffusion_loss: {diffusion_loss}")
                    loss_dict["diffusion_loss"] = diffusion_loss
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




