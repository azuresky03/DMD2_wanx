from main.utils import get_x0_from_noise, DummyNetwork, NoOpContext
# from main.wan.modules.model import WanModel
from main.wan.modules.model_cfg import WanModelCFG as WanModel
from main.wan.modules.lora import apply_lora_to_wanx, set_lora_state
from main.wan.flow_match import FlowMatchScheduler
from main.fm_solvers_unipc import FlowUniPCMultistepScheduler
import torch.nn.functional as F
import torch.nn as nn
import torch
import types 
from main.wan.modules.cls import Classifier
from fastvideo.utils.communications import broadcast
from torch.utils.checkpoint import checkpoint

from time import sleep
import os
def main_print(content):
    if int(os.environ["LOCAL_RANK"]) <= 0:
        print(content)

class WanGuidance(nn.Module):
    def __init__(self, args, device):
        super().__init__()
        self.args = args 

        self.wan = WanModel.from_pretrained(args.model_id)

        self.wan.requires_grad_(False)
        apply_lora_to_wanx(self.wan,args.lora_rank, args.lora_scale)
        set_lora_state(self.wan, enabled=True)

        # self.real_unet = self.wan
        # self.fake_unet = self.wan


        # self.scheduler = FlowMatchScheduler(shift=args.shift,sigma_min=0.0, extra_one_step=True)
        self.scheduler = FlowUniPCMultistepScheduler()
        self.scheduler.set_timesteps(num_inference_steps=args.denoising_timestep,shift=args.shift,device=device)

        self.num_train_timesteps = args.num_train_timesteps 
        self.min_step = int(args.min_step_percent * args.denoising_timestep)
        self.max_step = int(args.max_step_percent * args.denoising_timestep)
        
        self.real_guidance_scale = args.real_guidance_scale 
        self.fake_guidance_scale = args.fake_guidance_scale

        assert self.fake_guidance_scale == 1, "no guidance for fake"

        self.use_fp16 = args.use_fp16

        self.cls_on_clean_image = args.cls_on_clean_image 
        self.gen_cls_loss = args.gen_cls_loss 

        if self.cls_on_clean_image:
            # self.fake_unet.forward = types.MethodType(
            #     classify_forward, self.fake_unet
            # )

            wan_dim = self.wan.dim  # 使用WanModel的维度(5120)
            self.cls_pred_branch = Classifier(input_dim=wan_dim).to(device).to(torch.bfloat16 if self.use_fp16 else torch.float32)

            self.cls_pred_branch.requires_grad_(True)

        self.gradient_checkpointing = args.gradient_checkpointing 

        self.diffusion_gan = args.diffusion_gan 
        self.diffusion_gan_max_timestep = args.diffusion_gan_max_timestep

        self.network_context_manager = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if self.use_fp16 else NoOpContext()

    def one_step(self, latents, encoder_hidden_states, guidance_tensor, timesteps, index, y=None, clip_feature=None):

        noise = torch.randn_like(latents)
        if self.args.sp_size > 1:
            broadcast(noise)

        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)

        x = [noisy_latents[i] for i in range(noisy_latents.size(0))]
        with self.network_context_manager:
            pred_noise = self.wan(x,timesteps,None,self.args.max_seq_len,batch_context=encoder_hidden_states,guidance=guidance_tensor,y=y,clip_fea=clip_feature)[0]

        self.scheduler.model_outputs = [None] * self.scheduler.config.solver_order
        self.scheduler.lower_order_nums = 0
        self.scheduler._step_index = int(index.item())
        self.scheduler.last_sample = None
        model_pred, model_pred_x0 = self.scheduler.step(sample=noisy_latents, model_output=pred_noise, timestep=timesteps,return_dict=False)


        # self.scheduler.model_outputs = [None] * self.scheduler.config.solver_order
        # self.scheduler.lower_order_nums = 0
        # self.scheduler._step_index = int(index.item())
        # self.scheduler.last_sample = None        
        # _, model_pred_x0_new = self.scheduler.step(sample=noisy_latents, model_output=noise, timestep=timesteps,return_dict=False)

        return model_pred, pred_noise, noisy_latents, noise, model_pred_x0, None

    def compute_distribution_matching_loss(
        self, 
        latents,
        encoder_hidden_states,
        uncond_embedding,
        guidance_tensor,
        y=None,
        clip_feature=None
    ):
        original_latents = latents 
        with torch.no_grad():

            index = torch.randint(
                self.min_step,min(self.max_step, self.num_train_timesteps)-1,
                [1], 
                device=latents.device,
                dtype=torch.long
            )
            if self.args.sp_size > 1:
                broadcast(index)
            timesteps = self.scheduler.timesteps[index]

            uncond_guidance_tensor = torch.tensor([1*1000],
                                            device=latents.device,
                                            dtype=torch.bfloat16)
            set_lora_state(self.wan, enabled=True)
            _, _, _, _, fake_video, _ = self.one_step(latents, encoder_hidden_states, uncond_guidance_tensor if self.args.uncond_for_fake else guidance_tensor, timesteps, index,y,clip_feature)

            set_lora_state(self.wan, enabled=False)
            _, _, _, _, real_video, _ = self.one_step(latents, encoder_hidden_states, guidance_tensor, timesteps, index,y,clip_feature)

            p_real = (latents - real_video)
            p_fake = (latents - fake_video)

            grad = (p_real - p_fake) / torch.abs(p_real).mean(dim=[1, 2, 3], keepdim=True) 
            grad = torch.nan_to_num(grad)

        loss = 0.5 * F.mse_loss(original_latents.float(), (original_latents-grad).detach().float(), reduction="mean")         

        loss_dict = {
            "loss_dm": loss 
        }

        dm_log_dict = {
            "dmtrain_pred_real_video": real_video.detach().float(),
            "dmtrain_pred_fake_video": fake_video.detach().float(),
            "dmtrain_grad": grad.detach().float(),
            "dmtrain_gradient_norm": torch.norm(grad).item()
        }

        return loss_dict, dm_log_dict

    def compute_loss_fake(
        self,
        latents,
        encoder_hidden_states,
        uncond_embedding,
        guidance_tensor,
        y=None,
        clip_feature=None
    ):
        # if self.gradient_checkpointing:
        #     self.fake_unet.enable_gradient_checkpointing()
        latents = latents.detach()

        index = torch.randint(
            0,self.args.denoising_timestep-1,
            [1], 
            device=latents.device,
            dtype=torch.long
        )
        if self.args.sp_size > 1:
            broadcast(index)
        timesteps = self.scheduler.timesteps[index]

        uncond_guidance_tensor = torch.tensor([1*1000],
                                        device=latents.device,
                                        dtype=torch.bfloat16)
        set_lora_state(self.wan, enabled=True)
        fake_dist_predict, fake_noise_pred, noisy_latents, noise, model_pred_x0, model_pred_x0_new = self.one_step(latents, encoder_hidden_states, uncond_guidance_tensor if self.args.uncond_for_fake else guidance_tensor,timesteps, index, y, clip_feature)

        # epsilon prediction loss 
        # loss_fake = torch.mean(
        #     (fake_noise_pred.float() - noise.float())**2
        # )
        loss_fake = torch.mean(
            (latents.float() - model_pred_x0.float())**2
        )

        loss_dict = {
            "loss_fake_mean": loss_fake,
        }

        fake_log_dict = {
            "faketrain_latents": latents.detach().float(),
            "faketrain_noisy_latents": noisy_latents.detach().float(),
            "faketrain_x0_pred": model_pred_x0.detach().float(),
            "model_pred_x0_new": model_pred_x0_new.detach().float() if model_pred_x0_new is not None else None,
        }
        # if self.gradient_checkpointing:
        #     self.fake_unet.disable_gradient_checkpointing()
        return loss_dict, fake_log_dict

    def compute_generator_clean_cls_loss(self, video, encoder_hidden_states, guidance_tensor,y,clip_feature):
        loss_dict = {} 

        logits = self.compute_cls_logits(
            video, encoder_hidden_states, guidance_tensor,y,clip_feature
        )

        loss_dict["gen_cls_loss"] = F.softplus(-logits).mean()
        return loss_dict 

    def generator_forward(
        self,
        video,
        encoder_hidden_states,
        uncond_embedding,
        guidance_tensor,
        y=None,
        clip_feature=None,
    ):
        loss_dict = {}
        log_dict = {}

        # video.requires_grad_(True)
        dm_dict, dm_log_dict = self.compute_distribution_matching_loss(
            video, encoder_hidden_states, uncond_embedding, guidance_tensor,y,clip_feature)

        loss_dict.update(dm_dict)
        log_dict.update(dm_log_dict)


        if self.cls_on_clean_image:
            clean_cls_loss_dict = self.compute_generator_clean_cls_loss(
                video, encoder_hidden_states, guidance_tensor,y,clip_feature
            )
            loss_dict.update(clean_cls_loss_dict)

            if self.args.debug: main_print(f"finished cls loss, {loss_dict}, {torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024}GB")

        return loss_dict, log_dict 

    def compute_cls_logits(self, video, encoder_hidden_states, guidance_tensor, y=None, clip_feature=None):
        # cut = y is not None
        # if cut:
        #     cut_len = video.shape[2] // 3
        #     video = video[:, :, :cut_len, :, :]
        #     y = [y[i][:, :cut_len,:,:] for i in range(len(y))]
        #     seq_len = self.args.max_seq_len//3
        # else:
        #     seq_len = self.args.max_seq_len
        seq_len = self.args.max_seq_len
        with self.network_context_manager:
            if self.diffusion_gan:
                index = torch.randint(
                    0, self.diffusion_gan_max_timestep, [video.shape[0]], device=video.device, dtype=torch.long
                )
                index = index.to(video.device)
                if self.args.sp_size > 1:
                    broadcast(index)
                timesteps = self.scheduler.timesteps[index]
                noise = torch.randn_like(video)
                if self.args.sp_size > 1:
                    broadcast(noise)
                nosiy_video = self.scheduler.add_noise(video, noise, timesteps)
            else:
                timesteps = torch.zeros([video.shape[0]], dtype=torch.long, device=video.device)

            x = [nosiy_video[i] for i in range(nosiy_video.size(0))]

            uncond_guidance_tensor = torch.tensor([1*1000],
                                            device=video.device,
                                            dtype=torch.bfloat16)
            set_lora_state(self.wan, enabled=True)
            rep = self.wan(x,timesteps,None,seq_len,batch_context=encoder_hidden_states,guidance=uncond_guidance_tensor if self.args.uncond_for_fake else guidance_tensor,classify_mode=True,y=y,clip_fea=clip_feature)

            # logits = checkpoint(self.cls_pred_branch, rep, use_reentrant=False).squeeze(dim=1)
            logits = self.cls_pred_branch(rep).squeeze(dim=1)

        return logits

    def compute_guidance_clean_cls_loss(
            self, real_image, fake_image, 
            real_text_embedding, fake_text_embedding, guidance_tensor, y=None, clip_feature=None, real_y=None, real_clip_feature=None
        ):
        pred_realism_on_real = self.compute_cls_logits(
            real_image.detach(), 
            encoder_hidden_states=real_text_embedding,
            guidance_tensor=guidance_tensor, y=real_y, clip_feature=real_clip_feature
        )
        pred_realism_on_fake = self.compute_cls_logits(
            fake_image.detach(), 
            encoder_hidden_states=fake_text_embedding,
            guidance_tensor=guidance_tensor, y=y, clip_feature=clip_feature
        )

        log_dict = {
            "real_video_for_cls": real_image.detach(),
            "pred_realism_on_real": torch.sigmoid(pred_realism_on_real).detach(),
            "pred_realism_on_fake": torch.sigmoid(pred_realism_on_fake).detach()
        }

        classification_loss = F.softplus(pred_realism_on_fake).mean() + F.softplus(-pred_realism_on_real).mean()
        loss_dict = {
            "guidance_cls_loss": classification_loss
        }
        return loss_dict, log_dict 

    def guidance_forward(
        self,
        video,
        encoder_hidden_states,
        uncond_embedding,
        guidance_tensor,
        real_train_dict,
        y=None,
        clip_feature=None,
    ):
        fake_dict, fake_log_dict = self.compute_loss_fake(
            video, encoder_hidden_states, uncond_embedding, guidance_tensor,y,clip_feature)

        loss_dict = fake_dict 
        log_dict = fake_log_dict

        # print(f"finished fake loss, {fake_dict}, {torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024}GB")

        if self.cls_on_clean_image:
            clean_cls_loss_dict, clean_cls_log_dict = self.compute_guidance_clean_cls_loss(
                real_image=real_train_dict['latents'], 
                fake_image=video,
                real_text_embedding=real_train_dict['encoder_hidden_states'],
                fake_text_embedding=encoder_hidden_states, 
                guidance_tensor=guidance_tensor,
                y = y,
                clip_feature=clip_feature,
                real_y=real_train_dict['y'],
                real_clip_feature=real_train_dict['clip_feature'],
            )
            loss_dict.update(clean_cls_loss_dict)
            log_dict.update(clean_cls_log_dict)

        # print(f"finished guidance_forward, {clean_cls_loss_dict}, {torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024}GB")
        return loss_dict, log_dict 

    def forward(
        self,
        generator_turn=False,
        guidance_turn=False,
        generator_data_dict=None, 
        guidance_data_dict=None
    ):    
        if generator_turn:
            loss_dict, log_dict = self.generator_forward(
                video=generator_data_dict["video"],
                encoder_hidden_states=generator_data_dict["encoder_hidden_states"],
                uncond_embedding=generator_data_dict["uncond_embedding"],
                guidance_tensor=generator_data_dict["guidance_tensor"],
                y=generator_data_dict["y"],
                clip_feature=generator_data_dict["clip_feature"],
            )   
        elif guidance_turn:
            loss_dict, log_dict = self.guidance_forward(
                video=guidance_data_dict["video"],
                encoder_hidden_states=guidance_data_dict["encoder_hidden_states"],
                uncond_embedding=guidance_data_dict["uncond_embedding"],
                guidance_tensor=guidance_data_dict["guidance_tensor"],
                real_train_dict=guidance_data_dict["real_train_dict"],
                y=guidance_data_dict["y"],
                clip_feature=guidance_data_dict["clip_feature"],
            ) 
        else:
            raise NotImplementedError

        return loss_dict, log_dict 