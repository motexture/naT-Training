import argparse
import datetime
import inspect
import os
import json
import random
import cv2
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import subprocess
import wandb
import time
import gc
import itertools
import sys
import threading

from omegaconf import OmegaConf
from typing import Dict
from omegaconf import OmegaConf
from tqdm.auto import tqdm
from utils.train import TrainDataset, DynamicDatasetDownloader
from utils.validation import ValidationDataset
from pipeline.naT import naTPipeline
from accelerate import Accelerator
from models.unet import naT

global global_step
global_step = None

step_lock = threading.Lock()

def spinner(stop_event, status_message_func):
    spinner_symbols = itertools.cycle(['|', '/', '-', '\\'])
    while not stop_event.is_set():
        with step_lock:
            message = status_message_func()
        if 'step: 0' in message:
            message = message.replace("Global step: 0", "Global step: ")
        sys.stdout.write(f"\r{next(spinner_symbols)} {message}")
        sys.stdout.flush()
        time.sleep(0.1)

def get_status_message():
    return f"Global step: {global_step}"

def custom_collate_fn(batch):
    for b in batch:
        if b is None:
            return None
    return torch.utils.data.default_collate(batch)

def match_histogram(source, target):
    mean_src = source.mean(dim=(2, 3, 4), keepdim=True)
    std_src = source.std(dim=(2, 3, 4), keepdim=True)
    
    mean_tgt = target.mean(dim=(2, 3, 4), keepdim=True)
    std_tgt = target.std(dim=(2, 3, 4), keepdim=True)
    
    adjusted_target = (target - mean_tgt) * (std_src / (std_tgt + 1e-5)) + mean_src
    return adjusted_target

def normalize_latents(latents):
    mean = latents.mean()
    std = latents.std()
    normalized_latents = (latents - mean) / (std + 1e-8)
    return normalized_latents

def denormalize(normalized_tensor):
    if normalized_tensor.is_cuda:
        normalized_tensor = normalized_tensor.cpu()
    
    if normalized_tensor.dim() == 5:
        normalized_tensor = normalized_tensor.squeeze(0)
        
    denormalized = (normalized_tensor + 1.0) * 127.5
    denormalized = torch.clamp(denormalized, 0, 255)
    
    uint8_tensor = denormalized.to(torch.uint8)
    uint8_numpy = uint8_tensor.permute(1, 2, 3, 0).numpy()
    
    return uint8_numpy
    
def save_video(normalized_tensor, output_path, fps=30):
    denormalized_frames = denormalize(normalized_tensor)
    height, width = denormalized_frames.shape[1:3]

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for frame in denormalized_frames:
        bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(bgr_frame)
    
    out.release()

def sample_noise(latents):
    return torch.randn_like(latents, device=latents.device, dtype=latents.dtype)

def encode(tensor, vae):
    batch, channel, frames, height, width = tensor.shape

    tensor = tensor.permute(0, 2, 1, 3, 4)
    tensor = tensor.reshape(batch * frames, channel, height, width)
    
    latents = vae.encode(tensor).latent_dist.sample()

    latents = latents.reshape(batch, frames, latents.shape[1], latents.shape[2], latents.shape[3])
    latents = latents.permute(0, 2, 1, 3, 4)

    latents = vae.config.scaling_factor * latents

    return latents

def decode(latents, vae):
    latents = 1 / vae.config.scaling_factor * latents

    batch, channels, num_frames, height, width = latents.shape
    latents = latents.permute(0, 2, 1, 3, 4).reshape(batch * num_frames, channels, height, width)

    image = vae.decode(latents).sample
    video = (
        image[None, :]
        .reshape(
            (
                batch,
                num_frames,
                -1,
            )
            + image.shape[2:]
        )
        .permute(0, 2, 1, 3, 4)
    )

    video = video.float()
    return video

def freeze_models(models_to_freeze):
    for model in models_to_freeze:
        if model is not None: model.requires_grad_(False) 

def should_sample(global_step, validation_steps, validation_data):
    return (global_step % validation_steps == 0 or global_step == 1) and validation_data.sample_preview
    
def unfreeze_params(model, is_enabled=True, param="temp"):
    for name, module in model.named_modules():
        if param in name:
            for m in module.parameters():
                m.requires_grad_(is_enabled)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def encode_video(input_file, output_file, height):
    command = ['ffmpeg',
               '-i', input_file,
               '-c:v', 'libx264',
               '-crf', '23',
               '-preset', 'fast',
               '-c:a', 'aac',
               '-b:a', '128k',
               '-movflags', '+faststart',
               '-vf', f'scale=-1:{height}',
               '-y',
               output_file]
    
    subprocess.run(command, check=True)

def get_video_height(input_file):
    command = ['ffprobe', 
               '-v', 'quiet', 
               '-print_format', 'json', 
               '-show_streams', 
               input_file]

    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    video_info = json.loads(result.stdout)
    
    for stream in video_info.get('streams', []):
        if stream['codec_type'] == 'video':
            return stream['height']

    return None

def create_output_folders(output_dir, config):
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    out_dir = os.path.join(output_dir, f"train_{now}")
    
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(f"{out_dir}/samples", exist_ok=True)

    return out_dir

def wait_for_videos(dataset, path, min_videos=1, check_interval=0.1):
    while len(dataset.video_files) < min_videos:
        time.sleep(check_interval)
        dataset.find_videos(path)

def main(
    pretrained_model_path: str,
    output_dir: str,
    train_data: Dict,
    sample_data: Dict,
    validation_data: Dict,
    sample_steps: int,
    validation_steps: int,
    gradient_checkpointing: bool,
    **kwargs
):
    global global_step
    global_step = 0
    accelerator = Accelerator()

    *_, config = inspect.getargvalues(inspect.currentframe())

    os.environ["WANDB_API_KEY"] = "c7f1943ea0523f7946a439037f045521b8ff105e"

    wandb.init(project="naT")

    if accelerator.is_main_process:
        output_dir = create_output_folders(output_dir, config)

    if train_data.init_training:
        pipeline = naTPipeline.from_pretrained(pretrained_model_name_or_path=pretrained_model_path, unet=naT.from_pretrained_2d(pretrained_model_path, "unet"))
    else:
        pipeline = naTPipeline.from_pretrained(pretrained_model_name_or_path=pretrained_model_path)
    
    freeze_models([pipeline.text_encoder, pipeline.text_encoder_2, pipeline.unet, pipeline.vae])

    csv_path = "results_10M_train.csv"
    train_dir = "train"
    val_dir = "validate"

    downloader = DynamicDatasetDownloader(
        train_dir=train_dir,
        csv_path=csv_path,
        max_videos=train_data.max_videos
    )

    downloader.start_downloader_thread()

    dataset = TrainDataset(**train_data, path=train_dir, tokenizer=pipeline.tokenizer, text_encoder=pipeline.text_encoder, tokenizer_2=pipeline.tokenizer_2, text_encoder_2=pipeline.text_encoder_2, unet=pipeline.unet)
    wait_for_videos(dataset, train_dir, min_videos=train_data.min_videos)
    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=train_data.batch_size_per_gpu, shuffle=True, collate_fn=custom_collate_fn)
    validation_dataloader = torch.utils.data.DataLoader(
        ValidationDataset(**validation_data, path=val_dir, tokenizer=pipeline.tokenizer, text_encoder=pipeline.text_encoder, tokenizer_2=pipeline.tokenizer_2, text_encoder_2=pipeline.text_encoder_2, unet=pipeline.unet),
        batch_size=validation_data.batch_size_per_gpu,
        shuffle=False,
        collate_fn=custom_collate_fn
    )
    unet, optimizer, train_dataloader, validation_dataloader = accelerator.prepare(
        pipeline.unet,
        torch.optim.Adam(pipeline.unet.parameters(), lr=train_data.learning_rate),
        train_dataloader,
        validation_dataloader
    )
    pipeline.text_encoder = pipeline.text_encoder.to(accelerator.device)
    pipeline.text_encoder_2 = pipeline.text_encoder_2.to(accelerator.device)
    pipeline.vae = pipeline.vae.to(accelerator.device)

    pipeline.vae.enable_slicing()
    unet._set_gradient_checkpointing(value=gradient_checkpointing)
    unet.train()

    stop_event = threading.Event()

    def get_status_message():
        return f"Global step: {global_step}"

    spinner_thread = threading.Thread(target=spinner, args=(stop_event, get_status_message))
    spinner_thread.start()

    def finetune_unet(batch, device):
        pixel_values = batch["pixel_values"].to(device)
        encoder_hidden_states = batch['prompt_embeds'].to(device)
        text_embeds = batch['add_text_embeds'].to(device)
        time_ids = batch['add_time_ids'].to(device)

        with torch.no_grad():
            latents = encode(pixel_values, pipeline.vae)

        timesteps = torch.randint(0, pipeline.scheduler.config.num_train_timesteps, (latents.shape[0],), device=latents.device)
        timesteps = timesteps.long()

        noise = sample_noise(latents)

        noisy_latents = pipeline.scheduler.add_noise(latents, noise, timesteps)

        noisy_latents.requires_grad_(True)
        noise_pred = unet(noisy_latents, timesteps.detach(), encoder_hidden_states.detach(), text_embeds.detach(), time_ids.detach())

        mse_loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
        
        return mse_loss

    def validate(batch, pipeline, unet, device):
        pixel_values = batch["pixel_values"].to(device)
        encoder_hidden_states = batch['prompt_embeds'].to(device)
        text_embeds = batch['add_text_embeds'].to(device)
        time_ids = batch['add_time_ids'].to(device)

        latents = encode(pixel_values, pipeline.vae).to(device)

        timesteps = torch.randint(0, pipeline.scheduler.config.num_train_timesteps, (latents.shape[0],), device=latents.device)
        timesteps = timesteps.long()

        noise = sample_noise(latents).to(device)

        noisy_latents = pipeline.scheduler.add_noise(latents, noise, timesteps)
        noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states, text_embeds, time_ids)

        mse_loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
        
        return mse_loss

    def perform_validation(accelerator, pipeline, unet, validation_dataloader, device):
        val_loss = 0
        valid_count = 0

        val_progress_bar = tqdm(total=len(validation_dataloader), desc="Validating")
        with torch.no_grad():
            for val_batch in validation_dataloader:
                if val_batch is None:
                    if accelerator.is_main_process:
                        val_progress_bar.update(1)
                    continue
                loss = validate(val_batch, pipeline, unet, device)
                val_loss += loss.item()
                valid_count += 1
                if accelerator.is_main_process:
                    val_progress_bar.update(1)
        if valid_count > 0:
            val_loss /= valid_count

        return val_loss
        
    best_val_loss = 10
    while True:
        if accelerator.is_main_process:
            wait_for_videos(dataset, train_dir, min_videos=train_data.min_videos)

        try:
            for batch in train_dataloader:
                if batch is None:
                    continue

                with accelerator.autocast():
                    loss = finetune_unet(batch, accelerator.device)

                accelerator.backward(loss)
                accelerator.clip_grad_norm_(unet.parameters(), max_norm=1.0)

                optimizer.step()
                optimizer.zero_grad()

                if accelerator.is_main_process:
                    with step_lock:
                        global_step += 1
                        
                    wandb.log({
                        "loss": loss.item()
                    })

                if global_step % sample_steps == 0 and global_step > 1:
                    with accelerator.autocast():
                        unet.eval()
                        unet._set_gradient_checkpointing(value=not gradient_checkpointing)

                        prompt = random.choice(batch["text_prompt"])
                        save_filename = f"{global_step}-{prompt}"
                        out_file = f"{output_dir}/samples/{save_filename}.mp4"
                        encoded_out_file = f"{output_dir}/samples/{save_filename}_encoded.mp4"

                        with torch.no_grad():
                            latents = pipeline(
                                prompt,
                                device=accelerator.device,
                                width=sample_data.width,
                                height=sample_data.height,
                                num_frames=sample_data.num_frames,
                                num_inference_steps=sample_data.num_inference_steps,
                                guidance_scale=sample_data.guidance_scale
                            )
                            tensor = decode(latents, pipeline.vae)

                        save_video(tensor, out_file, sample_data.fps)

                        try:
                            encode_video(out_file, encoded_out_file, get_video_height(out_file))
                            os.remove(out_file)
                        except:
                            pass

                        torch.cuda.empty_cache()

                        unet.train()
                        unet._set_gradient_checkpointing(value=gradient_checkpointing)
                        
                if global_step % validation_steps == 0 and global_step > 1:
                    unet.eval()
                    unet._set_gradient_checkpointing(value=not gradient_checkpointing)
                    
                    if validation_data.save_anyway:
                        save_path = os.path.join(output_dir, f"checkpoint-{global_step}")
                        os.makedirs(save_path, exist_ok=True)
                        pipeline.unet = unet
                        pipeline.save_pretrained(save_path)
                        print(f"Saved model at {save_path} on step {global_step}")

                    with accelerator.autocast():
                        val_loss = perform_validation(accelerator, pipeline, unet, validation_dataloader, accelerator.device)
                        wandb.log({
                            "val_loss": val_loss
                        })
                        if accelerator.is_main_process:
                            if val_loss < best_val_loss:
                                best_val_loss = val_loss
                                save_path = os.path.join(output_dir, f"validated-{global_step}")
                                os.makedirs(save_path, exist_ok=True)
                                pipeline.unet = unet
                                pipeline.save_pretrained(save_path)
                                print(f"Saved model at {save_path} on step {global_step}")

                    unet._set_gradient_checkpointing(value=gradient_checkpointing)
                    unet.train()
                    
                    torch.cuda.empty_cache()
                    gc.collect()
        except Exception as e:
            print(e)           
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="training.yaml")
    parser.add_argument('--local_rank', default=-1, type=int, help='Local rank of this process. Used for distributed training.')
    args = parser.parse_args()

    main(**OmegaConf.load(args.config))