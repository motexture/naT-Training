import os
import decord
import numpy as np
import torch
import torch.nn.functional as F
import requests
import glob

from mpi4py import MPI
from typing import Optional
from torch.utils.data import Dataset

decord.bridge.set_bridge('torch')

COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()
SIZE = COMM.Get_size()

def request_save(url, save_fp):
    try:
        img_data = requests.get(url, timeout=5).content
        with open(save_fp, 'wb') as handler:
            handler.write(img_data)
    except Exception as e:
        print(e)
        pass

class ValidationDataset(Dataset):
    def __init__(
        self,
        width: int = 256,
        height: int = 256,
        num_frames: int = 24,
        frame_step: int = 1,
        path: str = "./data",
        tokenizer = None,
        tokenizer_2 = None,
        text_encoder = None,
        text_encoder_2 = None,
        unet = None,
        is_validation = False,
        **kwargs
    ):
        self.video_files = []
        self.is_validation = is_validation

        self.find_videos(path)
        
        self.width = width
        self.height = height
        self.n_sample_frames = num_frames
        self.frame_step = frame_step

        self.tokenizer = tokenizer
        self.tokenizer_2 = tokenizer_2
        self.text_encoder = text_encoder
        self.text_encoder_2 = text_encoder_2
        self.unet = unet

    def process_file(self, file_path):
        if file_path.endswith('.mp4'):
            return file_path
        return None

    def find_videos(self, path):
        video_files = glob.glob(os.path.join(path, "*.mp4"))  # Adjust the extension as needed
        self.video_files = [self.process_file(f) for f in video_files if self.process_file(f) is not None]

    def get_frames(self, vid_path):
        try:
            vr = decord.VideoReader(vid_path)
            video_length = len(vr)
            
            if video_length < self.n_sample_frames * 1:
                raise ValueError(f"Video too short. Need at least {self.n_sample_frames * 2} frames for a step of 3, but found {video_length}.")

            frame_step = self.frame_step

            idxs = np.arange(self.n_sample_frames) * frame_step
            idxs = np.clip(idxs, 0, video_length - 1)

            video = vr.get_batch(idxs)

            if video.shape[-1] == 4:
                video = video[..., :3]

            video = video.permute(0, 3, 1, 2).float()

            original_height, original_width = video.shape[-2:]
            crop_height = min(original_height, self.height)
            crop_width = min(original_width, self.width)

            start_y = (original_height - crop_height) // 2
            start_x = (original_width - crop_width) // 2

            video = video[:, :, start_y:start_y + crop_height, start_x:start_x + crop_width]

            video = F.interpolate(video, size=(self.height, self.width), mode='bicubic', align_corners=False)
            
            video = video.unsqueeze(0).permute(0, 2, 1, 3, 4)
            video = (video / 127.5) - 1.0

            return video
        except Exception as e:
            print(f"Unexpected error while processing {vid_path}: {e}")
            return None

    def get_add_time_ids(self, original_size, crops_coords_top_left, target_size, dtype):
        add_time_ids = list(original_size + crops_coords_top_left + target_size)

        passed_add_embed_dim = (
            self.unet.config.addition_time_embed_dim * len(add_time_ids) + self.text_encoder_2.config.projection_dim
        )
        expected_add_embed_dim = self.unet.add_embedding.linear_1.in_features

        if expected_add_embed_dim != passed_add_embed_dim:
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
            )

        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
        return add_time_ids
    
    def encode_prompt(
        self,
        prompt,
        num_images_per_prompt: int = 1,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None
    ):
        tokenizers = [self.tokenizer, self.tokenizer_2] if self.tokenizer is not None else [self.tokenizer_2]
        text_encoders = (
            [self.text_encoder, self.text_encoder_2] if self.text_encoder is not None else [self.text_encoder_2]
        )

        if prompt_embeds is None:
            prompt_embeds_list = []
            for tokenizer, text_encoder in zip(tokenizers, text_encoders):
                text_inputs = tokenizer(
                    prompt,
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )

                text_input_ids = text_inputs.input_ids.to(device=text_encoder.device)
                prompt_embeds = text_encoder(
                    text_input_ids,
                    output_hidden_states=True,
                )

                pooled_prompt_embeds = prompt_embeds[0]
                prompt_embeds = prompt_embeds.hidden_states[-2]

                bs_embed, seq_len, _ = prompt_embeds.shape
                prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
                prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

                prompt_embeds_list.append(prompt_embeds)

            prompt_embeds = torch.cat(prompt_embeds_list, dim=-1)

        bs_embed = pooled_prompt_embeds.shape[0]
        pooled_prompt_embeds = pooled_prompt_embeds.repeat(1, num_images_per_prompt).view(
            bs_embed * num_images_per_prompt, -1
        )

        return prompt_embeds, pooled_prompt_embeds

    @staticmethod
    def __getname__():
        return 'folder'

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, index):
        try:
            video = self.get_frames(self.video_files[index])
            if video is None:
                print(f"Invalid video at index {index}: {self.video_files[index]}")
                return None
        except Exception as e:
            print(f"Error loading video at index {index}: {e}")
            return None

        basename = os.path.basename(self.video_files[index]).replace('.mp4', '').replace('_', ' ')
        split_basename = basename.split('-')

        if len(split_basename) > 1:
            prompt = '-'.join(split_basename[:-1])
        else:
            prompt = split_basename[0]

        prompt_embeds, pooled_prompt_embeds = self.encode_prompt(prompt=prompt)

        original_size = (self.height, self.width)
        target_size = (self.height, self.width)

        add_text_embeds = pooled_prompt_embeds
        add_time_ids = self.get_add_time_ids(
            original_size, (0, 0), target_size, dtype=prompt_embeds.dtype
        )
        
        return {
            "pixel_values": video[0],
            "prompt_embeds": prompt_embeds[0],
            "add_text_embeds": add_text_embeds[0],
            "add_time_ids": add_time_ids[0],
            "text_prompt": prompt,
            "file": basename,
            "dataset": self.__getname__()
        }
