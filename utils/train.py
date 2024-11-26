import os
import decord
import numpy as np
import random
import torch
import torch.nn.functional as F
import concurrent.futures
import requests
import pandas as pd
import time
import threading
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

class DynamicDatasetDownloader:
    def __init__(self, train_dir, csv_path, processes=16, max_videos=200):
        self.train_dir = train_dir
        self.csv_path = csv_path
        self.processes = processes
        self.max_videos = max_videos
        self.download_in_progress = False
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=processes)
        self.setup_dataset_directory()
        self.lock = threading.Lock()
        self.df = self.load_csv()

    def setup_dataset_directory(self):
        if not os.path.exists(self.train_dir):
            os.makedirs(self.train_dir)

    def load_csv(self):
        df = pd.read_csv(self.csv_path)
        df.dropna(subset=['contentUrl'], inplace=True)  # Ensure URLs are not NaN
        return df.sample(frac=1).reset_index(drop=True)  # Shuffle the DataFrame

    def check_and_download_more_videos(self):
        num_videos = len([f for f in os.listdir(self.train_dir) if f.endswith('.mp4')])

        with self.lock:
            if not self.download_in_progress and num_videos < self.max_videos:
                self.download_in_progress = True
                self.executor.submit(self.download_more_videos)

    def download_more_videos(self):
        try:
            i = 0
            urls_todo = []
            save_fps = []
            self.df = self.df.sample(frac=1).reset_index(drop=True)
            for idx, row in self.df.iterrows():
                if i >= self.max_videos:
                    break
                video_name = f"{row['name']}-{row['videoid']}"
                sanitized_filename = video_name.replace('.mp4', '').replace('_', ' ')
                video_fp = os.path.join(self.train_dir, sanitized_filename + '.mp4')
                urls_todo.append(row['contentUrl'])
                save_fps.append(video_fp)
                i += 1
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.processes) as executor:
                future_to_url = {executor.submit(request_save, url, fp): url for url, fp in zip(urls_todo, save_fps)}

            with self.lock:
                self.download_in_progress = False
        except Exception as e:
            with self.lock:
                self.download_in_progress = False
            print(f"Error during download: {e}")

    def monitor_training_and_download(self):
        while True:
            self.check_and_download_more_videos()
            time.sleep(3)

    def start_downloader_thread(self):
        download_thread = threading.Thread(target=self.monitor_training_and_download)
        download_thread.daemon = True
        download_thread.start()

class TrainDataset(Dataset):
    def __init__(
        self,
        width: int = 256,
        height: int = 256,
        num_frames: int = 32,
        frame_step: int = 1,
        path: str = "./data",
        tokenizer = None,
        tokenizer_2 = None,
        text_encoder = None,
        text_encoder_2 = None,
        unet = None,
        **kwargs
    ):
        self.video_files = []

        self.path = path
        self.find_videos(self.path)
        
        self.width = width
        self.height = height
        self.num_frames = num_frames
        self.frame_step = frame_step

        self.tokenizer = tokenizer
        self.tokenizer_2 = tokenizer_2
        self.text_encoder = text_encoder
        self.text_encoder_2 = text_encoder_2
        self.unet = unet

        self.shuffle_videos()

    def shuffle_videos(self):
        random.shuffle(self.video_files)
    
    def process_file(self, file_path):
        if file_path.endswith('.mp4'):
            return file_path
        return None

    def find_videos(self, path):
        self.video_files.clear()
        video_files = glob.glob(os.path.join(path, "*.mp4"))
        self.video_files = [self.process_file(f) for f in video_files if self.process_file(f) is not None]
        random.shuffle(self.video_files)

    def get_frames(self, vid_path):
        try:
            vr = decord.VideoReader(vid_path)
            video_length = len(vr)
            
            frame_step = self.frame_step
            effective_length = video_length // frame_step
            
            effective_idx = random.randint(0, (effective_length - self.num_frames))
            idxs = frame_step * np.arange(effective_idx, effective_idx + self.num_frames)
            
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
        except Exception as e:
            return None
        
        if video is None:
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

        try:
            os.remove(self.video_files[index])
        except:
            pass
        self.find_videos(self.path)
        
        return {
            "pixel_values": video[0],
            "prompt_embeds": prompt_embeds[0],
            "add_text_embeds": add_text_embeds[0],
            "add_time_ids": add_time_ids[0],
            "text_prompt": prompt,
            "file": basename,
            "dataset": self.__getname__()
        }
