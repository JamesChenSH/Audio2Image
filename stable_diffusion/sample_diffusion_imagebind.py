import os
os.environ['HF_HOME'] = './cache/'

from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler, DDIMScheduler
from matplotlib import pyplot as plt
from tqdm import tqdm

import torch, random
import numpy as np

from torch.amp import autocast
from data_processing.build_diffusion_dataset import AudioImageDataset_Diffusion
from PIL import Image

from stable_diffusion_imagebind import generate_image_from_audio

from imagebind import data as ib_data
from imagebind.models import imagebind_model as ib_model
from imagebind.models.imagebind_model import ModalityType

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)


if __name__ == "__main__":
    # Load the dataset
    # ds_path = "data/DS_airport_diffusion.pt"
    # ds = torch.load(ds_path, weights_only=False)
    
    # Split Train, Val, Test
    # train_size = int(0.8*len(ds))
    # val_size = len(ds) - train_size
    
    # train, val = torch.utils.data.random_split(ds, [train_size, val_size])
    # train = Subset(train, range(1))

    # load model
    # OverallDiffusion
    model_id = "stabilityai/stable-diffusion-2-1-base"
    scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
    pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler)
    pipe = pipe.to("cuda")

    audio_sample_path = "../data/sound/train_station/00355_1.wav"
    audio_tokenized = ib_data.load_and_transform_audio_data(audio_sample_path)

    # Sample
    # audio_embedding = train.dataset.audio_data[0].unsqueeze(0)
    generate_image_from_audio(audio_tokenized, pipe.unet, pipe.vae, scheduler, 1000)
