import os, sys
os.environ['HF_HOME'] = '../cache/'

from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler, DDIMScheduler
from matplotlib import pyplot as plt
from tqdm import tqdm

import torch, random
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import soundfile as sf
import numpy as np

from torch.amp import autocast

sys.path.append('../')
from data_processing.build_diffusion_dataset import AudioImageDataset_Diffusion
from PIL import Image

from stable_diffusion_openl3 import generate_image_from_audio, AudioConditionalUNet

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)


if __name__ == "__main__":
    # Load the dataset
    ds_path = "../data/DS_airport_diffusion.pt"
    ds = torch.load(ds_path, weights_only=False)
    
    # Split Train, Val, Test
    train_size = int(0.8*len(ds))
    val_size = len(ds) - train_size
    
    train, val = torch.utils.data.random_split(ds, [train_size, val_size])
    # train = Subset(train, range(1))

    # load model
    # OverallDiffusion
    model_id = "stabilityai/stable-diffusion-2-1-base"
    scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
    pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler)
    pipe = pipe.to("cuda")

    # Load Our UNet
    audio_embedding_dim = 6144
    condition_embedding_dim = 1024
    conditional_unet = AudioConditionalUNet(pipe.unet, audio_embedding_dim, condition_embedding_dim).to('cuda')
    conditional_unet.load_state_dict(torch.load('../diffusion_models/conditional_unet.pth', weights_only=True))


    # Sample
    audio_embedding = train.dataset.audio_data[0].unsqueeze(0)
    image = train.dataset.img_data[0]
    image.save(f'original_image.jpg')
    cond_embedding = conditional_unet.audio_conditioning(audio_embedding)
    uncond_embedding = torch.zeros_like(cond_embedding)
    pipe.unet = conditional_unet.unet

    # pipe(prompt_embeds=cond_embedding, negative_prompt_embeds = uncond_embedding, height=256, width=256).images[0].save(f'generated_image.jpg')

    generate_image_from_audio(audio_embedding, conditional_unet, pipe.vae, scheduler, 50)
