import os, sys
os.environ['HF_HOME'] = '../cache/'

from diffusers import StableUnCLIPImg2ImgPipeline, EulerDiscreteScheduler, DDIMScheduler
from matplotlib import pyplot as plt
from tqdm import tqdm

import torch, random
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Subset
import numpy as np

from torch.amp import autocast
from PIL import Image

from sample_diffusion_imagebind import generate_image_from_audio

sys.path.append('../')
from data_processing.build_diffusion_dataset import AudioImageDataset_Diffusion
from imagebind import data as ib_data
from imagebind.models import imagebind_model as ib_model
from imagebind.models.imagebind_model import ModalityType

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)


if __name__ == "__main__":
    model_id = "stabilityai/stable-diffusion-2-1-unclip"

    # Sample Code
    scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
    pipe = StableUnCLIPImg2ImgPipeline.from_pretrained(model_id, scheduler=scheduler)
    pipe = pipe.to("cuda")

    # prompt = "a photo of an astronaut riding a horse on mars"
    # image = pipe(prompt).images[0]  
        
    # image.save("astronaut_rides_horse.png")
    print("Model loaded")

    config = {
        'device': 'cuda',

        ########### batch size restricted to be 16 in forward pass??? #################
        'batch size': 2,
        'train ratio': 0.9,
        'validation ratio': 0.1,
        'device': 'cuda',
        'epochs': 280,
        'unet_lr': 1e-5,

        'condition_embedding_dim': 1024
    }

    device = config['device']

    # Load the dataset
    ds_path = "../data/DS_ib_airport.pt"
    ds = torch.load(ds_path, weights_only=False)

    # Split Train, Val, Test
    train_size = int(config['train ratio']*len(ds))
    val_size = len(ds) - train_size
    
    train, val = torch.utils.data.random_split(ds, [train_size, val_size])
    train = Subset(train, range(4))
    train_dataloader = torch.utils.data.DataLoader(train, batch_size=config['batch size'], shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val, batch_size=config['batch size'], shuffle=True)   
    print("Dataset loaded")

    '''
    ========================= Model Additional Layers =========================
    '''
    # Define the model additional layers
    image_bind = ib_model.imagebind_huge(True).eval().to('cuda')
    unet = pipe.unet
    vae = pipe.vae
    image_encoder = pipe._encode_image

    '''
    ========================= Model Training =========================
    '''

    # Optimizer and loss
    optimizer = optim.AdamW([
        {"params": unet.parameters(), "lr": config['unet_lr']}
    ])
    criterion = nn.MSELoss()

    # Everything about gradient
    vae.requires_grad_(False)
    unet.requires_grad_(True)
    image_bind.requires_grad_(False)

    # Scaler for mixed precision
    scaler = torch.amp.GradScaler(device)
    unet.train()
    # Training loop
    for epoch in range(config['epochs']):
        
        total_loss = 0

        for audio_embedding, images in tqdm(train_dataloader):
            
            # Prepare Placeholder Text Prompts:
            prompt = ""
            prompt_embeds, negative_prompt_embeds = pipe._encode_prompt(
                prompt=prompt,
                device=device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=True
            )
            # prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
            prompt_embeds = prompt_embeds.unsqueeze(0).repeat(config['batch size'], 1, 1)

            # Preprocess audio and images
            audio_embedding = audio_embedding.to(device)
            clean_images = images.to(device)
            timesteps = torch.randint(
                0,
                scheduler.num_train_timesteps,
                (audio_embedding.shape[0],),
                device=clean_images.device,
            ).long()

            with autocast(device,):

                # Pass audio embedding through image encoder
                _, img_embedding = image_encoder(
                    image=None,
                    device=device,
                    batch_size=audio_embedding.shape[0],
                    num_images_per_prompt=1,
                    do_classifier_free_guidance=True,
                    noise_level=0,
                    generator=None,
                    image_embeds=audio_embedding,
                ).chunk(2)

                latents = vae.encode(clean_images).latent_dist.sample().detach()
                posterior = latents * 0.18215

                # Add noise
                noise = torch.randn_like(posterior)
                # noisy_latents = posterior + noise
                noisy_latents = scheduler.add_noise(posterior, noise, timesteps)
                # Predict noise with conditional UNet
                predicted_noise = unet(noisy_latents, timesteps, encoder_hidden_states=prompt_embeds, class_labels=img_embedding).sample
                # Compute loss
                loss = criterion(predicted_noise, noise)

            # Backpropagation and optimization
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(unet.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            optimizer.zero_grad()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_dataloader)}")

    # Save the model
    torch.save(unet.state_dict(), "./imagebind_trained_unet.pth")

    # Example usage - Load one audio embedding from dataset

    audio_embedding = val.dataset.audio_data[0].unsqueeze(0)
    generate_image_from_audio(audio_embedding, pipe, image_bind, scheduler)
