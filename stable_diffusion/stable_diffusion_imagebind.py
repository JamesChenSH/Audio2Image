import os
os.environ['HF_HOME'] = './cache/'

from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler, DDIMScheduler
from matplotlib import pyplot as plt
from tqdm import tqdm

import torch, random
import torch.nn as nn
import torch.optim as optim
import numpy as np

from torch.amp import autocast
from data_processing.build_diffusion_dataset import AudioImageDataset_Diffusion
from PIL import Image

from imagebind import data as ib_data
from imagebind.models import imagebind_model as ib_model
from imagebind.models.imagebind_model import ModalityType

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)


if __name__ == "__main__":
    model_id = "stabilityai/stable-diffusion-2-1-base"

    # Sample Code
    scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
    pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler)
    pipe = pipe.to("cuda")

    # prompt = "a photo of an astronaut riding a horse on mars"
    # image = pipe(prompt).images[0]  
        
    # image.save("astronaut_rides_horse.png")
    print("Model loaded")

    config = {
        ########### batch size restricted to be 16 in forward pass??? #################
        'batch size': 32,
        'train ratio': 0.9,
        'validation ratio': 0.1,
        'device': 'cuda',
        'epochs': 280,
        'linear_lr': 1e-3,
        'unet_lr': 1e-5,

        'condition_embedding_dim': 1024
    }

    # Load the dataset
    ds_path = "data/DS_diffusion.pt"
    ds = torch.load(ds_path, weights_only=False)
    
    # Split Train, Val, Test
    train_size = int(config['train ratio']*len(ds))
    val_size = len(ds) - train_size
    
    train, val = torch.utils.data.random_split(ds, [train_size, val_size])
    # train = Subset(train, range(1))
    train_dataloader = torch.utils.data.DataLoader(train, batch_size=config['batch size'], shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val, batch_size=config['batch size'], shuffle=True)   

    '''
    ========================= Model Additional Layers =========================
    '''
    # Define the model additional layers
    audio_embedding_dim = 6144
    condition_embedding_dim = config['condition_embedding_dim']
    image_bind = ib_model.imagebind_huge(True).eval().to('cuda')
    unet = pipe.unet
    vae = pipe.vae

    '''
    ========================= Test Model ==========================
    '''
    # audio_embedding = val.dataset.audio_data[0]
    # print(scheduler.num_train_timesteps)
    # generate_image_from_audio(audio_embedding, conditional_unet, vae, scheduler)
    # exit()
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

    # Scaler for mixed precision
    scaler = torch.amp.GradScaler("cuda")
    unet.train()
    # Training loop
    for epoch in range(config['epochs']):
        
        total_loss = 0

        for audio, images in tqdm(train_dataloader):
            # Preprocess audio and images
            audio = audio.to(config['device'])
            clean_images = images.to(config['device'])
            timesteps = torch.randint(
                0,
                scheduler.num_train_timesteps,
                (audio.shape[0],),
                device=clean_images.device,
            ).long()

            with autocast("cuda",):

                audio_embedding = image_bind({
                    ModalityType.AUDIO, audio
                })

                latents = vae.encode(clean_images).latent_dist
                posterior = latents.sample() * 0.18215

                # Add noise
                noise = torch.randn_like(posterior)
                # noisy_latents = posterior + noise
                noisy_latents = scheduler.add_noise(posterior, noise, timesteps)

                # Predict noise with conditional UNet
                predicted_noise = unet(noisy_latents, timesteps, encoder_hidden_states=audio_embedding).sample
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
    torch.save(unet.state_dict(), "diffusion_models/imagebind_trained_unet.pth")

    # Example usage - Load one audio embedding from dataset

    audio_embedding = val.dataset.audio_data[0].unsqueeze(0)
    generate_image_from_audio(audio_embedding, pipe, image_bind, scheduler)
