import os, sys
os.environ['HF_HOME'] = '../cache/'

from diffusers import StableUnCLIPImg2ImgPipeline, EulerDiscreteScheduler
from matplotlib import pyplot as plt
from tqdm import tqdm

import torch, random
import torch.nn as nn
import torch.optim as optim
# from torch.utils.data import Subset
import numpy as np

from torch.amp import autocast
from datetime import datetime

sys.path.append('../')
from data_processing.build_diffusion_dataset import AudioImageDataset_Diffusion
# from imagebind import data as ib_data
# from imagebind.models import imagebind_model as ib_model
# from imagebind.models.imagebind_model import ModalityType

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
        'epochs': 80,
        'unet_lr': 1e-6,

    }

    device = config['device']

    # Load the dataset 
    ds_path = "../data/DS_ib_768.pt"
    ds = torch.load(ds_path, weights_only=False)

    # Split Train, Val, Test
    train_size = int(config['train ratio']*len(ds))
    val_size = len(ds) - train_size
    
    train, val = torch.utils.data.random_split(ds, [train_size, val_size])
    # train = Subset(train, range(4))
    train_dataloader = torch.utils.data.DataLoader(train, batch_size=config['batch size'], shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val, batch_size=config['batch size'], shuffle=True)   
    print("Dataset loaded")

    '''
    ========================= Model Additional Layers =========================
    '''
    # Define the model additional layers
    unet = pipe.unet
    vae = pipe.vae
    image_encoder = pipe._encode_image

    '''
    ========================= Model Training =========================
    '''
    # Optimizer and loss
    start_lr = config['unet_lr']
    total_iters = config['epochs'] * len(train_dataloader)
    optimizer = optim.AdamW([{'params':unet.parameters(), 'lr':start_lr,  'betas':(0.9, 0.999), 'weight_decay':0.01}])
    criterion = nn.MSELoss()

    # Everything about gradient
    vae.requires_grad_(False)
    unet.requires_grad_(True)
    pipe.text_encoder.requires_grad_(False)

    # Use common text prompt embed
    # Prepare Placeholder Text Prompts:
    prompt = ""
    with torch.no_grad():
        prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
            prompt=prompt,
            device=device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=True
        )

    # Prepare for logging
    losses = []
    learning_rates = []


    # Scaler for mixed precision
    scaler = torch.amp.GradScaler(device)
    start_time = datetime.now()
    unet.train()
    # Training loop
    for epoch in range(config['epochs']):
        
        total_loss = 0
        for i, (audio_embedding, images) in enumerate(tqdm(train_dataloader)):
            
            optimizer.zero_grad()
            # Preprocess audio and images
            audio_embedding = audio_embedding.to(device)
            clean_images = images.to(device)
            encoder_hidden_state = prompt_embeds.repeat(audio_embedding.shape[0], 1, 1)

            # Config timesteps
            timesteps = torch.randint(
                0,
                scheduler.config.num_train_timesteps,
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
                predicted_noise = unet(noisy_latents, timesteps, encoder_hidden_states=encoder_hidden_state, class_labels=img_embedding).sample
                # Compute loss
                loss = criterion(predicted_noise, noise)

            # Log
            learning_rates.append(optimizer.param_groups[0]['lr'])
            losses.append(loss.item())

            # Backpropagation and optimization
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(unet.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            
            iters = epoch * len(train_dataloader) + i
            lr = start_lr * (1 - iters / total_iters) ** 0.9

            optimizer.param_groups[0]['lr'] = lr


        print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_dataloader)}")

    # Save the model
    try:
        torch.save(unet.state_dict(), "./imagebind_trained_unet_90_epoch_no_prompt.pth")
        config['losses'] = losses
        config['learning_rates'] = learning_rates
        import json
        with open(f"training_log_{start_time.date()}_{start_time.hour}_{start_time.minute}_{start_time.second}.json", 'w') as f:
            json.dump(config, f)
    except:
        print("Saving to cur directory failed, saving in 340")
        torch.save(unet.state_dict(), "/w/340/jameschen/imagebind_trained_unet_90_epoch_no_prompt.pth")
        config['losses'] = losses
        config['learning_rates'] = learning_rates
        import json
        with open(f"/w/340/jameschen/training_log_{start_time.date()}_{start_time.hour}_{start_time.minute}_{start_time.second}.json", 'w') as f:
            json.dump(config, f)
    # Example usage - Load one audio embedding from dataset

    # audio_embedding = val.dataset.audio_data[0].unsqueeze(0)

    # image_bind = ib_model.imagebind_huge(True).eval().to('cuda')
    # generate_image_from_audio(audio_embedding, pipe, image_bind, scheduler)


