import os
os.environ['HF_HOME'] = './cache/'

from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler, DDIMScheduler
from matplotlib import pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import torchsummary
import soundfile as sf
import numpy as np

from torch.amp import autocast
from data_processing.build_diffusion_dataset import AudioImageDataset_Diffusion

'''
=======================  !!!NOTE!!! ==============================
I am currently stopping here since Open L3 does not have pytorch support
from its original team. Looking for solutions rn.
'''
def initialize_weights(m):
    if isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class AudioConditionalUNet(nn.Module):
    def __init__(self, unet, audio_embedding_dim, embedding_dim):
        super(AudioConditionalUNet, self).__init__()
        self.audio_conditioning = nn.Linear(audio_embedding_dim, embedding_dim)
        init.xavier_uniform_(self.audio_conditioning.weight)
        if self.audio_conditioning.bias is not None:
            nn.init.zeros_(self.audio_conditioning.bias)
        self.unet = unet

    def forward(self, latent_image, timestep, audio_input):
        # print(f"latent_image shape: {latent_image.shape}") 
        # print(f"timestep shape: {timestep.shape}")         
        # print(f"audio_input shape: {audio_input.shape}") 
        # Generate audio conditioning
        audio_embed = self.audio_conditioning(audio_input)
        # print(f"audio_embed shape: {audio_embed.shape}")
        # Pass to UNet
        return self.unet(latent_image, timestep, audio_embed)


'''
============================ Inference Function =============================
'''

# For inference
def generate_image_from_audio(audio_embedding, conditional_unet, vae, scheduler, num_steps=50):


    # Initialize random latent vector
    # latent_dim = 768
    # image_latents = torch.randn(1, latent_dim, 256, 256).to("cuda")
    ################## Fix suggested by GPT #######################
    latent_dim = (768, 4, 64, 64)  # Match UNet in_channels
    image_latents = torch.randn(latent_dim).to("cuda", dtype=torch.float16)
    ###############################################################
    audio_embedding = audio_embedding.half().to("cuda")
    image_latents = image_latents.half().to("cuda")

    # Iterative denoising
    for t in reversed(range(num_steps)):
        predicted_noise = conditional_unet(image_latents, timestep=t, audio_input=audio_embedding)
        image_latents = scheduler.step(predicted_noise, t, image_latents).prev_sample

    # Decode the final latent vector
    generated_image = vae.decode(image_latents).clamp(0, 1)

    # Display the image
    plt.imshow(generated_image.squeeze().permute(1, 2, 0).cpu().numpy())
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    model_id = "stabilityai/stable-diffusion-2-1-base"

    # Sample Code
    scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
    pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=torch.float16)
    pipe = pipe.to("cuda")

    # prompt = "a photo of an astronaut riding a horse on mars"
    # image = pipe(prompt).images[0]  
        
    # image.save("astronaut_rides_horse.png")
    print("Model loaded")

    config = {
        ########### batch size restricted to be 16 in forward pass??? #################
        'batch size': 16,
        'train ratio': 0.8,
        'validation ratio': 0.1,
        'device': 'cuda',
        'epochs': 1,
        'lr': 1e-6,

        'condition_embedding_dim': 1024
    }

    # Load the dataset
    ds_path = "data/DS_airport_diffusion.pt"
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
    unet = pipe.unet
    conditional_unet = AudioConditionalUNet(unet, audio_embedding_dim, condition_embedding_dim).to(config['device'])
    vae = pipe.vae
    '''
    ========================= Model Training =========================
    '''

    # Optimizer and loss
    optimizer = optim.AdamW(conditional_unet.audio_conditioning.parameters(), lr=config['lr'])
    criterion = nn.MSELoss()
    torch.nn.utils.clip_grad_norm_(conditional_unet.audio_conditioning.parameters(), max_norm=1.0)

    # Training loop
    for epoch in range(config['epochs']):
        for audio, images in tqdm(train_dataloader):
            # Preprocess audio and images
            audio = audio.to(config['device']).to(torch.float16)
            clean_images = images.to(config['device'])

            latents = vae.encode(clean_images.half()).latent_dist
            posterior = latents.mean

            timesteps = torch.randint(
                0,
                scheduler.num_train_timesteps,
                (audio.shape[0],),
                device=clean_images.device,
            ).long()

            # Add noise
            noise = torch.randn_like(posterior)
            # noisy_latents = posterior + noise
            noisy_latents = scheduler.add_noise(posterior, noise, timesteps)

            with autocast("cuda",):
                # Predict noise with conditional UNet
                predicted_noise = conditional_unet(noisy_latents, timesteps, audio).sample
                # Compute loss
                loss = criterion(predicted_noise, noise)

            # Backpropagation and optimization
            optimizer.zero_grad()
            loss.backward()
            for param in conditional_unet.audio_conditioning.parameters():
                print(f"Before optimizer step: {param.data}")

            optimizer.step()

            for param in conditional_unet.audio_conditioning.parameters():
                print(f"After optimizer step: {param.data}")

            print(f"Epoch {epoch+1}, Loss: {loss.item()}")
            if epoch == 1:
                exit()

    # Example usage - Load one audio embedding from dataset

    audio_embedding = val.dataset.audio_data[0]
    # generate_image_from_audio(audio_embedding, conditional_unet, vae, scheduler)
