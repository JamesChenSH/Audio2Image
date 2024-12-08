import os
os.environ['HF_HOME'] = './cache/'

from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler, DDIMScheduler
from matplotlib import pyplot as plt
from tqdm import tqdm

import torch, random
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import torchsummary
import soundfile as sf
import numpy as np

from torch.amp import autocast
from data_processing.build_diffusion_dataset import AudioImageDataset_Diffusion


random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

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
        audio_input = (audio_input - audio_input.mean()) / (audio_input.std() + 1e-8)

        audio_embed = self.audio_conditioning(audio_input)
        # print(f"audio_embed shape: {audio_embed.shape}")
        # Pass to UNet
        return self.unet(latent_image, timestep, encoder_hidden_states=audio_embed)


'''
============================ Inference Function =============================
'''

# For inference
def generate_image_from_audio(audio_embedding:torch.Tensor, conditional_unet, vae, scheduler, num_steps=50):


    # Initialize random latent vector
    # latent_dim = 768
    # image_latents = torch.randn(1, latent_dim, 256, 256).to("cuda")
    ################## Fix suggested by GPT #######################
    latent_dim = (1, 4, 32, 32)  # Match UNet in_channels
    image_latents = torch.randn(latent_dim).to("cuda")
    ###############################################################
    audio_embedding = audio_embedding.to("cuda").unsqueeze(0)
    image_latents = image_latents.to("cuda")
    # Iterative denoising
    for t in reversed(range(num_steps)):
        with torch.no_grad(), autocast("cuda",):
            predicted_noise = conditional_unet(image_latents, timestep=torch.tensor([t,], dtype=torch.long).cuda(), audio_input=audio_embedding).sample
            predicted_noise = scheduler.scale_model_input(predicted_noise, t)
            image_latents = scheduler.step(predicted_noise, torch.tensor([t,], dtype=torch.long).cuda(), image_latents).prev_sample

    # Decode the final latent vector
    with torch.no_grad(), autocast("cuda",):
        generated_image = vae.decode(image_latents).sample.clamp(0, 1)

    # Display the image
    plt.imshow(generated_image.squeeze().permute(1, 2, 0).to(dtype=torch.float32).cpu().detach().numpy())
    plt.axis('off')
    plt.savefig('sample_image.png')


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
        'epochs': 3000,
        'linear_lr': 1e-3,
        'unet_lr': 1e-5,

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
    optimizer = optim.AdamW([
        {"params": conditional_unet.audio_conditioning.parameters(), "lr": config['linear_lr']},
        {"params": conditional_unet.unet.parameters(), "lr": config['unet_lr']}
    ])
    criterion = nn.MSELoss()

    # Everything about gradient
    vae.requires_grad_(False)
    conditional_unet.requires_grad_(True)

    # Scaler for mixed precision
    scaler = torch.amp.GradScaler("cuda")

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
                latents = vae.encode(clean_images).latent_dist
                posterior = latents.sample() * 0.18215

                # Add noise
                noise = torch.randn_like(posterior)
                # noisy_latents = posterior + noise
                noisy_latents = scheduler.add_noise(posterior, noise, timesteps)

                # Predict noise with conditional UNet
                predicted_noise = conditional_unet(noisy_latents, timesteps, audio).sample
                # Compute loss
                loss = criterion(predicted_noise, noise)

            # Backpropagation and optimization
            scaler.scale(loss).backward()
            # for name, param in conditional_unet.named_parameters():
            #     if param.requires_grad:
            #         if param.grad is None:
            #             print(f"Gradient for {name} is None")
            #         else:
            #             print(f"Gradient for {name}: Norm = {param.grad.norm()}")

            torch.nn.utils.clip_grad_norm_(conditional_unet.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            optimizer.zero_grad()
            # for name, param in conditional_unet.named_parameters():
                # print(f"Parameter: {name}, Before optimizer step: {param.mean()}")

            # optimizer.step()

            # for name, param in conditional_unet.named_parameters():
            #     print(f"Parameter: {name}, After optimizer step: {param.mean()}")

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_dataloader)}")

    # Save the model
    torch.save(conditional_unet.state_dict(), "diffusion_models/conditional_unet.pth")
    torch.save(vae.state_dict(), "diffusion_models/vae.pth")

    # Example usage - Load one audio embedding from dataset

    audio_embedding = val.dataset.audio_data[0]
    generate_image_from_audio(audio_embedding, conditional_unet, vae, scheduler)
