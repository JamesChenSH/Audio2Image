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
from PIL import Image

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

GENERATOR = torch.Generator(device='cuda').manual_seed(42)

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

        # Placeholder for audio embedding, used for testing unconditional diffusion
        uncond_placeholder = torch.zeros_like(audio_embed)
        audio_embed_input = torch.cat([uncond_placeholder, audio_embed])

        # Debug use: Try text prompt
        # prompt = "Beautiful picture of a wave breaking"  # @param
        # negative_prompt = "zoomed in, blurry, oversaturated, warped"  # @param
        # # Encode the prompt
        # text_embeddings = pipe._encode_prompt(prompt, 'cuda', 1, True, negative_prompt)

        # print(f"audio_embed shape: {audio_embed.shape}")
        # Pass to UNet
        return self.unet(latent_image, timestep, encoder_hidden_states=audio_embed_input)


'''
============================ Inference Function =============================
'''
def generate_image_from_text(pipe):
    '''
    This is a standard inference method for generating images from text prompts.
    Directly taken from the official documentation.

    Link: https://huggingface.co/learn/diffusion-course/en/unit3/2
    '''

    device = "cuda"
    guidance_scale = 7.5  # @param
    num_inference_steps = 35  # @param
    prompt = "A car hanging on a bridge"  # @param
    negative_prompt = None  # @param

    # Encode the prompt
    text_embeddings = pipe._encode_prompt(prompt, device, 1, True, negative_prompt)
    print(text_embeddings)
    print(text_embeddings.shape)
    # Create our random starting point
    latents = torch.randn((1, 4, 32, 32), device=device, generator=GENERATOR)
    latents *= pipe.scheduler.init_noise_sigma

    # Prepare the scheduler
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)

    # Loop through the sampling timesteps
    for i, t in enumerate(pipe.scheduler.timesteps):

        # Expand the latents if we are doing classifier free guidance
        latent_model_input = torch.cat([latents] * 2)
        print(latent_model_input.shape)
        # Apply any scaling required by the scheduler
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)

        # Predict the noise residual with the UNet
        with torch.no_grad():
            noise_pred = pipe.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

        # Perform guidance
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # Compute the previous noisy sample x_t -> x_t-1
        latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample

    # Decode the resulting latents into an image
    with torch.no_grad():
        image = pipe.decode_latents(latents.detach())

    # View
    image = pipe.numpy_to_pil(image)[0]
    image.show()


# For inference
def generate_image_from_audio(
        audio_embedding:torch.Tensor, 
        conditional_unet:AudioConditionalUNet, 
        vae, 
        scheduler, 
        num_steps=50
    ):


    # Initialize random latent vector
    # latent_dim = 768
    # image_latents = torch.randn(1, latent_dim, 256, 256).to("cuda")
    ################## Fix suggested by GPT #######################
    latent_dim = (1, 4, 32, 32)  # Match UNet in_channels
    image_latents = torch.randn(latent_dim, device='cuda', generator=GENERATOR)
    image_latents *= scheduler.init_noise_sigma
    ###############################################################
    audio_embedding = audio_embedding.to("cuda")

    image_latents = image_latents.to("cuda")

    scheduler.set_timesteps(num_steps, device="cuda")

    # Iterative denoising
    for i, t in tqdm(enumerate(scheduler.timesteps)):

        image_latents_input = torch.cat([image_latents] * 2)
        image_latents_input = scheduler.scale_model_input(image_latents_input, t)
        
        with torch.no_grad(), autocast("cuda",):
            noise_pred = conditional_unet(image_latents_input, timestep=t, audio_input=audio_embedding).sample
            
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + 8 * (noise_pred_text - noise_pred_uncond)

        image_latents = scheduler.step(noise_pred, t, image_latents).prev_sample

    # Decode the final latent vector
    with torch.no_grad(), autocast("cuda",):
        latents = 1 / 0.18215 * image_latents.detach()
        images = vae.decode(latents).sample
        images = (images / 2 + 0.5).clamp(0, 1)

        images = images.cpu().permute(0, 2, 3, 1).float().numpy()

    images = (images * 255).round().astype("uint8")
    # Save the image 
    for i, image in enumerate(images):
        pil_images = Image.fromarray(image)
        pil_images.save(f'sample_image_{i}.jpg')


if __name__ == "__main__":
    model_id = "stabilityai/stable-diffusion-2-1-base"

    # Sample Code
    scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
    pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler)
    pipe = pipe.to("cuda")

    # prompt = "a photo of an astronaut riding a horse on mars"
    # image = pipe(prompt).images[0]  
        
    # image.save("astronaut_rides_horse.png")
    # exit()

    print("Model loaded")

    config = {
        ########### batch size restricted to be 16 in forward pass??? #################
        'batch size': 32,
        'train ratio': 0.9,
        'validation ratio': 0.1,
        'device': 'cuda',
        'epochs': 5000,
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
        {"params": conditional_unet.audio_conditioning.parameters(), "lr": config['linear_lr']},
        {"params": conditional_unet.unet.parameters(), "lr": config['unet_lr']}
    ])
    criterion = nn.MSELoss()

    # Everything about gradient
    vae.requires_grad_(False)
    conditional_unet.requires_grad_(True)

    # Scaler for mixed precision
    scaler = torch.amp.GradScaler("cuda")
    conditional_unet.train()
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

            torch.nn.utils.clip_grad_norm_(conditional_unet.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            optimizer.zero_grad()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_dataloader)}")

    # Save the model
    torch.save(conditional_unet.state_dict(), "diffusion_models/conditional_unet.pth")
    torch.save(vae.state_dict(), "diffusion_models/vae.pth")

    # Example usage - Load one audio embedding from dataset

    audio_embedding = val.dataset.audio_data[0].unsqueeze(0)
    generate_image_from_audio(audio_embedding, conditional_unet, vae, scheduler)
