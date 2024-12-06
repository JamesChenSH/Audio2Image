import os
os.environ['HF_HOME'] = './cache/'

from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio

import soundfile as sf
import numpy as np

'''
=======================  !!!NOTE!!! ==============================
I am currently stopping here since Open L3 does not have pytorch support
from its original team. Looking for solutions rn.
'''
import openl3

from matplotlib import pyplot as plt


class AudioConditioning(nn.Module):
    def __init__(self, audio_encoder, embedding_dim):
        super(AudioConditioning, self).__init__()
        self.audio_encoder = audio_encoder
        self.fc = nn.Linear(audio_encoder.output_dim, embedding_dim)

    def forward(self, audio_input):
        audio_features = self.audio_encoder(audio_input)
        return self.fc(audio_features)  # Project t


class AudioConditionalUNet(nn.Module):
    def __init__(self, unet, audio_conditioning):
        super(AudioConditionalUNet, self).__init__()
        self.unet = unet
        self.audio_conditioning = audio_conditioning

    def forward(self, latent_image, timestep, audio_input):
        # Generate audio conditioning
        audio_embed = self.audio_conditioning(audio_input)

        # Pass to UNet
        return self.unet(latent_image, timestep, audio_embed)




# For inference
def generate_image_from_audio(audio_file, audio_encoder, conditional_unet, vae, num_steps=50):
    # Load and preprocess audio
    waveform, sample_rate = torchaudio.load(audio_file)
    mel_spec = torchaudio.transforms.MelSpectrogram()(waveform)
    mel_spec_db = torchaudio.transforms.AmplitudeToDB()(mel_spec)

    # Encode audio input
    audio_embedding = audio_encoder(mel_spec_db.unsqueeze(0).cuda())  # Add batch dimension

    # Initialize random latent vector
    latent_dim = 512
    image_latents = torch.randn(1, latent_dim, 64, 64).cuda()

    # Iterative denoising
    for t in reversed(range(num_steps)):
        predicted_noise = conditional_unet(image_latents, timestep=t, audio_input=audio_embedding)
        noise_scale = calculate_noise_scale(t)  # Define noise schedule based on diffusion
        image_latents = image_latents - noise_scale * predicted_noise

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


    config = {
        'batch size': 32,
        'train ratio': 0.8,
        'validation ratio': 0.1,
        'test ratio': 0.1,
        'device': 'cuda',
        'epochs': 300,
        'lr': 1e-6,


        # Model Scale related
        'embedding_dim': 512,
        'encoder_head_num': 8,
        'decoder_head_num': 8,

        'encoder_ff_dim': 4*512,
        'decoder_ff_dim': 4*512,

        'num_encoder_layers': 6,
        'num_decoder_layers': 6
    }

    # Load the dataset
    ds_path = "data/DS_airport.pt"
    ds = torch.load(ds_path, weights_only=False)
    
    # Split Train, Val, Test
    train_size = int(config['train ratio']*len(ds))
    val_size = int(config['validation ratio']*len(ds))
    test_size = len(ds) - train_size - val_size
    
    train, val, test = torch.utils.data.random_split(ds, [train_size, val_size, test_size])
    # train = Subset(train, range(1))
    train_dataloader = torch.utils.data.DataLoader(train, batch_size=config['batch size'], shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val, batch_size=config['batch size'], shuffle=True)    
    test_dataloader = torch.utils.data.DataLoader(test, batch_size=config['batch size'], shuffle=True)



    '''
    All of these are generated by GPT
    '''
    # Define components
    audio_encoder = YourAudioEncoder()  # Open L3 layer here
    audio_conditioning = AudioConditioning(audio_encoder, embedding_dim=768)
    unet = pipe.unet()
    conditional_unet = AudioConditionalUNet(unet, audio_conditioning)
    vae = pipe.vae()

    # Optimizer and loss
    optimizer = optim.AdamW(conditional_unet.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    # Training loop
    for epoch in range(num_epochs):
        for audio, images in dataloader:
            # Preprocess audio and images
            audio_input = preprocess_audio(audio)
            latents = vae.encode(images)

            # Add noise
            noise = torch.randn_like(latents)
            noisy_latents = latents + noise

            # Predict noise with conditional UNet
            predicted_noise = conditional_unet(noisy_latents, timestep, audio_input)

            # Compute loss
            loss = criterion(predicted_noise, noise)

            # Backpropagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}, Loss: {loss.item()}")

    # Example usage
    generate_image_from_audio("audio_file.wav", audio_encoder, conditional_unet, vae)
