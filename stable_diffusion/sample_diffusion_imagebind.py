import os, sys
os.environ['HF_HOME'] = '../cache/'

from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler, DDIMScheduler, StableDiffusionImg2ImgPipeline, StableUnCLIPImg2ImgPipeline
from diffusers import UNet2DConditionModel
from matplotlib import pyplot as plt
from tqdm import tqdm

import torch, random
import numpy as np

from torch.amp import autocast
from PIL import Image

sys.path.append('../')
from data_processing.build_diffusion_dataset import AudioImageDataset_Diffusion

from imagebind import data as ib_data
from imagebind.models import imagebind_model as ib_model
from imagebind.models.imagebind_model import ModalityType

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)



'''
============================ Inference Function =============================
'''
def generate_image_from_audio(audio_data:torch.Tensor, pipe, imagebind, scheduler, num_steps=50):
    '''
    The function to generate image from audio.

    @param audio_data: The audio data tensor
    @param pipe: The stable diffusion pipeline
    @param imagebind: The imagebind model
    @param scheduler: The scheduler model
    @param num_steps: The number of steps to generate the image

    @return: None
    '''
    device = "cuda"

    # Initialize random latent vector
    latent_dim = (1, 4, 32, 32)  # Match UNet in_channels
    image_latents = torch.randn(latent_dim).to("cuda")
    image_latents *= scheduler.init_noise_sigma
    
    # Use placeholder prompt
    negative_prompt = torch.zeros((1, 77, 1024)).to("cuda")
    conditional_embed = torch.zeros((1, 77, 1024)).to("cuda")
    prompt_embed = torch.cat([negative_prompt, conditional_embed])

    audio_data = audio_data.to("cuda")
    image_latents = image_latents.to("cuda")

    scheduler.set_timesteps(num_steps, device="cuda")

    # Iterative denoising
    for i, t in tqdm(enumerate(scheduler.timesteps)):

        image_latents_input = torch.cat([image_latents] * 2)
        image_latents_input = scheduler.scale_model_input(image_latents_input, t)
        
        with torch.no_grad(), autocast("cuda",):
            audio_embedding = imagebind.forward({
                ModalityType.AUDIO: audio_data
            })[ModalityType.AUDIO]

            added_cond_kwargs = (
                {"image_embeds": audio_embedding}
            )
            noise_pred = pipe.unet(image_latents_input, timestep=t, encoder_hidden_states=prompt_embed, added_cond_kwargs=added_cond_kwargs).sample
            
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + 8 * (noise_pred_text - noise_pred_uncond)

        image_latents = scheduler.step(noise_pred, t, image_latents).prev_sample

    # Decode the final latent vector
    with torch.no_grad(), autocast("cuda",):
        latents = 1 / 0.18215 * image_latents.detach()
        images = pipe.vae.decode(latents).sample
        images = (images / 2 + 0.5).clamp(0, 1)

        images = images.cpu().permute(0, 2, 3, 1).float().numpy()

    images = (images * 255).round().astype("uint8")
    # Save the image 
    for i, image in enumerate(images):
        pil_images = Image.fromarray(image)
        pil_images.save(f'sample_imageBind_{i}.jpg')


if __name__ == "__main__":
    # Load the dataset
    # ds_path = "data/DS_airport_diffusion.pt"
    # ds = torch.load(ds_path, weights_only=False)
    
    # Split Train, Val, Test
    # train_size = int(0.8*len(ds))
    # val_size = len(ds) - train_size
    
    # train, val = torch.utils.data.random_split(ds, [train_size, val_size])
    # train = Subset(train, range(1))

    # load diffsion model
    model_id = "stabilityai/stable-diffusion-2-1-unclip"
    scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
    pipe = StableUnCLIPImg2ImgPipeline.from_pretrained(model_id, scheduler=scheduler)
    pipe = pipe.to("cuda")

    imagebind_model = ib_model.imagebind_huge(pretrained=True).eval().to('cuda')

    audio_sample_path = ["../data/sound/airport/00063_3.wav"]
    audio_tokenized = ib_data.load_and_transform_audio_data(audio_sample_path, 'cuda')

    # Sample
    # audio_embedding = train.dataset.audio_data[0].unsqueeze(0)
    uncond_embedding = torch.zeros((1, 77, 1024)).to("cuda")
    cond_embedding = torch.zeros((1, 77, 1024)).to("cuda")
    image_embedding = imagebind_model.forward({
        ModalityType.AUDIO: audio_tokenized
    })[ModalityType.AUDIO]
    # Free imagebind from GPU to save VRAM
    imagebind_model.detach()
    
    # Load our fine-tuned Unet
    pipe.unet.load_state_dict(torch.load('../stable_diffusion/ib_unet.pth', weights_only=True))
    # image_embedding = torch.zeros_like(image_embedding)
    pipe(prompt_embeds=cond_embedding, negative_prompt_embeds = uncond_embedding, height=256, width=256, image_embeds=image_embedding).images[0].save(f'generated_image.jpg')
    # generate_image_from_audio(audio_tokenized, pipe, imagebind_model, scheduler, 50)
