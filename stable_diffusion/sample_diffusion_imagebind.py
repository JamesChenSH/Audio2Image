import os, sys, argparse
os.environ['HF_HOME'] = '../cache/'

from diffusers import EulerDiscreteScheduler, StableUnCLIPImg2ImgPipeline
from diffusers import UNet2DConditionModel
from matplotlib import pyplot as plt

import torch, random
import numpy as np

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
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--fine_tuned", action="store_true")
    parser.add_argument("--prompted", action="store_true")

    args = parser.parse_args()


    # load diffsion model
    model_id = "stabilityai/stable-diffusion-2-1-unclip"
    scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
    pipe = StableUnCLIPImg2ImgPipeline.from_pretrained(model_id, scheduler=scheduler)
    pipe = pipe.to("cuda")

    imagebind_model = ib_model.imagebind_huge(pretrained=True).eval().to('cuda')

    audio_sample_path = ["../data/sound/train station/04789_1.wav"]
    audio_tokenized = ib_data.load_and_transform_audio_data(audio_sample_path, 'cuda')

    # Sample
    # audio_embedding = train.dataset.audio_data[0].unsqueeze(0)
    uncond_embedding = torch.zeros((1, 77, 1024)).to("cuda")
    cond_embedding = torch.zeros((1, 77, 1024)).to("cuda")

    with torch.no_grad():
        image_embedding = imagebind_model.forward({
            ModalityType.AUDIO: audio_tokenized
        })[ModalityType.AUDIO]
    # Free imagebind from GPU to save VRAM
    imagebind_model.to('cpu')
    
    audio_file_name = audio_sample_path[0].split('/')[-1].split('.')[0]
    img_out_path = f"../output_images/new_768_{audio_file_name}_"

    # Load our fine-tuned Unet
    if args.fine_tuned:
        # pipe.unet.load_state_dict(torch.load('../stable_diffusion/imagebind_trained_unet_with_prompt.pth', weights_only=True))
        pipe.unet.load_state_dict(torch.load('../stable_diffusion/imagebind_trained_unet_90_epoch_no_prompt.pth', weights_only=True))
        img_out_path = img_out_path + "fine_tuned_"
    # Check prompted or not
    if args.prompted:
        img_out_path = img_out_path + "prompted.jpg"
        pipe(
            prompt='a satellite image with details on the ground',
            negative_prompt_embeds = uncond_embedding, 
            height=768, width=768, 
            image_embeds=image_embedding
        ).images[0].save(img_out_path)
    else:
        img_out_path = img_out_path + "non_prompted.jpg"
        pipe(
            prompt_embeds=cond_embedding, 
            negative_prompt_embeds=uncond_embedding, 
            height=768, width=768, 
            image_embeds=image_embedding
        ).images[0].save(img_out_path)
    # generate_image_from_audio(audio_tokenized, pipe, imagebind_model, scheduler, 50)
