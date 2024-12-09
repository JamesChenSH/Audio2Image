import os
os.environ['HF_HOME'] = './cache/'

import imagebind as ib
import torch
from diffusers import StableUnCLIPImg2ImgPipeline

# construct models
device = "cuda:0" if torch.cuda.is_available() else "cpu"
pipe = StableUnCLIPImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1-unclip", torch_dtype=torch.float16
).to(device)
model = ib.imagebind_huge(pretrained=True).eval().to(device)

# generate image
with torch.no_grad():
    audio_paths=["/w/284/jameschen/Audio2Image/data/sound/train station/00355_1.wav"]
    embeddings = model.forward({
        ib.ModalityType.AUDIO: ib.load_and_transform_audio_data(audio_paths, device),
    })
    embeddings = embeddings[ib.ModalityType.AUDIO]
    print(embeddings.shape)
    # images = pipe(image_embeds=embeddings.half()).images
    # images[0].save("audio2img.png")