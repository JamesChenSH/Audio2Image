from main import Audio2Image
from data_processing.build_database import AudioImageDataset
import torchvision.transforms as transforms

import torch
import numpy as np
import PIL

def tensor_to_gs_image(tensor):
    tensor = tensor
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor)>3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    tensor = np.reshape(tensor, (32, 32))
    print(tensor.shape)
    return PIL.Image.fromarray(tensor)



if __name__ == "__main__":
    
    # Load the dataset
    ds_path = "data/DS_audio_gs.pt"
    ds = torch.load(ds_path)
    
    
    a2i_model = torch.load('./model/model.pt', map_location='cpu')
    a2i_model.device = 'cpu'
    print(a2i_model.device)
    # Generate an image from the dataset
    audio_data = ds.audio_data.to('cpu')
    image = a2i_model.generate_image(audio_data[1].unsqueeze(0), process_bar=True)
    
    # display the image
    print(image.shape)
    img_png = tensor_to_gs_image(image.cpu())
    img_png.show()