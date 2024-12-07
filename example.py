from main import Audio2Image
from data_processing.build_database import AudioImageDataset
import torchvision.transforms as transforms

import torch, random
import numpy as np
import PIL

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)


def tensor_to_gs_image(tensor: torch.Tensor):
    '''
    Takes in tensor with value from 0-255
    returns an image
    '''
    tensor = tensor.to(torch.uint8).numpy()
    if np.ndim(tensor)>3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    tensor = np.reshape(tensor, (32, 32))
    print(tensor.shape)
    return PIL.Image.fromarray(tensor)



if __name__ == "__main__":
    
    # Load the dataset
    ds_path = "data/DS_airport.pt"
    ds = torch.load(ds_path)

    a2i_model = Audio2Image()
    a2i_model.device = 'cuda'

    train_size = int(0.8*len(ds))
    val_size = len(ds) - train_size
    
    train, val = torch.utils.data.random_split(ds, [train_size, val_size])

    # checkpoint = torch.load("model/checkpoint_last_epoch_199_loss5.97459.pt")
    checkpoint = torch.load("model/model_dim_512_layer_enc_6_dec_6/model_bs_32_lr_0.1_epoch_1500.pt", weights_only=False)
    
    a2i_model.model.load_state_dict(checkpoint)
    
    print(a2i_model.device)
    # Generate an image from the dataset
    audio_data = ds.audio_data.to('cuda')
    img_data = ds.img_data.to('cuda')
    
    tensor_to_gs_image(img_data[0].cpu()).show()
    
    # for i in range(len(ds)):
        # image = a2i_model.model.generate_image(audio_data[i].unsqueeze(0), process_bar=True, sampling=True)
    
        # # display the image
        # img_png = tensor_to_gs_image(image.cpu())
        # # img_png.show()
        # img_png.save(f"./output_images/airport/output{i}.png")
    
    