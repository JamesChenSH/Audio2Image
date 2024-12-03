import os
import scipy.io.wavfile as wav
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class AudioImageDataset(Dataset):
    def __init__(self, audio_path, img_path):
        # Load the .pt files as tensors
        self.audio_data = torch.load(audio_path)
        self.img_data = torch.load(img_path).long()

        # Ensure they have the same number of samples
        if len(self.audio_data) != len(self.img_data):
            raise ValueError("Audio and image data must have the same number of samples")

    def __len__(self):
        # Number of samples in the dataset
        return len(self.audio_data)

    def __getitem__(self, idx):
        # Get audio and image data for a specific index
        audio_sample = self.audio_data[idx]
        img_sample = self.img_data[idx]
        return audio_sample, img_sample
    


def build_dataloader(dataset_path, batch_size=32):
    dataset = torch.load(dataset_path)

    # Create a DataLoader to handle batching
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return data_loader



def build_dataset(base_path, audio_set_path='audio_tensor.pt', img_set_path='img_tensor.pt', output_path='Dataset_image_audio.pt'):
    # Create an instance of the custom dataset
    
    joint_audio_path = os.path.join(base_path, audio_set_path)
    joint_img_path = os.path.join(base_path, img_set_path)
    output_path = os.path.join(base_path, output_path)
    
    dataset = AudioImageDataset(joint_audio_path, joint_img_path)

    # Save the dataset directly
    torch.save(dataset, output_path)
    
    print("Saved {} rows to {}".format(len(dataset), output_path))


# Run the function with relative paths
if __name__ == "__main__":
    
    data_path_base = os.path.join("data")
    
    name_audio_ds = "audio_tensor.pt"
    
    name_img_gs_ds = "gs_image_tensor.pt"
    name_img_rgb_ds = "rgb_image_tensor.pt"

    # Save the grayscale version database
    name_audio_gs_dataset = "DS_audio_gs.pt"
    build_dataset(
        base_path=data_path_base, 
        audio_set_path=name_audio_ds, 
        img_set_path=name_img_gs_ds, 
        output_path=name_audio_gs_dataset)

    # Repeat for RGB version if needed
    name_audio_rgb_dataset = "DS_audio_rgb.pt"
    build_dataset(
        base_path=data_path_base,
        audio_set_path=name_audio_ds, 
        img_set_path=name_img_rgb_ds, 
        output_path=name_audio_rgb_dataset)

    # data_loarder_gs = build_dataloader(gs_dataset_path)
    # data_loarder_rgb = build_dataloader(rgb_dataset_path)
