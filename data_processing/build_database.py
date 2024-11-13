import os
import scipy.io.wavfile as wav
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class AudioImageDataset(Dataset):
    def __init__(self, audio_path, img_path):
        # Load the .pt files as tensors
        self.audio_data = torch.load(audio_path)
        self.img_data = torch.load(img_path)

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


def build_dataset(joint_audio_path, joint_img_path, ds_save_path):
    # Create an instance of the custom dataset
    dataset = AudioImageDataset(joint_audio_path, joint_img_path)

    # Save the dataset directly
    torch.save(dataset, ds_save_path)


def build_dataloader(dataset_path, batch_size=32):
    dataset = torch.load(dataset_path)

    # Create a DataLoader to handle batching
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return data_loader


# Run the function with relative paths
if __name__ == "__main__":
    joint_audio_dir = os.path.join("..", "data", "audio_tensor.pt")
    joint_gs_img_dir = os.path.join("..", "data", "gs_image_tensor.pt")
    joint_rgb_img_dir = os.path.join("..", "data", "rgb_image_tensor.pt")

    # Save the grayscale version database
    gs_dataset_path = os.path.join("..", "data", "DS_audio_gs.pt")
    # build_dataset(joint_audio_dir, joint_gs_img_dir, gs_dataset_path)

    # Repeat for RGB version if needed
    rgb_dataset_path = os.path.join("..", "data", "DS_audio_rgb.pt")
    # build_dataset(joint_audio_dir, joint_rgb_img_dir, rgb_dataset_path)

    data_loarder_gs = build_dataloader(gs_dataset_path)
    data_loarder_rgb = build_dataloader(rgb_dataset_path)
