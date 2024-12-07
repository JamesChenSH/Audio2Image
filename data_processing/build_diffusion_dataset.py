import os
import scipy.io.wavfile as wav
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from torch.utils.data import Dataset, Subset
from torchvision import transforms
from PIL import Image

from datasets import load_dataset
    

class AudioImageDataset_Diffusion(Dataset):
    def __init__(self, audio_path, img_folder, subset_path=None):

        # Load the audio tensors. No need for other operation
        self.audio_data = torch.load(audio_path)

        # Prepare image
        self.img_data = []
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

        subfolders = [f.path for f in os.scandir(img_folder) if f.is_dir()]
        subfolders.sort()  # Sorting subfolders in lexicographical order

        for subfolder in subfolders:
            if (subset_path is not None) and (subfolder != subset_path):
                print("skipping")
                continue
            files = [f.path for f in os.scandir(subfolder) if f.is_file()]
            files.sort()  # Sorting files in lexicographical order
            
            for file in files:
                if os.path.basename(file) == ".DS_Store":
                    continue
                self.img_data.append(Image.open(file).convert("RGB"))

        # Ensure they have the same number of samples
        if len(self.audio_data) != len(self.img_data):
            print(len(self.audio_data), len(self.img_data))
            raise ValueError("Audio and image data must have the same number of samples")
        

    def __len__(self):
        # Number of samples in the dataset
        return len(self.audio_data)

    def __getitem__(self, idx):
        # Get audio and image data for a specific index
        audio_sample = self.audio_data[idx]
        if self.transform:
            img_sample = self.transform(self.img_data[idx])
        else:
            img_sample = self.img_data[idx]
        return audio_sample, img_sample
    

def build_dataloader_Diffusion(dataset_path, batch_size=32):
    dataset = torch.load(dataset_path)

    # Create a DataLoader to handle batching
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return data_loader


def build_dataset(base_path, audio_set_path='audio_tensor.pt', img_dir='./data/image_original', output_path='Dataset_image_audio.pt', subset=None):
    # Create an instance of the custom dataset
    
    joint_audio_path = os.path.join(base_path, audio_set_path)
    output_path = os.path.join(base_path, output_path)
    
    subset_path = f"data/image_original\\{subset}" if subset is not None else None

    dataset = AudioImageDataset_Diffusion(joint_audio_path, img_dir, subset_path)

    # Save the dataset directly
    torch.save(dataset, output_path)
    
    print("Saved {} rows to {}".format(len(dataset), output_path))



if __name__ == "__main__":

    data_path_base = os.path.join("data")
    
    name_audio_ds = "audio_train_station_tensor_embed.pt"
    
    build_dataset(data_path_base, name_audio_ds, 'data/image_original', "DS_train_station_diffusion.pt", subset='train_station')