import os
import sys
import scipy.io.wavfile as wav
import numpy as np
import torch
import matplotlib.pyplot as plt
import imagebind as ib

# sys.path.append('../')
from imagebind.models.imagebind_model import imagebind_huge



def process_audio_files(root_dir, target_dir, model):
    count = 0
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    subfolders = [f.path for f in os.scandir(root_dir) if f.is_dir()]
    subfolders.sort()  # Sorting subfolders in lexicographical order

    for subfolder in subfolders:
        # Pre-processing
        print(f'Embedding {subfolder} ...')

        # Get and sort all files in the current subfolder
        files = [f.path for f in os.scandir(subfolder) if f.is_file()]
        files.sort()  # Sorting files in lexicographical order

        # Embedding these sounds
        with torch.no_grad():
            audio_paths=files
            embeddings = model.forward({
                ib.ModalityType.AUDIO: ib.data.load_and_transform_audio_data(audio_paths, device),
            })
            embeddings = embeddings[ib.ModalityType.AUDIO]
            print(f'Embedded Tensor shape: {embeddings.shape}')
        
        # Save the embedded files in this subfolder as a tensor:
        category = os.path.basename(subfolder)
        target_tensor_file = os.path.join(target_dir, category) + ".pt"
        torch.save(embeddings, target_tensor_file)
        
        # pos-processing
        count += len(files)
    print(f"Processed {count} audio files.")


def load_pretrained_image_bind_model():
    # Load the pre-trained ImageBind model
    model = imagebind_huge(pretrained=True)
    model.eval()  # Set to evaluation mode
    return model


if __name__ == "__main__":
    dataset_folder = os.path.join("data")
    
    sound_root_dir = os.path.join(dataset_folder, "sound")
    embedded_dir = os.path.join(dataset_folder, "audio_embedded_by_image_bind")
    model = load_pretrained_image_bind_model()
    process_audio_files(sound_root_dir, embedded_dir, model)

    # joint_dir = os.path.join(dataset_folder, "audio_airport_tensor.pt")
    # joint_dir = os.path.join(dataset_folder, "audio_tensor.pt")
    # build_joint_sound_tensor(processed_dir, joint_dir)

    