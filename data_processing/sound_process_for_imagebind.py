import os, sys
import torch

sys.path.append('../')
import imagebind as ib

def process_audio_files(root_dir, target_dir):
    count = 0
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

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
            audio_paths = files
            embeddings = ib.data.load_and_transform_audio_data(audio_paths, device)
            print(f'Embedded Tensor shape: {embeddings.shape}')
        
        # Save the embedded files in this subfolder as a tensor:
        category = os.path.basename(subfolder)
        target_tensor_file = os.path.join(target_dir, category) + ".pt"
        torch.save(embeddings, target_tensor_file)
        
        # pos-processing
        count += len(files)
    print(f"Processed {count} audio files.")


def build_joint_sound_embedded_tensor(embedded_dir, joint_dir):
    tensor_list = []
    files = [f.path for f in os.scandir(embedded_dir) if f.is_file()]
    files.sort()  # Sorting files in lexicographical order

    for file in files:
        try:
            curr_tensor = torch.load(file)
            tensor_list.append(curr_tensor)
        except RuntimeError as e:
            print(f"Error stacking tensors: {e}")

    joint_tensor_list = torch.cat(tensor_list, dim=0)
    print(joint_tensor_list.shape)

    torch.save(joint_tensor_list, joint_dir)
    return None


if __name__ == "__main__":
    dataset_folder = os.path.join("../data")
    
    sound_root_dir = os.path.join(dataset_folder, "sound")
    embedded_dir = os.path.join(dataset_folder, "audio_embedded_by_image_bind")
    process_audio_files(sound_root_dir, embedded_dir)

    joint_dir = os.path.join(dataset_folder, "audio_embedded_joint_tensor.pt")
    build_joint_sound_embedded_tensor(embedded_dir, joint_dir)
