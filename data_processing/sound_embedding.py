import torchopenl3
import soundfile as sf
import os
import torch


def build_joint_sound_embed_tensor(processed_dir, joint_dir):
    count = 0
    tensor_list = []

    # Get and sort all subfolders in the root folder
    subfolders = [f.path for f in os.scandir(processed_dir) if f.is_dir()]
    subfolders.sort()  # Sorting subfolders in lexicographical order

    for subfolder in subfolders:
        print(f"Reading files from: {subfolder}")
        if (subfolder != "data\\sound\\airport"):
            print("skipping")
            continue

        # Get and sort all files in the current subfolder
        files = [f.path for f in os.scandir(subfolder) if f.is_file()]
        files.sort()  # Sorting files in lexicographical order

        # Loop through each file in the current subfolder
        for file in files:
            try:
                audio, sr = sf.read(file)
                emb, ts = torchopenl3.get_audio_embedding(audio, sr)
                tensor_list.append(emb.squeeze(dim=0))
                count += 1
            except RuntimeError as e:
                print(f"Error stacking tensors: {e}")
            
            if count == 5: 
                break
            
    joint_tensor_list = torch.stack(tensor_list)  # Adds a new dimension
    print(joint_tensor_list.shape)

    torch.save(joint_tensor_list, joint_dir)

    print("Total tensors loaded:", count)


# Run the function with relative paths
if __name__ == "__main__":
    
    dataset_folder = os.path.join("data") 
    sound_root_dir = os.path.join(dataset_folder, "sound")
    joint_dir = os.path.join(dataset_folder, "audio_airport_tensor_embed.pt")
    build_joint_sound_embed_tensor(sound_root_dir, joint_dir)

    # audio, sr = sf.read('data\\sound\\airport\\00063_1.wav')
    # emb, ts = torchopenl3.get_audio_embedding(audio, sr)

    # print(emb.shape)
    # print(type(emb))

    # print(ts)
    # print(type(ts))