import os
import scipy.io.wavfile as wav
import numpy as np
import torch
import matplotlib.pyplot as plt


def perform_fft(data, sample_rate):
    # Perform the FFT
    fft_data = np.fft.fft(data)
    fft_freq = np.fft.fftfreq(len(data), 1 / sample_rate)
    fft_data = np.abs(fft_data)
    fft_data = fft_data[:len(fft_data) // 2]
    fft_freq = fft_freq[:len(fft_freq) // 2]

    # plt.figure()
    # plt.plot(fft_freq, np.real(fft_data))
    # plt.show()
    # exit()

    return fft_data, fft_freq


def save_as_tensor(fft_data_segments, fft_freq, target_path, filename):
    # Convert FFT data and frequencies to PyTorch tensors
    fft_data_tensor = torch.Tensor(fft_data_segments)

    # Define the save path and filename
    tensor_filename = os.path.splitext(filename)[0] + ".pt"
    target_file = os.path.join(target_path, tensor_filename)

    # Save the tensor
    torch.save(fft_data_tensor, target_file)


def process_audio_files(root_dir, target_dir):
    count = 0
    for dirpath, _, filenames in os.walk(root_dir):
        if dirpath == root_dir:
            continue



        current_folder = os.path.basename(dirpath)
        curr_target_dir = os.path.join(target_dir, current_folder)

        # Create the target directory if it doesn't exist
        if not os.path.exists(curr_target_dir):
            os.makedirs(curr_target_dir)
        
        for filename in filenames:
            if filename.endswith(".wav"):
                file_path = os.path.join(dirpath, filename)
                sample_rate, data = wav.read(file_path)

                # split data into 10 pieces
                PIECE_NUM = 50
                segment_length = len(data) // PIECE_NUM
                gap_between_two_segments = int(0.5 * segment_length)
                fft_data_segments = []

                idx = 0
                while idx <= len(data) - segment_length:
                    start = idx
                    end = idx + segment_length
                    data_segment = data[start: end]
                    idx += gap_between_two_segments

                    # Perform FFT on the audio data
                    fft_data, fft_freq = perform_fft(data_segment, sample_rate)
                    fft_data_segments.append(fft_data)

                # Convert to a matrix
                fft_data_matrix = np.array(fft_data_segments)

                # Standardize the matrix (zero mean, unit variance)
                mean_val = fft_data_matrix.mean()
                std_val = fft_data_matrix.std()
                standardized_fft_data_matrix = (fft_data_matrix - mean_val) / std_val

                # Save FFT data and frequencies as a tensor file
                save_as_tensor(np.array(standardized_fft_data_matrix), fft_freq, curr_target_dir, filename)
                
                count += 1
    print(f"Processed {count} audio files.")


def build_joint_sound_tensor(processed_dir, joint_dir):
    count = 0
    tensor_list = []

    # Get and sort all subfolders in the root folder
    subfolders = [f.path for f in os.scandir(processed_dir) if f.is_dir()]
    subfolders.sort()  # Sorting subfolders in lexicographical order

    for subfolder in subfolders:
        print(f"Reading files from: {subfolder}")
        if (subfolder != "data\\audio_processed\\airport"):
            print("skipping")
            continue

        # Get and sort all files in the current subfolder
        files = [f.path for f in os.scandir(subfolder) if f.is_file()]
        files.sort()  # Sorting files in lexicographical order

        # Loop through each file in the current subfolder
        for file in files:
            try:
                curr_tensor = torch.load(file)
                tensor_list.append(curr_tensor)
                count += 1
            except RuntimeError as e:
                print(f"Error stacking tensors: {e}")
            
    joint_tensor_list = torch.stack(tensor_list)  # Adds a new dimension
    print(joint_tensor_list.shape)

    torch.save(joint_tensor_list, joint_dir)

    print("Total tensors loaded:", count)


# Run the function with relative paths
if __name__ == "__main__":
    
    dataset_folder = os.path.join("data")
    
    sound_root_dir = os.path.join(dataset_folder, "sound")
    processed_dir = os.path.join(dataset_folder, "audio_processed")
    process_audio_files(sound_root_dir, processed_dir)

    # process_audio_files(sound_root_dir, processed_dir)
    joint_dir = os.path.join(dataset_folder, "audio_airport_tensor.pt")
    # joint_dir = os.path.join(dataset_folder, "audio_tensor.pt")
    build_joint_sound_tensor(processed_dir, joint_dir)
