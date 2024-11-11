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


def save_as_tensor(fft_data, fft_freq, target_path, filename):
    # Convert FFT data and frequencies to PyTorch tensors
    fft_data_tensor = torch.Tensor(fft_data)

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

                # Perform FFT on the audio data
                fft_data, fft_freq = perform_fft(data, sample_rate)

                # Save FFT data and frequencies as a tensor file
                save_as_tensor(fft_data, fft_freq, curr_target_dir, filename)
                
                count += 1
    print(f"Processed {count} audio files.")


# Run the function with relative paths
if __name__ == "__main__":
    root_dir = os.path.join("data", "sound")
    target_dir = os.path.join("data", "audio_processed")
    process_audio_files(root_dir, target_dir)
