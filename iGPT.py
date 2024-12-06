from transformers import ImageGPTImageProcessor, ImageGPTForCausalImageModeling
import torch
import matplotlib.pyplot as plt
import numpy as np

'''
Some works related to Image GPT and similar stuff:

1. Image PreProcessing: 
    RGB 3-d image value is converted to 512 clusters of 1-D values with k means and k=512

2. Model Sizes:
    Model Name | Layers | Embedding Size | Total Parameters
      iGPT-XL      60         3072            6.8B 
      iGPT-L       48         1536            1.4B  
      iGPT-M       36         1024            455M  
      iGPT-S,      24          512             76M

3. Current Model Structure: 
    -> Word(input) vocab + Positional Encoding
    -> Transformer Decoder

4. Some ideas about Fine-Tune:
    - Preprocess our dataset images to 512 clusters of 1-D values as described 
      in step 1, and build new dataset
    - Substitute the 1st layer (Word Embedding) to LinearLayerEmbedding for 
      audio data to the embedding dimension.
    - Fine tune the model with 3 epochs and learning rate of 1e-6 ish.
      *Note: Before fine tuning, we can fix the Image GPT model's weights, then 
      train the linear embedding for audio and use this as a baseline. 
'''


if __name__ == "__main__":
    processor = ImageGPTImageProcessor.from_pretrained('openai/imagegpt-medium')
    model = ImageGPTForCausalImageModeling.from_pretrained('openai/imagegpt-medium')
    print(model.config.vocab_size) # 512
    exit()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # unconditional generation of 8 images
    batch_size = 8
    context = torch.full((batch_size, 1), model.config.vocab_size - 1) #initialize with SOS token (with ID 512)
    context = torch.tensor(context).to(device)
    output = model.generate(input_ids=context, max_length=model.config.n_positions + 1, temperature=1.0, do_sample=True, top_k=40)

    clusters = np.array(processor.clusters)
    n_px = processor.size["width"]

    samples = output[:,1:].cpu().detach().numpy()
    samples_img = [np.reshape(np.rint(127.5 * (clusters[s] + 1.0)), [n_px, n_px, 3]).astype(np.uint8) for s in samples] # convert color cluster tokens back to pixels
    f, axes = plt.subplots(1, batch_size, dpi=300)
    for i, (img, ax) in enumerate(zip(samples_img, axes)):
        ax.axis('off')
        ax.imshow(img)
        plt.savefig(f"output_{i}.png")