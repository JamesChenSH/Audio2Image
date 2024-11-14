import numpy as np
import torch
import torch.utils.data
import tqdm
from skimage.metrics import structural_similarity as ssim

from typing import List

from model_layers import Audio2ImageModel

class Audio2Image():
    
    def __init__(self, 
        dataset:torch.utils.data.Dataset,
        audio_depth:int = 10000, # [src_len, audio_depth]
        # input: [audio_timeline, audio_fourier] -> [img_pixel, 0-255]
        img_depth:int = 256, 
        device:str = 'cpu',
        embedding_dim:int = 1024, 
        encoder_head_num:int = 2, 
        decoder_head_num:int = 2,
        encoder_ff_dim:int = 4*1024, 
        decoder_ff_dim:int = 4*1024,
        encoder_dropout_rate:float = 0.1, 
        decoder_dropout_rate:float = 0.1,
        encoder_attn_dropout:float = 0.0,
        decoder_attn_dropout:float = 0.0, 
        num_enc_layers:int = 6, 
        num_dec_layers:int = 6, 
    ):
        """
        This is the main model for the Audio 2 Image project. We only need to build this once
        in the training script. The hyperparameters are set to default similar to that of GPT-2.
        
        With the defualt settings, there are 176,893,184 parameters in the model with FP32 precision, 
        requiring around 10-12 GB of memory on a GPU with batch size 32.
        
        Parameters:
        audio_depth: int
            Depth of the audio data, range [TBD]
        img_depth: int
            Depth of the image data, range [0, 255]
        dataset: torch.utils.data.Dataset
            Vision and audio dataset in torch format, one to one
            [[audio[0], img[0]], [audio[1], img[1]], ...]
        device: str
            Device to run the model on, either 'cuda' or 'cpu' or 'mps'
        embedding_dim: int
            Dimension of the embedding
        encoder_head_num: int
            Number of Multi-Head-Attention heads in the encoder
        decoder_head_num: int
            Number of MH-Attention heads in the decoder
        encoder_ff_dim: int
            Dimension of the feed forward layer in the encoder
        decoder_ff_dim: int
            Dimension of the ff layer in the decoder
        encoder_dropout_rate: float
            FF layer dropout rate in the encoder
        decoder_dropout_rate: float
            FF layer dropout rate in the decoder
        encoder_attn_dropout: float
            Attention dropout rate in the encoder
        decoder_attn_dropout: float
            Attention dropout rate in the decoder
        num_enc_layers: int
            Number of encoder layers
        num_dec_layers: int
            Number of decoder layers
        """
        self.audio_depth = audio_depth
        self.img_depth = img_depth
        self.dataset = dataset
        self.embedding_dim = embedding_dim
        self.encoder_head_num = encoder_head_num
        self.decoder_head_num = decoder_head_num
        self.encoder_ff_dim = encoder_ff_dim
        self.decoder_ff_dim = decoder_ff_dim
        self.encoder_dropout_rate = encoder_dropout_rate
        self.decoder_dropout_rate = decoder_dropout_rate
        self.encoder_attn_dropout = encoder_attn_dropout
        self.decoder_attn_dropout = decoder_attn_dropout
        self.num_enc_layers = num_enc_layers
        self.num_dec_layers = num_dec_layers
        
        self.device = device
        if device == 'cuda' and torch.cuda.is_available():
            self.device_type = "cuda"
        elif device == "mps":
            self.device_type = "cpu"
        else:
            self.device_type = self.device
        
        self.model = Audio2ImageModel(
            self.audio_depth, 
            self.img_depth,
            self.embedding_dim, 
            self.encoder_head_num, 
            self.decoder_head_num, 
            self.encoder_ff_dim, 
            self.decoder_ff_dim, 
            self.encoder_dropout_rate, 
            self.decoder_dropout_rate, 
            self.encoder_attn_dropout, 
            self.decoder_attn_dropout, 
            self.num_enc_layers, 
            self.num_dec_layers
        )
        
        self.encoder = self.model.encoder
        self.decoder = self.model.decoder
        
        # HyperParameters
        self.label_smoothing = 0.1
        self.learning_rate = 1e-3
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, betas=(0.9, 0.98), eps=1e-9)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', factor=0.5, patience=10)
        self.criterion = torch.nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)       
        self.validation_criterion = torch.nn.SSIM()
        self.epochs = 12
        self.patience = 5
        
        
        
    def train(
        self,
        training_dataloader:torch.utils.data.DataLoader,
        val_dataloader:torch.utils.data.DataLoader,
        batch_size: int = 64,
    ) -> None:
        '''
        Parameters:
        input_audio: np.ndarray
            Input audio data, 3D array of shape (num_samples, audio_size, audio_val)
        output_imgs: np.ndarray
            Output image data, 3D array of shape (num_samples, img_size, img_val)
        '''
        self.model.to(self.device)
        self.criterion.to(self.device)
        
        passed_epochs = 0
        
        for epoch in self.epochs:
            self.model.train()
            
            total_loss = 0
            print(f"== Epoch: {epoch}, Device: {self.device} ==")

            for batch, (audio, img) in enumerate(training_dataloader):
                audio = audio.to(self.device)
                img = img.to(self.device)
                
                self.optimizer.zero_grad()
                # Input a shifted out_image to model as well as input audio
                output = self.model(audio, img[:, :-1])
                # Outputs a predicted image
                loss = self.criterion(output, img[:, 1:])
                self.optimizer.step()
                self.scheduler.step(loss)
                total_loss += loss.item()
                
                loss /= batch_size
                loss.backward()
            
            print(f"== Training Loss: {total_loss}, Device: {self.device}")
            
            self.model.eval()
            
            val_loss = 0
            
            with torch.no_grad():
                for i, (audio, img) in enumerate(val_dataloader):
                    # Compare the predicted image with the actual image with some function
                    gen_img = self.model.generate_image(audio)
                    loss = ssim(gen_img, img)
                    val_loss += loss
            
            print(f"== Validation Loss: {val_loss}, Device: {self.device}")
            

    def test(
        self,
        testing_dataloader:torch.utils.data.DataLoader
    ):
        self.model.to(self.device)
        self.criterion.to(self.device)
        
        self.model.eval()
        
        test_loss = 0
        
        with torch.no_grad():
            for i, (audio, img) in enumerate(testing_dataloader):
                gen_img = self.model.generate_image(audio)
                loss = ssim(gen_img, img)
                test_loss += loss
        
        print(f"Test Loss: {test_loss}, Device: {self.device}")     
        


if __name__ == "__main__":

    model = Audio2Image(dataset=None)
    total_params = sum(p.numel() for p in model.model.parameters())
    print(f"Number of parameters: {total_params}")