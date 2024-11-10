import numpy as np
import torch
import torch.utils.data
import tqdm

from typing import List

from .model_layers import Audio2ImageModel

class Audio2Image():
    
    def __init__(self, 
        dataset:torch.utils.data.Dataset,
        audio_depth:int = 5,
        img_depth:int = 3, 
        
        device:str = 'cpu',
        embedding_dim:int = 32, 
        encoder_head_num:int = 2, 
        decoder_head_num:int = 2,
        encoder_ff_dim:int = 128, 
        decoder_ff_dim:int = 128,
        encoder_dropout_rate:float = 0.1, 
        decoder_dropout_rate:float = 0.1,
        encoder_attn_dropout:float = 0.0,
        decoder_attn_dropout:float = 0.0, 
        num_enc_layers:int = 2, 
        num_dec_layers:int = 3, 
    ):
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
        self.epochs = 12
        self.patience = 5
        
        
        
    def train(
        self,
        training_dataloader:torch.utils.data.DataLoader,
        val_dataloader:torch.utils.data.DataLoader,
        output_padding: int = 200,
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
        
        for epoch in self.epochs:
            self.model.train()
            
            total_loss = 0
            print(f"== *Training Epoch: {epoch}, Device: {self.device}* ==")

            for batch, (audio, img) in enumerate(training_dataloader):
                audio = audio.to(self.device)
                img = img.to(self.device)
                
                self.optimizer.zero_grad()
                output = self.model(audio, img[:, :-1])
                loss = self.criterion(output, img[:, 1:])
                self.optimizer.step()
                self.scheduler.step(loss)
                total_loss += loss.item()
                
                loss /= batch_size
                loss.backward()
            
            print(f"== Loss: {total_loss}, Device: {self.device}")
            
            self.model.eval()
            
            val_loss = 0
            
            with torch.no_grad():
                for i, (audio, img) in enumerate(val_dataloader):
                    # TODO
                    pass
            
            print(f"== Loss: {val_loss}, Device: {self.device}")
            
    def test(
        self,
        testing_dataloader:torch.utils.data.DataLoader,
        output_padding: int = 200,
        batch_size: int = 64,
    ):
        self.model.to(self.device)
        self.criterion.to(self.device)
        
        self.model.eval()
        
        test_loss = 0
        
        with torch.no_grad():
            for i, (audio, img) in enumerate(testing_dataloader):
                # TODO
                pass
        
        print(f"Test Loss: {test_loss}, Device: {self.device}")     
        
            
            