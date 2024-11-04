import numpy as np
import torch
import tqdm

from typing import List

from .layers import Audio2ImageModel

class Audio2Image():
    
    def __init__(self, 
        img_list:List[int],
        device:str = 'cpu',
        img_depth:int = 5, 
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
        spectrum_range:int = 3600, 
    ):
        self.img_list = img_list
        self.img_depth = img_depth
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
        self.spectrum_range = spectrum_range
        
        self.device = device
        if device == 'cuda' and torch.cuda.is_available():
            self.device_type = "cuda"
        elif device == "mps":
            self.device_type = "cpu"
        else:
            self.device_type = self.device
        
        self.model = Audio2ImageModel(
            self.embedding_dim, 
            self.spectrum_range, 
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
        
        def fit(
            self,
            input_imgs:np.ndarray,
            output_specs:np.ndarray,
            output_padding: int = 200,
            batch_size: int = 64,
            batch_size_validation_factor: int = 5,
            # Training Hyper-Parameters
            epochs: int = 32,
            validation_split: float = 0.2,
            checkpoint_freq: int = 0,
            terminate_on_nan: bool = True
        ) -> None:
            optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=10)
            criterion = torch.nn.MSELoss()
            
            self.model.to(self.device)
            criterion.to(self.device)
            
            