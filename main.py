import torch, os, random, json
import torch.utils.data
import torch.nn as nn
import torch.nn.init as init
from torch.utils.data import Subset
import numpy as np

from argparse import ArgumentParser
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim

from datetime import datetime

from model.model_layers import Audio2ImageModel
from data_processing.build_database import AudioImageDataset

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

class  Audio2Image():
    '''
    The Wrapper class for Audio2Image model. We can define the structure of model here
    
    model input: [audio_timeline, audio_fourier] 
    model output: [img_pixel, 0-255]
    '''
    def __init__(self,
        audio_depth:int = 441, # [src_len, audio_depth]
        audio_len:int = 499,
        img_depth:int = 256,
        img_len:int = 32*32, 
        device:str = 'cuda',                     # 'cuda' or 'cpu' or 'mps'
        embedding_dim:int = 512,                # 1024 for optimal
        encoder_head_num:int = 8,               
        decoder_head_num:int = 8, 
        encoder_ff_dim:int = 4*512,             # 4*1024 for optimal
        decoder_ff_dim:int = 4*512,             # 4*1024 for optimal
        encoder_dropout_rate:float = 0.3, 
        decoder_dropout_rate:float = 0.3,
        encoder_attn_dropout:float = 0.3,
        decoder_attn_dropout:float = 0.3, 
        num_enc_layers:int = 6,                 # 12 for optimal
        num_dec_layers:int = 6,                  # 12 for optimal  
        
        epochs:int = 100,
        patience:int = 5,
        lr:float = 1e-3,
        warmup:int = 300
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
        self.audio_len = audio_len
        self.img_depth = img_depth
        self.img_len = img_len
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
        
        self.epochs = epochs
        self.patience = patience
        
        if device == 'cuda' and torch.cuda.is_available():
            self.device = "cuda"
        elif device == 'mps':
            self.device = "mps"
        else:
            self.device = 'cpu'
        
        
        # Hard Coded Params
        self.img_spec_tokens = (256, 257, 258)
        self.aud_spec_tokens = (
            torch.tile(torch.tensor([0.0]), (1, audio_depth)).to(self.device),  
            torch.tile(torch.tensor([-1.0]), (1, audio_depth)).to(self.device),
            torch.tile(torch.tensor([-2.0]), (1, audio_depth)).to(self.device),
        )
        
    
        self.model = Audio2ImageModel(
            self.audio_depth, 
            self.audio_len,
            self.img_depth,
            self.img_len,
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
            self.num_dec_layers,
            self.img_spec_tokens,
            self.aud_spec_tokens,
            self.device
        ).to(self.device)
        
        print(f"Model created on device: {self.device}")
        
        self.encoder = self.model.encoder
        self.decoder = self.model.decoder
        
        # HyperParameters
        self.label_smoothing = 0.1
        self.learning_rate = lr
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, betas=(0.9, 0.98), eps=1e-9, weight_decay=0.2)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, 
                                lr_lambda=lambda step: self.lr_scheduler(self.embedding_dim, step, warmup=warmup))
        self.criterion = torch.nn.CrossEntropyLoss(label_smoothing=self.label_smoothing, reduction='mean')       
        self.validation_criterion = ssim
        
    def lr_scheduler(self, dim_model: int, step:int, warmup:int):
        if step == 0:
            step = 1
        return (dim_model ** -0.5) * min(step ** -0.5, step * warmup ** -1.5)
        
    def train(
        self,
        training_dataloader:torch.utils.data.DataLoader,
        val_dataloader:torch.utils.data.DataLoader,
        model_dir:str
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
        
        cached_param = None
        lowest_val_loss = float('inf')
        wait_count = 0
        
        for epoch in range(self.epochs):
            self.model.train()
            
            total_loss = 0
            print(f"== Epoch: {epoch}, Device: {self.device} ==")

            for audio, img in tqdm(training_dataloader):
                audio = audio.to(self.device)
                img = img.to(self.device)
                # Input a shifted out_image to model as well as input audio
                output = self.model(audio, img[:, :-1])
                
                # Outputs a predicted image
                loss = self.criterion(output.reshape(-1, output.shape[-1]), img[:, 1:].contiguous().view(-1))
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                total_loss += loss.item()
            
            print(f"== Training Loss: {total_loss / len(train_dataloader)}, Device: {self.device}")
            
            self.model.eval()
            val_loss = 0
            
            with torch.no_grad():
                for i, (audio, img) in enumerate(val_dataloader):
                    # Compare the predicted image with the actual image with some function
                    
                    audio = audio.to(self.device)
                    img = img.to(self.device)
                    
                    gen_img = self.model(audio, img[:, :-1])
                    loss = self.criterion(gen_img.reshape(-1, gen_img.shape[-1]), img[:, 1:].contiguous().view(-1))
                    val_loss += loss.item()
                
                val_loss /= len(val_dataloader)
                
                if val_loss < lowest_val_loss:
                    lowest_val_loss = val_loss
                    wait_count = 0
                    cached_param = self.model.state_dict()
                else:
                    wait_count += 1
                    print(f"Waiting: {wait_count}")
                    if wait_count == self.patience:
                        print("Checkpoint Saved")
                        torch.save(cached_param, f"{model_dir}/checkpoint_epoch_{epoch}_loss_{str(round(lowest_val_loss, 4)).replace('.', '_')}.pt")
            
            print(f"== Validation Loss: {val_loss}, Device: {self.device}")
        torch.save(cached_param, f"{model_dir}/checkpoint_last_epoch_{epoch}_loss{round(val_loss, 5)}.pt")
        print(f"Training Complete")
        return total_loss / len(train_dataloader), val_loss
            

    def test(
        self,
        testing_dataloader:torch.utils.data.DataLoader,
        img_len
    ):
        
        print(f"== Testing Model ==")
        self.model.to(self.device)
        self.criterion.to(self.device)
        
        self.model.eval()
        
        test_loss = 0
        
        with torch.no_grad():
            for audio, img in tqdm(testing_dataloader):
                audio = audio.to(self.device)
                img = img.to(self.device)
                img = img.int()
                gen_img = self.model.generate_image(audio, sampling=True, img_len=img_len)

                gen_img_np = gen_img.detach().cpu().numpy().astype(np.float32)
                img_np = img.detach().cpu().numpy().astype(np.float32) 

                loss = self.validation_criterion(gen_img_np, img_np, data_range=256.0)
                test_loss += loss/test_dataloader.batch_size
        
        print(f"Test Loss: {test_loss}, Device: {self.device}")             



# ========= Helpers ========== #


def learning_rate_finder(model, training_loader, min_lr, max_lr):
    
    def lr_scheduler(dim_model: int, step:int, warmup:int):
        if step == 0:
            step = 1
        return (dim_model ** -0.5) * min(step ** -0.5, step * warmup ** -1.5)
    
    
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1, reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=min_lr, betas=(0.9, 0.98), eps=1e-9)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: lr_scheduler(512, step, warmup=300))
    cur_lr = min_lr
    num_iter = len(training_loader)
    model = model.to("cuda")
    lrs = []
    losses = []
    
    for step, (inputs, targets) in tqdm(enumerate(training_loader)):
        inputs = inputs.to("cuda")
        targets = targets.to("cuda")
        if step >= num_iter:
            break
        
        lr = cur_lr * (max_lr / cur_lr) ** (step / num_iter)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            
        lrs.append(lr)

        output = model(inputs, targets[:, :-1])
        loss = criterion(output.reshape(-1, output.shape[-1]), targets[:, 1:].contiguous().view(-1))
        losses.append(loss.item())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
    
    import matplotlib.pyplot as plt
    plt.plot(lrs, losses)
    print(lrs, losses)
    plt.xscale('log')
    plt.xlabel('LR')
    plt.ylabel('Loss')
    plt.title("Learning Rate Finder")
    plt.savefig('./learning_rates.png')



def initialize_weights(m):
    if isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


if __name__ == "__main__":

    # Argument Parsing
    parser = ArgumentParser()
    parser.add_argument("--find_lr", action="store_true")
    
    args = parser.parse_args()


    config = {
        'batch size': 8,
        'train ratio': 0.8,
        'validation ratio': 0.1,
        'test ratio': 0.1,
        'device': 'cuda',
        'epochs': 300,
        'lr': 0.003,
        'warm_up': 300,
        'patience': 10,


        # Model Scale related
        'aud_dim': 441,         # 441 | 221
        'aud_len': 499,         # 499 | 500
        'img_dim': 256,         
        'img_len': 32*32*3,       # 32*32*3 for RGB

        'embedding_dim': 512,
        'encoder_head_num': 4,
        'decoder_head_num': 8,

        'encoder_ff_dim': 4*512,
        'decoder_ff_dim': 4*512,

        'num_encoder_layers': 0,
        'num_decoder_layers': 24,

        'dropout_ff_enc': 0.3,
        'dropout_ff_dec': 0.3,
        'dropout_attn_enc': 0.3,
        'dropout_attn_dec': 0.3

    }

    a2i_core = Audio2Image(
        audio_depth=config['aud_dim'],
        audio_len=config['aud_len'],
        img_depth=config['img_dim'],
        img_len=config['img_len'],
        embedding_dim=config['embedding_dim'],
        encoder_head_num=config['encoder_head_num'],
        decoder_head_num=config['decoder_head_num'],
        encoder_ff_dim=config['encoder_ff_dim'],
        decoder_ff_dim=config['decoder_ff_dim'],
        num_enc_layers=config['num_encoder_layers'],
        num_dec_layers=config['num_decoder_layers'],
        
        encoder_dropout_rate=config['dropout_attn_enc'],
        decoder_dropout_rate=config['dropout_attn_dec'],
        encoder_attn_dropout=config['dropout_attn_enc'],
        decoder_attn_dropout=config['dropout_attn_dec'],

        device=config['device'], 
        epochs=config['epochs'], 
        lr=config['lr'],
        warmup=config['warm_up'],
        patience=config['patience']
    )

    a2i_core.model.apply(initialize_weights)
    # Try to load previous model
    # a2i_core.model.load_state_dict(torch.load("./model/model_dim_512_layer_enc_6_dec_6/model_bs_32_lr_0.1.pt"))

    # Chack size of model
    total_params = sum(p.numel() for p in a2i_core.model.parameters())
    print(f"Number of parameters: {total_params}")

    # Load the dataset
    ds_path = "data/DS_airport_499.pt"
    ds = torch.load(ds_path, weights_only=False)
    
    # Split Train, Val, Test
    train_size = int(config['train ratio']*len(ds))
    val_size = int(config['validation ratio']*len(ds))
    test_size = len(ds) - train_size - val_size
    
    train, val, test = torch.utils.data.random_split(ds, [train_size, val_size, test_size])
    # train = Subset(train, range(1))
    train_dataloader = torch.utils.data.DataLoader(train, batch_size=config['batch size'], shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val, batch_size=config['batch size'], shuffle=True)    
    test_dataloader = torch.utils.data.DataLoader(test, batch_size=config['batch size'], shuffle=True)
    
    if args.find_lr:
        # Find correct LR for model.
        learning_rate_finder(a2i_core.model, train_dataloader, 1e-5, 1)
        exit()
    
    # # Test code
    # audio_data = ds.audio_data.to(a2i_core.device)
    # print(a2i_core.model.generate_image(audio_data[0].unsqueeze(0)))
    
    run_time = datetime.now()
    model_dir = f"./model/run_{run_time.year}_{run_time.month}_{run_time.day}_{run_time.hour}_{run_time.minute}_{run_time.second}"
    model_json = f"{model_dir}/model_dim_{a2i_core.embedding_dim}_layer_enc_{a2i_core.num_enc_layers}_dec_{a2i_core.num_dec_layers}.json"

    # Create save directory
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir, 0o750)

    # Dump model structure as json
    with open(model_json, 'w') as js:
        json.dump(config, js)

    # Train 
    train_loss_final, val_loss_final = a2i_core.train(train_dataloader, val_dataloader, model_dir)
    # Save the model
    model_path = f"{model_dir}/model_bs_{config['batch size']}_lr_{config['lr']}_epoch_{config['epochs']}_trn_loss_{str(round(train_loss_final, 2)).replace('.', '_')}_val_loss{str(round(val_loss_final, 2)).replace('.','_')}.pt"
    torch.save(a2i_core.model.state_dict(), model_path)
    
    # Test
    a2i_core.test(test_dataloader, config['img_len'])

