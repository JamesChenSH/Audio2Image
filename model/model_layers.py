import math
from typing import Optional
import torch
import torch.nn as nn

from tqdm import tqdm

############ Helpers ############

# Sinusoidal Positional Encoding
def positional_encoding_sinusoidal(
    model_dim:int, 
    seq_len:int=100, 
    temp:int=10000,
    ):
    '''
    Sinosoidal Positional Encoding
    
    seq_len: Length of the sequence
    model_dim: Model dimension
    temp: Temperature scaling
    '''
    # Idea: For each pixel in the image, we have a positional encoding
    # Calculate pe by taking the sin and cos of the x and y position of the pixel
    pe = torch.zeros(seq_len, model_dim)
    position = torch.arange(0, seq_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, model_dim, 2) * -(math.log(temp) / model_dim))
    
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0)


############ Layers ############

# TODO: Embedding

# We Embned the image inputs into 12 dimensions to fit into the model. 
# Then we apply a positional encoding to the image embeddings for the 
# entire sequence using cos sin.
class TransformerLinearEmbedding(nn.Module):
    def __init__(
        self, 
        input_depth:int=5, 
        embedding_dim:int=12
        ):
        super().__init__()
        self.input_depth = input_depth
        self.model_dim = embedding_dim
        self.embeddingLayer = nn.Linear(input_depth, embedding_dim)
        
    def forward(
        self, 
        x: torch.Tensor):
        return self.embeddingLayer(x)


class TransformerValueEmbedding(nn.Module):
    def __init__(
        self, 
        value_range:int=256, 
        embedding_dim:int=12
        ):
        super().__init__()
        self.value_range = value_range
        self.model_dim = embedding_dim
        self.embeddingLayer = nn.Embedding(value_range, embedding_dim)
        
    def forward(
        self, 
        x: torch.Tensor):
        return self.embeddingLayer(x)


# TODO: Transformer Blocks 

class FeedForwardLayer(nn.Module):
    def __init__(
        self, 
        embedding_dim:int, 
        ff_dim:int, 
        dropout_rate:float
        ) -> None:
        super(FeedForwardLayer, self).__init__()
        self.dense1 = nn.Linear(embedding_dim, ff_dim)
        self.dense2 = nn.Linear(ff_dim, embedding_dim)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.activation = nn.ReLU()
        
    def forward(self, x: torch.Tensor):
        x = self.dropout1(self.activation(self.dense1(x)))
        x = self.dropout2(self.dense2(x))
        return x


class EncoderTransformerBlock(nn.Module):
    def __init__(
        self, 
        embedding_dim:int, 
        head_num:int, 
        ff_dim:int, 
        dropout_rate:float, 
        attn_dropout:float, 
        need_weights:bool=False
        ) -> None:
        super(EncoderTransformerBlock, self).__init__()
        
        self.selfAttention = nn.MultiheadAttention(batch_first=True, embed_dim=embedding_dim, num_heads=head_num, dropout=attn_dropout)
        self.attnDropout = nn.Dropout(attn_dropout)
        self.layerNorm1 = nn.LayerNorm(embedding_dim)
        self.layerNorm2 = nn.LayerNorm(embedding_dim)
        self.feedForward = FeedForwardLayer(embedding_dim, ff_dim, dropout_rate)
        
        self.need_weights = need_weights
        self.attention_score = None
    
    
    def forward(self, x:torch.Tensor, src_mask:torch.Tensor=None):
        """ 
        Post Layer Add&Norm as in original Transformer Paper
        
        x: Input tensor [batch_size, src_len, embedding_dim]
        src_mask: Mask for the input tensor [batch_size, src_len]
        
        Return: Output tensor [batch_size, src_len, embedding_dim]
        """
        
        # Self Multihead Attention
        # Q,K,V = x for self attention.
        attention_output, attention_score = self.selfAttention(x, x, x, need_weights=self.need_weights, key_padding_mask=src_mask)
        # Layer Normalization
        attn_layer_norm_output = self.layerNorm1(self.attnDropout(attention_output) + x)
        # Feed Forward
        ff_output = self.feedForward(attn_layer_norm_output)
        # Layer Normalization
        ff_norm = self.layerNorm2(ff_output + attn_layer_norm_output)
        
        if self.need_weights :
            self.attention_score = attention_score
        return ff_norm

    def get_attention_scores(self):
        return self.attention_score
    


# TODO: IN PROGRESS
class DecoderTransformerBlock(nn.Module):
    # Do we need self attention + Cross attention?
    def __init__(
        self, 
        embedding_dim:int, 
        head_num:int, 
        ff_dim:int, 
        dropout_rate:float, 
        attn_dropout:float, 
        need_weights:bool=False
        ) -> None:
        super(DecoderTransformerBlock, self).__init__()
        
        self.selfAttention = nn.MultiheadAttention(batch_first=True, embed_dim=embedding_dim, num_heads=head_num, dropout=dropout_rate)
        self.selfAttentionDropout = nn.Dropout(attn_dropout)
        self.crossAttention = nn.MultiheadAttention(batch_first=True, embed_dim=embedding_dim, num_heads=head_num, dropout=dropout_rate)
        self.crossAttentionDropout = nn.Dropout(attn_dropout)
        
        self.layerNorm1 = nn.LayerNorm(embedding_dim)
        self.layerNorm2 = nn.LayerNorm(embedding_dim)
        self.layerNorm3 = nn.LayerNorm(embedding_dim)
        
        self.feedForward = FeedForwardLayer(embedding_dim, ff_dim, dropout_rate)
        
        self.need_weights = need_weights
        self.self_attention_score = None
        self.cross_attention_score = None
    
    def forward(self, decoder_x:torch.Tensor, decoder_mask: torch.Tensor, encoder_x:torch.Tensor, encoder_mask: torch.Tensor):
        """ 
        Post Layer Add&Norm as in original Transformer Paper 
        
        decoder_x: decoder input sequence [batch_size, tgt_len, embedding_dim]
        encoder_x: encoder input sequence [batch_size, src_len, embedding_dim]
        
        Return: Output tensor [batch_size, tgt_len, embedding_dim]
        
        """
        
        # Self Attention - Use causal mask
        self_attention_output, self_attention_score = self.selfAttention(decoder_x, decoder_x, decoder_x, need_weights=self.need_weights, attn_mask=decoder_mask)
        # Layer Normalization
        self_attention_layer_norm_output = self.layerNorm1(self.selfAttentionDropout(self_attention_output) + decoder_x)
        # Cross Attention - Use padding mask
        cross_attention_output, cross_attention_score = self.crossAttention(decoder_x, encoder_x, encoder_x, need_weights=self.need_weights, key_padding_mask=encoder_mask)
        # Layer Normalization
        cross_attention_layer_norm_output = self.layerNorm2(self.crossAttentionDropout(cross_attention_output) + self_attention_layer_norm_output)
        # Feed Forward
        ff_output = self.feedForward(cross_attention_layer_norm_output)
        # Layer Normalization
        ff_norm = self.layerNorm3(ff_output + cross_attention_layer_norm_output)
        
        # Return the output
        if self.need_weights:
            self.self_attention_score = self_attention_score
            self.cross_attention_score = cross_attention_score
        return ff_norm
    
    def get_attention_scores(self):
        return self.self_attention_score, self.cross_attention_score


 
# TODO: Encoder
# Need to apply positional encoding to the image embeddings
class TransformerEncoder(nn.Module):
    def __init__(
        self, 
        embedding_dim:int, 
        head_num:int, 
        ff_dim:int, 
        dropout_rate:float, 
        attn_dropout:float, 
        num_enc_layers:int
        ) -> None:
        super(TransformerEncoder, self).__init__()
        
        self.layerNorm = nn.LayerNorm(embedding_dim)
        self.layers = nn.ModuleList()
        for i in range(num_enc_layers):
            if i == num_enc_layers - 1:
                self.layers.append(EncoderTransformerBlock(embedding_dim, head_num, ff_dim, dropout_rate, attn_dropout, need_weights=True))
            else:
                self.layers.append(EncoderTransformerBlock(embedding_dim, head_num, ff_dim, dropout_rate, attn_dropout))
        
        
    def forward(self, src_x:torch.Tensor, src_mask:torch.Tensor=None):
        """
        x: Input sequence                        [batch_size, src_len, embedding_dim]
        src_mask: Padding Mask for the input     [batch_size, src_len]
        
        Return : Output tensor                   [batch_size, src_len, embedding_dim]
        """
        
        src_x = self.layerNorm(src_x)
        for layer in self.layers:
            src_x = layer(src_x, src_mask)
        return src_x


    def get_attention_scores(self):
        """ 
        Get the attention scores from the last layer
        """
        return self.layers[-1].get_attention_scores()
    
    
    
# TODO: Decoder
# Need to apply positional encoding to the target embeddings as well
class TransformerDecoder(nn.Module):
    def __init__(
        self, 
        img_depth:int,
        embedding_dim:int, 
        head_num:int, 
        ff_dim:int, 
        dropout_rate:float, 
        attn_dropout:float, 
        num_dec_layers:int
        ) -> None:
        super(TransformerDecoder, self).__init__()
        # Decoder Layers
        self.layerNorm = nn.LayerNorm(embedding_dim)
        self.layers = nn.ModuleList()
        for i in range(num_dec_layers):
            if i == num_dec_layers - 1:
                self.layers.append(DecoderTransformerBlock(embedding_dim, head_num, ff_dim, dropout_rate, attn_dropout, need_weights=True))
            else:
                self.layers.append(DecoderTransformerBlock(embedding_dim, head_num, ff_dim, dropout_rate, attn_dropout))
                
        self.linearLayer = DecoderLinearLayer(embedding_dim, img_depth)
    
    
    def forward(self, decoder_x:torch.Tensor, decoder_mask:torch.Tensor, encoder_x:torch.Tensor, encoder_mask:torch.Tensor):
        """
        decoder_x: Decoder input  [batch_size, tgt_len, embedding_dim]
        encoder_x: Encoder output [batch_size, src_len, embedding_dim]
        
        Return: generated output  [batch_size, tgt_len, img_range]
        """
        decoder_x = self.layerNorm(decoder_x)
        for layer in self.layers:
            decoder_x = layer(decoder_x, decoder_mask, encoder_x, encoder_mask)
        decoder_x = self.linearLayer(decoder_x)
        return decoder_x


# Post Decoder Linear Layers
class DecoderLinearLayer(nn.Module):
    def __init__(
        self, 
        embedding_dim:int, 
        out_dimension:float,
        ) -> None:
        super(DecoderLinearLayer, self).__init__()
        self.linear = nn.Linear(embedding_dim, out_dimension)
        
    def forward(self, x:torch.Tensor):
        return self.linear(x)


# TODO: Overall model
class Audio2ImageModel(nn.Module):
    def __init__(
        self, 
        audio_depth:int, 
        audio_len:int,
        img_depth: int,
        img_len:int,
        embedding_dim:int, 
        encoder_head_num:int, 
        decoder_head_num:int,
        encoder_ff_dim:int, 
        decoder_ff_dim:int,
        encoder_dropout_rate:float, 
        decoder_dropout_rate:float,
        encoder_attn_dropout:float,
        decoder_attn_dropout:float, 
        num_enc_layers:int, 
        num_dec_layers:int, 
        img_special_token:tuple,
        aud_special_tokens:tuple,
        device:str='cuda'
        ) -> None:
        super(Audio2ImageModel, self).__init__()
        
        self.img_depth = img_depth
        self.audio_depth = audio_depth
        self.embedding_dim = embedding_dim  
        self.device = device
        
        self.encoder_head_num = encoder_head_num
        self.decoder_head_num = decoder_head_num
        
        self.img_sos, self.img_eos, self.img_pad = img_special_token
        self.audio_sos, self.aud_eos, self.audio_pad = aud_special_tokens
        
        self.aud_pe = positional_encoding_sinusoidal(embedding_dim, audio_len).to(self.device)
        self.img_pe = positional_encoding_sinusoidal(embedding_dim, img_len).to(self.device)
        
        self.audio_embedding = TransformerLinearEmbedding(audio_depth, embedding_dim)             # [batch_size, aud_len, embedding_dim]
        self.img_embedding = TransformerValueEmbedding(img_depth, embedding_dim)                 # [batch_size, img_len, embedding_dim]
        
        self.encoder = TransformerEncoder(
            embedding_dim, 
            encoder_head_num, 
            encoder_ff_dim, 
            encoder_dropout_rate, 
            encoder_attn_dropout, 
            num_enc_layers)
         
        self.decoder = TransformerDecoder(
            img_depth,
            embedding_dim, 
            decoder_head_num, 
            decoder_ff_dim, 
            decoder_dropout_rate, 
            decoder_attn_dropout, 
            num_dec_layers)
        
    ###############################
    #       Helper Functions      #
    ###############################
    def get_audio_embedding(self, aud):
        """
        Generate embeddings map for audio input
        
        aud: Audio input [batch_size, aud_len, audio_depth]
        
        Return: Audio embeddings [batch_size, aud_len, embedding_dim]
        """
        aud_emb = self.audio_embedding(aud)
        if self.aud_pe is not None:
            aud_emb += self.aud_pe[:, :aud.size(1)]
        return aud_emb
    
    
    def get_img_embedding(self, img):
        """
        Generate embeddings map for image output
        
        img: Image input [batch_size, img_len], 0-258
        
        Return: Image embeddings [batch_size, img_len, embedding_dim]
        """
        img_emb = self.img_embedding(img)
        if self.img_pe is not None:
            img_emb += self.img_pe[:, :img.size(1)]
        return img_emb
        
        
    def generate_padding_mask(self, x:torch.Tensor, use_audio=False):
        """
        Create a Mask where the padding tokens in the input is masked as True
        
        Input: x, [batch_size, seq_len, depth]  if x is audio
                  [batch_size, seq_len]         if x is image
        
        Return: Padding mask [batch_size, seq_len]
        
        """
        padding_token = self.img_pad
        if use_audio:
            padding_token = self.audio_pad
            return torch.all(torch.eq(x, padding_token), dim=-1)
        return torch.eq(x, padding_token)
    
    
    def generate_causal_mask(self, x:torch.Tensor):
        '''
        Create causal mask for decoder output sequence
        
        x: Input tensor [batch_size, tgt_len, embedding_dim]
        
        Return: Causal mask [tgt_len, tgt_len]
    
        
        '''
        return torch.triu(torch.ones(x.size(1), x.size(1)), diagonal=1).bool().to(self.device)

        
    ###############################
    #       Forward Function      #
    ###############################
    def forward(self, input_tokens:torch.Tensor, output_tokens:torch.Tensor):
        '''
        input_tokens: Audio tokens [batch_size, aud_len, audio_depth]
        output_tokens: Image tokens [batch_size, img_len, img_range]
        
        return: Image tokens [batch_size, img_len, img_range]
        '''
        # Create padding mask for src
        src_mask = self.generate_padding_mask(input_tokens, use_audio=True)
        # Create causal mask for tgt
        tgt_mask = self.generate_causal_mask(output_tokens)
        
        # embed the image to vectors
        encoder_x = self.get_audio_embedding(input_tokens)
        # apply positional encoding
        decoder_x = self.get_img_embedding(output_tokens)
        
        encoded_src = self.encoder(encoder_x, src_mask)
        decoded_val = self.decoder(decoder_x, tgt_mask, encoded_src, src_mask)
        
        return decoded_val
    
    
    ###############################
    #     Generation Functions    #
    ###############################
    def sampling_alg(self, decoded_seq, top_p=0.8):
        logits = decoded_seq[:, -1]
        probs = torch.softmax(logits, dim=-1)
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        mask = cumulative_probs > top_p
        if mask.size(0) > 1:
            mask[:, 1:] = mask[:, :-1].clone()
        mask[:, 0] = False  # Always keep the first token
        sorted_probs[mask] = 0
        sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)  # Renormalize probabilities

        # Sample from the filtered distribution
        next_token = torch.multinomial(sorted_probs, num_samples=1)

        # Map sampled token indices back to original vocabulary
        generated_token = sorted_indices.gather(dim=-1, index=next_token)
        return generated_token


    def generate_image(self, input_tokens:torch.Tensor, img_len:int=1024, process_bar:bool=False, sampling=False):
        """
        Generate image from audio input. Use Greedy Decode ATM.
        
        input_tokens: Audio tokens [batch_size, aud_len, audio_depth]
        out_len: Maximum length of the output image (dimensions of image)
        
        Return: Image tokens [batch_size, img_len]        
        """
        
        generation_seq = torch.zeros(input_tokens.size(0), 1, dtype=int).fill_(0).to(self.device)
        src_mask = self.generate_padding_mask(input_tokens, use_audio=True).to(self.device)
        src_x = self.get_audio_embedding(input_tokens).to(self.device)
        encoded_src = self.encoder(src_x, src_mask)
        
        if process_bar:
            iterative = tqdm(range(img_len))
        else:
            iterative = range(img_len)

        for _ in iterative:
            tgt_causal_mask = self.generate_causal_mask(generation_seq).unsqueeze(0).repeat(generation_seq.size(0), 1, 1)
            tgt_padding_mask = self.generate_padding_mask(generation_seq).unsqueeze(1).repeat(1, generation_seq.size(1), 1)
            tgt_mask = tgt_causal_mask | tgt_padding_mask
            
            # print(tgt_mask.shape)
            tgt_mask = tgt_mask.unsqueeze(1).repeat(1, self.decoder_head_num, 1, 1)
            # print(tgt_mask.shape)
            tgt_mask = tgt_mask.view(-1, tgt_mask.size(2), tgt_mask.size(3))        
            # print(tgt_mask.shape)
            tgt_x = self.get_img_embedding(generation_seq)
            
            decoded_seq = self.decoder(tgt_x, tgt_mask, encoded_src, src_mask)
            if sampling == False:
                generated_token = decoded_seq[:, -1].argmax(dim=-1).unsqueeze(1)
            else:
                generated_token = self.sampling_alg(decoded_seq, 0.8)
            generation_seq = torch.cat([generation_seq, generated_token], dim=1)
        
        # TODO: Use softmax to get pixel value of entire image
        img_out = generation_seq[:, 1:]

        return img_out
            
            
