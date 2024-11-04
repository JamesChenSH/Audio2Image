import math
import torch
import torch.nn as nn


############ Helpers ############

# Sinusoidal Positional Encoding
def positional_encoding_sinusoidal(
    model_dim:int, 
    seq_len:int=24*24, 
    temp:int=10000
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

# We Embned the image inputs into 512 dimensions to fit into the model. 
# Then we apply a positional encoding to the image embeddings for the 
# entire sequence using cos sin.
class TransformerEmbedding(nn.Module):
    def __init__(
        self, 
        img_depth:int=5, 
        embedding_dim:int=12
        ):
        super().__init__()
        self.img_dim = img_depth
        self.model_dim = embedding_dim
        self.embeddingLayer = nn.Linear(img_depth, embedding_dim)
        
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
        dropout_rate:float, 
        activation:str="Relu"
        ) -> None:
        super(FeedForwardLayer, self).__init__()
        self.dense1 = nn.Linear(embedding_dim, ff_dim)
        self.dense2 = nn.Linear(ff_dim, embedding_dim)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        if activation == "Relu":
            self.activation = nn.ReLU()
        else:
            raise ValueError("Invalid activation function")
        
    def forward(
        self, 
        x: torch.Tensor
        ):
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
        
        self.selfAttention = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=head_num, dropout=dropout_rate)
        self.attnDropout = nn.Dropout(attn_dropout)
        self.layerNorm1 = nn.LayerNorm(embedding_dim)
        self.layerNorm2 = nn.LayerNorm(embedding_dim)
        self.feedForward = FeedForwardLayer(embedding_dim, ff_dim, dropout_rate)
        
        self.need_weights = need_weights
        self.attention_score = None
    
    
    def forward(
        self, 
        x:torch.Tensor):
        # Post Layer Add&Norm as in original Transformer Paper
        
        # Self Multihead Attention
        # Q,K,V = x for self attention.
        attention_output, attention_score = self.selfAttention(x, x, x, need_weights=self.need_weights)
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
        
        self.selfAttention = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=head_num, dropout=dropout_rate)
        self.selfAttentionDropout = nn.Dropout(attn_dropout)
        self.crossAttention = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=head_num, dropout=dropout_rate)
        self.crossAttentionDropout = nn.Dropout(attn_dropout)
        
        self.layerNorm1 = nn.LayerNorm(embedding_dim)
        self.layerNorm2 = nn.LayerNorm(embedding_dim)
        self.layerNorm3 = nn.LayerNorm(embedding_dim)
        
        self.feedForward = FeedForwardLayer(embedding_dim, ff_dim, dropout_rate)
        
        self.need_weights = need_weights
        self.self_attention_score = None
        self.cross_attention_score = None
    
    def forward(
        self, 
        x:torch.Tensor):
        # Post Layer Add&Norm as in original Transformer Paper
        
        # Self Multihead Attention
        self_attention_output, self_attention_score = self.selfAttention(x, x, x, need_weights=self.need_weights)
        # Layer Normalization
        self_attention_layer_norm_output = self.layerNorm1(self.selfAttentionDropout(self_attention_output) + x)
        # Cross Multihead Attention
        cross_attention_output, cross_attention_score = self.crossAttention(self_attention_layer_norm_output, x, x, need_weights=self.need_weights)
        # Layer Normalization
        cross_attention_layer_norm_output = self.layerNorm2(self.crossAttentionDropout(cross_attention_output) + self_attention_layer_norm_output)
        # Feed Forward
        ff_output = self.feedForward(cross_attention_layer_norm_output)
        # Layer Normalization
        ff_norm = self.layerNorm3(ff_output + cross_attention_layer_norm_output)
        
        # Return the output
        if self.need_weights :
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
        
    def forward(
        self, 
        x:torch.Tensor):
        x = self.layerNorm(x)
        for layer in self.encoderLayers:
            x = layer(x)
        return x

    def get_attention_scores(self):
        # Get the attention scores from the last layer
        return self.layers[-1].get_attention_scores()
    
    
    
# TODO: Decoder
# Need to apply positional encoding to the spectrum embeddings as well
class TransformerDecoder(nn.Module):
    def __init__(
        self, 
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
    
    def forward(
        self, 
        x:torch.Tensor
        ):
        x = self.layerNorm(x)
        for layer in self.layers:
            x = layer(x)
        return x


# Post Decoder Linear Layers
# Linearly map model output to spectrum values
class DecoderLinearLayer(nn.Module):
    def __init__(
        self, 
        embedding_dim:int, 
        spectrum_range:float,
        use_log_softmax:bool=False
        ) -> None:
        super(DecoderLinearLayer, self).__init__()
        self.linear = nn.Linear(embedding_dim, spectrum_range)
        self.use_log_softmax = use_log_softmax
        
    def forward(
        self, 
        x:torch.Tensor
        ):
        # Output vector should point to a value for a certain spectrum wavelength
        if self.use_log_softmax:
            return nn.functional.log_softmax(self.linear(x), dim=-1)
        return self.linear(x)


# TODO: Overall model
class Audio2ImageModel(nn.Module):
    def __init__(
        self, 
        img_depth:int, 
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
        spectrum_range:int, 
        use_log_softmax:bool=False
        ) -> None:
        super(Audio2ImageModel, self).__init__()
        
        self.x_pe = positional_encoding_sinusoidal(embedding_dim)
        self.embeddingLayer = TransformerEmbedding(img_depth, embedding_dim)
        self.encoder = TransformerEncoder(embedding_dim, encoder_head_num, encoder_ff_dim, encoder_dropout_rate, encoder_attn_dropout, num_enc_layers)
        self.decoder = TransformerDecoder(embedding_dim, decoder_head_num, decoder_ff_dim, decoder_dropout_rate, decoder_attn_dropout, num_dec_layers)
        self.linearLayer = DecoderLinearLayer(embedding_dim, spectrum_range, use_log_softmax)
        
    def forward(
        self, 
        x:torch.Tensor
        ):
        # embed the image to vectors
        x = self.embeddingLayer(x)
        # apply positional encoding
        x += self.x_pe[:, :x.size(1)].requires_grad(False)
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.linearLayer(x)
        return x