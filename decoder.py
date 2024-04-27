import torch
import torch.nn as nn
from multiheadattention import MultiheadAttention
from feedforward import FeedForwardBlock
from skip_connection import SkipConnection
from layer_normalization import LayerNormalization


class DecoderBlock(nn.Module):
    """
    Decoder block of the Transformer model.
    
    Args:
        multihead_attention (MultiheadAttention): The multihead attention block.
        cross_attention_block (MultiheadAttention): The cross attention block.
        feed_forward_block (FeedForwardBlock): The feed forward block.
        dropout (float): The dropout rate.
    """
    
    def __init__(self, multihead_attention: MultiheadAttention, cross_attention_block: MultiheadAttention,
                 feed_forward_block: FeedForwardBlock, dropout: float):
        super().__init__()
        self.multihead_attention = multihead_attention
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.skip_connection = SkipConnection(dropout=dropout)
    
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        """
        Forward pass of the DecoderBlock.
        
        Args:
            x (Tensor): The input tensor.
            encoder_output (Tensor): The output tensor from the encoder.
            src_mask (Tensor): The mask for the source sequence.
            tgt_mask (Tensor): The mask for the target sequence.
        
        Returns:
            Tensor: The output tensor after applying self-attention and cross-attention.
        """
        # Apply self-attention
        x = self.skip_connection(x, lambda x: self.multihead_attention(x, x, x, tgt_mask))
        # Apply cross-attention
        x = self.skip_connection(x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        return x
    
class Decoder(nn.Module):
    """
    The Decoder module of the Transformer model.
    
    Args:
        layers (nn.ModuleList): List of decoder layers.
        
    Attributes:
        layers (nn.ModuleList): List of decoder layers.
        layernorm (LayerNormalization): Layer normalization module.
    """
    
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.layernorm = LayerNormalization()
        
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        """
        Forward pass of the Decoder module.
        
        Args:
            x: Input tensor.
            encoder_output: Output tensor from the encoder.
            src_mask: Mask for the source sequence.
            tgt_mask: Mask for the target sequence.
            
        Returns:
            Tensor: Output tensor after passing through the decoder layers and layer normalization.
        """
        for layer in self.layers:
            # Apply each decoder block
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.layernorm(x)

class LinearLayer(nn.Module):
    
    def __init__(self,embedding_dim,vocab_size):
        super().__init__()
        self.linearlayer = nn.Linear(embedding_dim,vocab_size)
        
    def forward(self,x):
        return torch.log_softmax(self.proj(x),dim=-1)   