import torch.nn as nn
from multiheadattention import MultiheadAttention
from feedforward import FeedForwardBlock
from skip_connection import SkipConnection
from layer_normalization import LayerNormalization



class EncoderBlock(nn.Module):
    """
    EncoderBlock class represents a single block in the encoder of a Transformer model.
    
    Args:
        multihead_attention (MultiheadAttention): The multihead attention module.
        feed_forward_block (FeedForwardBlock): The feed forward block module.
        dropout (float): The dropout probability.
    """
    def __init__(self, multihead_attention: MultiheadAttention, feed_forward_block: FeedForwardBlock, dropout: float):
        super().__init__()
        self.multihead_attention = multihead_attention
        self.feed_forward_block = feed_forward_block
        self.skip_connection = SkipConnection(dropout=dropout)
        
    def forward(self, x, src_mask):
        """
        Forward pass of the EncoderBlock.
        
        Args:
            x (torch.Tensor): The input tensor.
            src_mask (torch.Tensor): The source mask tensor.
        
        Returns:
            torch.Tensor: The output tensor.
        """
        x = self.skip_connection(x, lambda x: self.multihead_attention(x, x, x, src_mask))
        x = self.skip_connection(x, self.feed_forward_block)
        return x
    
    
    
class Encoder(nn.Module):
    """
    The Encoder class implements the encoder component of the Transformer model.
    
    Args:
        layers (nn.ModuleList): A list of layers to be applied in the encoder.
    
    Attributes:
        layers (nn.ModuleList): A list of layers to be applied in the encoder.
        layernorm (LayerNormalization): An instance of the LayerNormalization class.
    """
    
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.layernorm = LayerNormalization()
    
    def forward(self, x, mask):
        """
        Forward pass of the encoder.
        
        Args:
            x: The input tensor.
            mask: The mask tensor.
        
        Returns:
            The output tensor after passing through all the layers in the encoder.
        """
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
            


      
        