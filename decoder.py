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
    
  
    def forward(self,x,encoder_output,src_mask,tgt_mask):
        x = self.skip_connection(x,lambda x: self.multihead_attention(x,x,x,tgt_mask))
        x = self.skip_connection(x,lambda x: self.cross_attention_block(x,encoder_output,encoder_output,src_mask))
        return x
    
class Decoder(nn.Module):
    
    def __init__(self,layers:nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.layernorm = LayerNormalization()
        
    def forward(self,x,encoder_output,src_mask,tgt_mask):
        for layer in self.layers:
            x = layer(x)