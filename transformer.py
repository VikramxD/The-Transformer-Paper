import torch.nn as nn
from encoder import Encoder,EncoderBlock
from decoder import Decoder,LinearLayer,DecoderBlock
from embedding import InputEmbedding,PositionalEncoding
from  multiheadattention import MultiheadAttention
from feedforward import FeedForwardBlock





class Transformer(nn.Module):
    """
    A Transformer model that consists of an encoder and a decoder.

    Args:
        encoder (Encoder): The encoder module.
        decoder (Decoder): The decoder module.
        src_embed (InputEmbedding): The source input embedding module.
        tgt_embed (InputEmbedding): The target input embedding module.
        src_pos (PositionalEncoding): The source positional encoding module.
        tgt_pos (PositionalEncoding): The target positional encoding module.
        linear_layer (LinearLayer): The linear layer module.

    Attributes:
        encoder (Encoder): The encoder module.
        decoder (Decoder): The decoder module.
        src_embedding (InputEmbedding): The source input embedding module.
        tgt_embedding (InputEmbedding): The target input embedding module.
        src_pos (PositionalEncoding): The source positional encoding module.
        tgt_pos (PositionalEncoding): The target positional encoding module.
        linear_layer (LinearLayer): The linear layer module.
    """

    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbedding, tgt_embed: InputEmbedding,
                 src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, linear_layer: LinearLayer):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embedding = src_embed
        self.tgt_embedding = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.linear_layer = linear_layer

    def encode(self, src, src_mask):
        """
        Encodes the source input sequence.

        Args:
            src (Tensor): The source input sequence.
            src_mask (Tensor): The source mask.

        Returns:
            Tensor: The encoded source sequence.
        """
        src = self.src_embedding(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)

    def decode(self, encoder_output, src_mask, tgt, tgt_mask):
        """
        Decodes the target input sequence.

        Args:
            encoder_output (Tensor): The output of the encoder.
            src_mask (Tensor): The source mask.
            tgt (Tensor): The target input sequence.
            tgt_mask (Tensor): The target mask.

        Returns:
            Tensor: The decoded target sequence.
        """
        tgt = self.tgt_embedding(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)

    def linear_step(self, x):
        """
        Applies the linear layer to the input.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor after applying the linear layer.
        """
        return self.linear_layer(x)


def build_model(src_vocab_size:int,src_seq_len:int,tgt_sequence_len:int,tgt_vocab_size:int,embedding_dimension:int=512, n_layer:int=6,num_heads:int=8,dropout=0.1, feed_forward_dim=2048):
    """
    Build a Transformer model.

    Args:
        src_vocab_size (int): The size of the source vocabulary.
        src_seq_len (int): The length of the source sequence.
        tgt_sequence_len (int): The length of the target sequence.
        tgt_vocab_size (int): The size of the target vocabulary.
        embedding_dimension (int, optional): The dimension of the embedding layer. Defaults to 512.
        n_layer (int, optional): The number of encoder and decoder layers. Defaults to 6.
        num_heads (int, optional): The number of attention heads. Defaults to 8.
        dropout (float, optional): The dropout rate. Defaults to 0.1.
        feed_forward_dim (int, optional): The dimension of the feed-forward layer. Defaults to 2048.

    Returns:
        model: The built Transformer model.
    """
    #Build Embeddings
    src_embedding = InputEmbedding(embedding_dimension,src_seq_len,vocab_size=src_vocab_size)
    tgt_embedding = InputEmbedding(embedding_dimension,vocab_size=tgt_vocab_size)
    
    # Build Positional Embeddings
    src_positional_encoding = PositionalEncoding(embedding_dim=embedding_dimension,sequence_len=src_seq_len,dropout=dropout)
    tgt_positional_encoding = PositionalEncoding(embedding_dim=embedding_dimension,sequence_len=tgt_sequence_len,dropout=dropout)
    
    ## Build Encoder Blocks
    encoder_blocks = []
    for _ in range(n_layer):
        encoder_multihead_attention_block = MultiheadAttention(embedding_dim=embedding_dimension,num_heads=num_heads,dropout=dropout)
        feed_forward_block = FeedForwardBlock(embedding_dim=embedding_dimension,feed_forward_dim=feed_forward_dim,dropout=dropout)
        encoder_block = EncoderBlock(encoder_multihead_attention_block,feed_forward_block=feed_forward_block,dropout=dropout)   
        encoder_blocks = encoder_blocks.append(encoder_block)
        
    ## Build Decoder Blocks
    decoder_blocks = []
    for _ in range(n_layer):
        decoder_multihead_attention_block = MultiheadAttention(embedding_dim=embedding_dimension ,num_heads=num_heads, dropout=dropout)
        decoder_cross_attention_block = MultiheadAttention(embedding_dim=embedding_dimension,num_heads=num_heads,dropout=dropout)
        feed_forward_block = FeedForwardBlock(embedding_dim=embedding_dimension,feed_forward_dim=feed_forward_dim,dropout=dropout)
        decoder_block = DecoderBlock(decoder_multihead_attention_block,decoder_cross_attention_block,feed_forward_block,dropout)
        decoder_blocks.append(decoder_block)    
    
    ## Build Encoder,Decoder
    encoder = Encoder(nn.ModuleList(encoder_block))
    decoder = Decoder(nn.ModuleList(decoder_block))
    
    # Linear Layer
    linear_layer = LinearLayer(embedding_dim=embedding_dimension,vocab_size=tgt_vocab_size)
    transformer = Transformer(encoder,decoder,src_embedding,tgt_embedding,src_positional_encoding,tgt_positional_encoding,linear_layer)
    
    for params in transformer.parameters():
        if params.dim() > 1:
            nn.init.xavier_uniform_(params)
            
    return transformer
        
    