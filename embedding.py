"""Contains implementation of the Paper Attention is all you need in pytorch"""

import torch
import torch.nn as nn

class InputEmbedding(nn.Module):
    def __init__(self, embedding_dim: int, vocab_size: int):
        """
        Initializes the InputEmbedding class.

        Args:
            embedding_dim (int): The dimension of the word embeddings.
            vocab_size (int): The size of the vocabulary.

        Returns:
            None
        """
        super(InputEmbedding).__init__()
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
    def forward(self, x):
        """
        Forward pass of the InputEmbedding module.

        Args:
            x: The input tensor.

        Returns:
            The embedded input tensor.
        """
        return self.embedding(x) * torch.sqrt(self.embedding_dim)


class PositionalEncoding(nn.Module):
    """
    Positional Encoding module for Transformer models.

    Args:
        embedding_dim (int): The dimension of the input embeddings.
        sequence_len (int): The length of the input sequence.
        dropout (float): The dropout probability.
    """
    def __init__(self, embedding_dim: int, sequence_len: int, dropout: float):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.sequence_len = sequence_len
        self.dropout = nn.Dropout(dropout)

        # Creating a matrix of size (sequence_len,embedding_dim)
        positional_encoding = torch.zeros(sequence_len, self.embedding_dim)

        # Create a vector of shape (sequence_len,1)
        position = torch.arange(0, sequence_len, dtype=torch.float).unsqueeze(dim=1)
        division_term = torch.exp(torch.arange(0, embedding_dim, 2)).float() * (-torch.log(10000.0) / embedding_dim)

        # Apply the sin formula to the even positions and cosine formula to the odd positions
        positional_encoding[:, 0::2] = torch.sin(position * division_term)
        positional_encoding[:, 1::2] = torch.cos(position * division_term)
        
        positional_encoding = positional_encoding.unsqueeze(dim=0)
        
    def forward(self,x):
        x = x + (self.pe[:,:x.shape[1],:]).requires_grad(False)
        return self.dropout(x)