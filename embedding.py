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
        
    def forward(self,x):
        return self.embedding(x)* torch.sqrt(self.embedding_dim)
        
    