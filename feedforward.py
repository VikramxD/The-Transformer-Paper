
import torch.nn as nn

class FeedForwardBlock(nn.Module):
    """
    A feed-forward block in the Transformer model.

    Args:
        embedding_dim (int): The dimensionality of the input embeddings.
        feed_forward_dim (int): The dimensionality of the hidden layer in the feed-forward network.
        dropout (float): The dropout probability.

    Attributes:
        linear_1 (nn.Linear): The first linear layer.
        dropout (nn.Dropout): The dropout layer.
        linear_2 (nn.Linear): The second linear layer.
    """

    def __init__(self, embedding_dim, feed_forward_dim, dropout) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(embedding_dim, feed_forward_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(feed_forward_dim, embedding_dim)
        
    def forward(self, x):
        """
        Forward pass of the feed-forward block.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        x = self.linear_1(x)
        x = nn.ReLU(x)
        x = self.dropout(x)
        x = self.linear_2(x)
        return x

        