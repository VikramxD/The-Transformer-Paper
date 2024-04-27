from layer_normalization import LayerNormalization
import torch.nn as nn


class SkipConnection(nn.Module):
    """
    A class representing a skip connection in a neural network.

    Args:
        dropout (float): The dropout probability.
    """

    def __init__(self, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self, x, sublayer):
        """
        Perform a forward pass through the skip connection.

        Args:
            x: The input tensor.
            sublayer: The sublayer to apply to the input tensor.

        Returns:
            The output tensor after applying the skip connection.
        """
        return x + self.dropout(sublayer(self.norm(x)))
