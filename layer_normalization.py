import torch.nn as nn


class LayerNormalization(nn.Module):
    """
    Applies layer normalization to the input tensor.
    
    Args:
        eps (float, optional): A value added to the denominator for numerical stability. Default is 1e-5.
    """
    
    def __init__(self, eps: float = 1e-5) -> None:
        self.eps = eps
        
    def forward(self, x):
        """
        Applies layer normalization to the input tensor.
        
        Args:
            x (torch.Tensor): The input tensor.
        
        Returns:
            torch.Tensor: The normalized tensor.
        """
        return nn.LayerNorm(x, eps=self.eps)

    
    