"""
Implementation of the Multihead Attention Module Takes 3 copies of the Input Embedding with Positional Encoding Applied and fills it into Query Key Value ,
then splits it along Embedding Dimension and learns parameters Wq,Wv,Wk alongside it a mask is also applied to hide values above the diagnol which are not
necessary in the learning process
"""

import torch
import torch.nn as nn


class MultiheadAttention(nn.Module):
    """
    A multi-head attention module.

    Args:
        embedding_dim (int): The dimensionality of the input embeddings.
        num_heads (int): The number of attention heads.
        dropout (float): The dropout probability.

    Attributes:
        embedding_dim (int): The dimensionality of the input embeddings.
        num_heads (int): The number of attention heads.
    """

    def __init__(self, embedding_dim, num_heads, dropout):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        assert embedding_dim % num_heads == 0
        self.dropout = nn.Dropout(dropout)
        self.d_k = embedding_dim // num_heads
        self.w_q = nn.Linear(embedding_dim, embedding_dim)
        self.w_k = nn.Linear(embedding_dim, embedding_dim)
        self.w_v = nn.Linear(embedding_dim, embedding_dim)
        self.w_o = nn.Linear(embedding_dim, embedding_dim)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        """
        Compute the scaled dot-product attention.

        Args:
            query (torch.Tensor): The query tensor.
            key (torch.Tensor): The key tensor.
            value (torch.Tensor): The value tensor.
            mask (torch.Tensor): The attention mask tensor.
            dropout (nn.Dropout): The dropout layer.

        Returns:
            torch.Tensor: The output tensor after attention.
            torch.Tensor: The attention scores.
        """
        d_k = query.shape[-1]
        attention_scores = (query @ key.transpose(-2, -1)) / torch.sqrt(d_k)
        if mask is not None:
            attention_scores.masked_fill_(mask == 0, 1e-9)
            attention_scores = attention_scores.softmax(dim=-1)
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v, mask):
        """
        Perform forward pass of the multi-head attention module.

        Args:
            q (torch.Tensor): The query tensor.
            k (torch.Tensor): The key tensor.
            v (torch.Tensor): The value tensor.
            mask (torch.Tensor): The attention mask tensor.

        Returns:
            torch.Tensor: The output tensor after attention.
        """
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)
        # (Batch,Sequence_len,embedding_dim)--> (Batch,Sequence_Len,num_heads,d_k) --> (Batch,num_heads,Sequence_len,d_k)
        query = query.view(
            query.shape[0], query.shape[1], self.num_heads, self.d_k
        ).transpose(1, 2)
        key = query.view(
            key.shape[0], key.shape[1], self.num_heads, self.d_k
        ).transpose(1, 2)
        query = query.view(
            value.shape[0], value.shape[1], self.num_heads, self.d_k
        ).transpose(1, 2)

        x, self.attention_scores = MultiheadAttention.attention(
            query, key, value, mask, self.dropout
        )
        # (Batch ,num_heads ,seq_len,d_k) => (Batch,Seq_length,num_heads,d_k)
        x = x.transpose(1, 2).contigous().view(x.shape[0], -1, self.h * self.d_k)
        return self.w_o(x)
