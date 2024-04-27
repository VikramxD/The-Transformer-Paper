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
        assert embedding_dim%num_heads == 0
        self.dropout = nn.Dropout(dropout)
        self.d_k = embedding_dim//num_heads
        self.w_q = nn.Linear(embedding_dim,embedding_dim)
        self.w_k = nn.Linear(embedding_dim,embedding_dim)
        self.w_v = nn.Linear(embedding_dim,embedding_dim)
        self.w_o = nn.Linear(embedding_dim,embedding_dim)
    
    @staticmethod
    def attention(query,key,value,mask,dropout:nn.Dropout):
        d_k = query.shape[-1]
        attention_scores = (query@key.transpose(-2,-1))/torch.sqrt(d_k)
        return attention_scores
    
        
    def forward(self,q,k,v,mask):
            query =self.w_q(q)
            key = self.w_k(k)
            value = self.w_v(v)
            # (Batch,Sequence_len,embedding_dim)--> (Batch,Sequence_Len,num_heads,d_k) --> (Batch,num_heads,Seq_len,d_k)
            query = query.view(query.shape[0],query.shape[1],self.num_heads).transpose(1,2)
            key = query.view(key.shape[0],key.shape[1],self.num_heads).transpose(1,2)
            query = query.view(value.shape[0],value.shape[1],self.num_heads).transpose(1,2)
            
            x, self.attention_scores = MultiheadAttention.attention(query,key,value,mask,self.dropout)
        
        
