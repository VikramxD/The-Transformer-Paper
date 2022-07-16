import numpy as np 
import tensorflow as tf

def tf_batch_matmul(a, b):
    
    a_row, a_col = a.shape[1], a.shape[2]
    b_row, b_col = b.shape[1], b.shape[2]
    
    tiled_a = tf.reshape(tf.tile(a, [1, b_col, 1]), shape=[-1, b_col, a_row, a_col])
    tiled_b = tf.reshape(tf.tile(b, [1, 1, a_row]), shape=[-1, b_row, a_row, b_col])
    
    return tf.reduce_sum(
        tf.transpose(tiled_a, [0, 2, 3, 1]) * tf.transpose(tiled_b, [0, 2, 1, 3]), axis=2
    )


class SelfAttention(tf.keras.Model):
    def __init__(self,d_model,output_size ,dropout = 0.3):
        super(self,SelfAttention).__init__()
        self.query = tf.nn.Linear(d_model ,output_size)
        self.key = tf.nn.Linear(d_model,output_size)
        self.value = tf.nn.Linear(d_model,output_size)
        self.dropout = self.nn.Dropout(dropout)

    def forward(self,q,k,v,mask = None):
        bs = q.shape[0]
        tgt_len = q.shape[1]
        seq_len = k.shape[1]
        query = self.query(q)
        key = self.key(k)
        value = self.value(v)

        dim_k = key.size(-1)
        scores = tf_batch_matmul(query,key.transpose(1,2))/np.sqrt(dim_k)
        if mask is not None:
            expanded_mask = mask[: , None, :].expand(bs,tgt_len ,seq_len)
            scores = scores.masked_fill(expanded_mask == 0 , -float("Inf"))

        weights = tf.nn.softmax(scores , dim = -1)
        if mask is not None:
            expanded_mask = mask[: , None, :].expand(bs,tgt_len ,seq_len)

            subsequent_mask  = 1 - np.triu(
                tf.ones((tgt_len , tgt_len),device = mask.device , dtype = tf.uint8),
                diagonal = 1
                )
            subsequent_mask = subsequent_mask[None , :,:].expand(bs,tgt_len ,tgt_len)

            scores = scores.masked_fill(expanded_mask == 0 , -float("Inf"))
            scores = scores.masked_fill(subsequent_mask == 0 , -float("Inf"))
       
        outputs = tf_batch_matmul(weights,value)
        return outputs

    

class MultiheadedAttention(tf.keras.Model):
    def __init__(self,d_model ,num_heads,dropout):
        super(self,MultiheadedAttention).__init__()
        self.d_model = d_model 
        self.num_heads = num_heads
        self.dropout = dropout 
        self.attn_output_size = self.d_model // self.num_heads
        
        self.attentions = [SelfAttention(d_model ,self.attn_output_size)
        for _ in range(self.num_heads)
        ]
        self.output = tf.nn.Linear(self.d_model,self.d_model)

    def call(self,q,k,v,mask):
        x = tf.concat(
            [layer(q,k,v,mask) for layer in self.attentions

            ],dim = -1
        )
        x = self.output(x)
        return x



class EncoderLayer(tf.keras.Model):
    def __init__(self,d_model ,num_heads,d_ff= 2048,dropout = 0.3):
        super(EncoderLayer,self).__init__()
        self.attention = MultiheadedAttention(d_model ,num_heads ,dropout = dropout)
    
        self.simple_Neural = tf.keras.models.Sequential(
               [  tf.layers.Linear(d_model,d_ff),
                 tf.nn.relu(inplace = True),
                 tf.nn.Dropout(dropout),
                 tf.layers.Linear(d_ff,d_model),
                 tf.nn.Dropout(dropout),
               ]
        )
        
        self.attention_norm = tf.keras.layers.Normalization(d_model)
        self.simple_neural_norm = tf.keras.layers.Normalization(d_model)

    def call(self,src,src_mask):
        x = src
        x = x + self.attention(q = x,k = x ,v = x ,mask = src_mask)
        x = self.attention_norm(x)
        x = x + self.simple_neural_norm(x)
        x = self.simple_neural_norm(x)
        return x
    

class DecoderLayer(tf.keras.Model):
    def __init__(self,d_model ,num_heads,d_ff = 2048 , dropout = 0.3):
        super().__init__()


        self.masked_attention = MultiheadedAttention(d_model , num_heads ,dropout = dropout )
        
        
        self.attention = MultiheadedAttention(d_model,num_heads,dropout = dropout)
       
       
        self.simple_neural = tf.keras.Model.Sequentail([
            tf.nn.Linear(d_model ,d_ff),
            tf.nn.relu(inplace = True),
            tf.nn.Dropout(dropout),
            tf.nn.Dropout(dropout)
        ])
         
        self.masked_attention_norm = tf.keras.layers.LayerNormalization(d_model)
        self.attention_norm = tf.keras.layers.LayerNormalization(d_model)
        self.simple_neural_norm = tf.keras.layers.LayerNormalization(d_model)

        

        


    def call(self,tgt ,enc ,tgt_mask ,enc_mask):
        x = tgt 
        x = x + self.masked_attention(q = x , k = x , v =x, mask = tgt_mask)
        x = self.masked_attention_norm(x)
        x = x + self.attention(q =x ,k = enc ,v = enc , mask = enc_mask)
        x = self.attention_norm(x)
        x = x + self.simple_neural(x)
        x = x + self.simple_neural_norm(x)

       
        return x
       
       

    
    



class Encoder(tf.keras.Model):
    def __init__(self,d_model,num_heads,num_encoders):
        super(Encoder,self).__init__()
        self.enc_layers = [EncoderLayer(d_model,num_heads) for _ in range(num_encoders)]


    def call(self,src,src_mask):
        output = src
        for layer in self.enc_layers:
            output = layer(output,src_mask)
        return output






class Decoder(tf.keras.Model):
    def __init__(self,d_model,num_heads,num_decoders):
        super(Decoder,self).__init__()
        self.dec_layers = [DecoderLayer(d_model,num_heads) for _ in range(num_decoders)],
    def call(self,tgt,enc,tgt_mask ,enc_mask):
        
        output =tgt 
        for layer in self.dec_layers:
            output = layer(output,enc ,tgt_mask,enc_mask)
        return output







class Transformer(tf.keras.Model):

    def __init__(self, d_model = 512, num_heads = 8, num_encoders = 6 ,num_decoders = 6):
        super(Transformer,self).__init__()
        self.encoders = Encoder(d_model , num_heads , num_encoders)
        self.decoders = Decoder(d_model ,num_heads,num_decoders)
    
    def call(self,tgt,src,src_mask,tgt_mask):
        encoder_output = self.encoder(src,src_mask)
        decoder_output = self.decoders(tgt ,encoder_output,src_mask,tgt_mask)
        return decoder_output


