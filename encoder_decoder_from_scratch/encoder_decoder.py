import torch
import torch.nn as nn

## step 1 make embedddings

class InputEmbeddings(nn.Module):
    def __init__(self, embd_dim, vocab_size):
        super().__init__()
        self.embd_dim = embd_dim # embedding size is d_model in the paper
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embd_dim)
        self.embedding.weight.data.uniform_(-1, 1)


    def forward(self, x):
        return self.embedding(x) * torch.sqrt(self.embd_dim) # bcause it is done in the paper



## step 2 add position embeddings


class PositionEncoding(nn.Module):

    def __init__(self, embd_dim, seq_len, dropout: float) -> None:
        super().__init__()
        self.embd_dim = embd_dim
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout) # to avoid overfitting 
        pe = torch.zeros(seq_len, embd_dim) # position encoding matrix
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1) # dimension (seq_len,1)
        
        div_term = torch.exp(torch.arange(0, embd_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / embd_dim)) # picks every even value
        # dimension pos(seq,1), div_term= div_term(embd_dim/2,) becuase of arrane(0, embd,2)
        # pos(seq_len,1)*div_term(embd_dim/2)-->dim(seq_len,embd/2)-- each seq_len[x] gets multiplied by each embd_dim/2[y]
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # add one more dimension for batch--> dim(1,seq_len,embd_dim)
        self.register_buffer('pe', pe)# keep it as not a learninable parameter and we can save it
    
    def forward(self,x):
        x=x+(self.pe[:, :x.shape[1], :]).requires_grad_(False)### why do we have till sequence lentgh s.shape[1] 
        return self.dropout(x)



## step 3 layer normalization

    # take mean and variance of each batch j --> cap_x_j= (xj-meanj)/root(varj+eps)
    # gamma and beta are learnable parameters --> 
    # gamma is multiplicative to each x and beta is additive to each x-- network tune these two parameters to introduce fluctionations
    # eps is a small number to avoid division by zero
class LayerNormalization(nn.Module):
    def __init__(self, eps: float = 10**-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1)) # multiplied
        self.bias = nn.Parameter(torch.zeros(1)) # added # what does parameter layer do in torch

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True) # -1 so we take mean across each dimension except batch ? how --> mean across last dimension
        std = x.std(dim=-1, keepdim=True) ## print krke dekh lena
        return self.alpha * (x-mean) / (std + self.eps) + self.bias 
    


#### step 4 position wise feed forward layer
# dff=?


class FeedForwardLayer(nn.Module):
    def __init__(self, embd_dim: int, dff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(embd_dim, dff) # W1 and b1--> dff is in the paper
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(dff, embd_dim) # W2 and b2

    def forward(self, x):
        # (batch, seq_len, d_model) --> (batch, seq_len, dff) --> (batch, seq_len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x)))) 
    
#### step 5: multi head attention

## seq= sequence length
## d_model= size of embedding vector
## h= number of heads
## dk=dv=embd_dim/h   
class MultiHeadAttention(nn.Module):
    def __init__(self, embd_dim: int, h: int, dropout:float)->None:
        super().__init__()
        self.embd_dim = embd_dim
        self.h = h
        self.d_k = embd_dim // h # d_k= d_v
        self.w_q = nn.Linear(embd_dim, embd_dim) # Wq
        self.w_k = nn.Linear(embd_dim, embd_dim) # Wk
        self.w_v = nn.Linear(embd_dim, embd_dim) # Wv
        self.w_o = nn.Linear(embd_dim, embd_dim) # Wo --> matrix that is dmodel(h*dv ) x dmodel
        self.dropout = nn.Dropout(dropout)


    @staticmethod
    def attention(q,k,v,mask,dropout: nn.Dropout):
        d_k=q.shape[-1]
        # (batch,h,seq_leb,d_k)-> (batch,h,seq_len,seq_len)


        attention_scores = (q @ k.transpose(-2, -1)) / torch.sqrt(d_k) # (batch, h, seq_len, d_k) @ (batch, h, d_k, seq_len) --> (batch, h, seq_len, seq_len)
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim=-1) # (batch, h, seq_len, seq_len)
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        return (attention_scores @ v), attention_scores # (batch, h, seq_len, seq_len) @ (batch, h, seq_len, d_v) --> (batch, h, seq_len, d_v

    def forward(self,q,k,v):
        # (batch, seq_len, d_model) --> (batch, seq_len, h, d_k) --> (batch, seq_len, h, d_k) --> (batch , seq_len, d_model)
        q = self.w_q(q) # (batch, seq_len, d_model) --> (batch, seq_len,    d_model)
        k = self.w_k(k) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        v = self.w_v(v) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        # (batch, seq_len, h, d_k) matmul (batch, seq_len, h, d_k).transpose(2, 3) --> (batch, seq_len, h, d_k) matmul (batch, seq_len, d_k, h) --> (batch, seq_len, h, d_k)
        q=q.view(q.shape[0], q.shape[1], self.h, self.d_k).transpose(1, 2) # why do we transpose->
        k=k.view(k.shape[0], k.shape[1], self.h, self.d_k).transpose(1, 2)
        v=v.view(v.shape[0], v.shape[1], self.h, self.d_k).transpose(1, 2)
        x,self.attention_scores = MultiHeadAttention.attention(q, k, v, None, self.dropout)

        # (batch,h,seq_len,d_k) --> (batch,seq_len,h,d_k) --> (batch,seq_len,d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k) # video 41:40

        # (batch,seq_len,d_model)--> (batch,seq_len,d_model)
        return self.w_o(x)
    
class ResidualConnection(nn.Module):
    def __init__(self, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderLayer(nn.Module):
    def __init__(self, self_attention_block: MultiHeadAttention, feed_forward_block: FeedForwardLayer, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)]) # search modulelist, range is 2 because 2 times normalization is used in the paper

    def forward(self, x, src_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask)) ## querry key value 
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x

class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)





class DecoderLayer(nn.Module):
    def __init__(self, self_attention_block: MultiHeadAttention, cross_attention_block: MultiHeadAttention, feed_forward_block: FeedForwardLayer, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)]) # decode has three residual connection

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x
    
class Decoder(nn.Module):
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)

class ProjectionLayer(nn.Module):
    def __init__(self, d_model, vocab_size) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # (batch, seq_len, d_model) --> (batch, seq_len, vocab_size)
        return torch.log_softmax(self.proj(x), dim=-1) 



class Transformer(nn.Module):

    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbeddings, tgt_embed: InputEmbeddings, src_pos: PositionEncoding, tgt_pos: PositionEncoding, projection_layer: ProjectionLayer) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)

    def decode(self, tgt, encoder_output, src_mask, tgt_mask):
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)

    def project(self, x):
        return self.projection_layer(x)


def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, d_model: int=512, N: int=6, h: int=8, dropout: float=0.1, d_ff: int=2048):
    # Create embedding layers
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)

    # Create position encoding layers
    src_pos = PositionEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionEncoding(d_model, tgt_seq_len, dropout)



    # create encoder layer
    encoder_layers=[]
    for _ in range(N):
        encoder_self_attention_block=MultiHeadAttention(d_model, h, dropout)
        feed_forward_block=FeedForwardLayer(d_model, d_ff, dropout)
        encoder_block=EncoderLayer(encoder_self_attention_block, feed_forward_block, dropout)
        encoder_layers.append(encoder_block)

    # create the decoder blocks
    decoder_layers=[]
    for _ in range(N):
        decoder_self_attention_block=MultiHeadAttention(d_model, h, dropout)
        decoder_cross_attention_block=MultiHeadAttention(d_model, h, dropout)
        feed_forward_block=FeedForwardLayer(d_model, d_ff, dropout)
        decoder_block=DecoderLayer(decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
        decoder_layers.append(decoder_block)


    # Create encoder and decoder
    encoder = Encoder(nn.ModuleList(encoder_layers))
    decoder = Decoder(nn.ModuleList(decoder_layers))
    # Create projection layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

    # Create the transformer
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)

    # Initialize the parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer 

