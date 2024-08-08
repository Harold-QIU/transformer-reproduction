import torch
from torch import nn
import torch.nn.functional as f
import math

class TokenEmbedding(nn.Embedding):
    def __init__(self, num_embeddings, embedding_dim, device):
        super().__init__(num_embeddings, embedding_dim, padding_idx=1, device=device)
        
class PositionalEmbedding(nn.Module):
    def __init__(self, embedding_dim, max_len, device):
        super().__init__()
        
        assert (
            embedding_dim % 2 == 0
        ), "Embedding dimension must be even under this implementation"
        
        self.encoding=torch.zeros(max_len, embedding_dim, device=device)
        self.encoding.requires_grad = False
        pos = torch.arange(0, max_len, device=device)
        pos = pos.float().unsqueeze(dim=1)
        _2i = torch.arange(0, embedding_dim, step=2, device=device).float()
        self.encoding[:, 0::2] = torch.sin(pos/10000**(_2i/embedding_dim))
        self.encoding[:, 1::2] = torch.cos(pos/10000**(_2i/embedding_dim)) # This is why positional embeding should be a even number
        
    def forward(self, x):
        batch_size, seq_len = x.size()
        return self.encoding[:seq_len, :]
    
class TransformerEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, max_len, dropout, device):
        super().__init__()
        self.token_embs = TokenEmbedding(num_embeddings, embedding_dim, device)
        self.pos_emb = PositionalEmbedding(embedding_dim, max_len, device)
        self.drop_out = nn.Dropout(p=dropout)
    
    def forward(self, x):
        token_embs = self.token_embs(x)
        pos_emb = self.pos_emb(x)
        return self.drop_out(token_embs + pos_emb)

class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, n_head):
        super().__init__()
        self.n_head = n_head      
        self.embeddimg_dim = embedding_dim
        self.model_dim = self.embeddimg_dim // self.n_head
        
        assert (
            self.model_dim * self.n_head == self.embeddimg_dim
        ), "Embedding size needs to be divisible by heads"
        
        self.w_q = nn.Linear(embedding_dim, embedding_dim)
        self.w_k = nn.Linear(embedding_dim, embedding_dim)
        self.w_v = nn.Linear(embedding_dim, embedding_dim)
        self.w_combine = nn.Linear(embedding_dim, embedding_dim)
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, q, k, v, mask=None): # ? Why don't we normalize q, k, v before use it
        batch_size, seq_len, emb_dim = q.shape
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)
        q = q.view(batch_size, seq_len, self.n_head, self.model_dim).permute(0, 2, 1, 3)
        k = k.view(batch_size, seq_len, self.n_head, self.model_dim).permute(0, 2, 1, 3)
        v = v.view(batch_size, seq_len, self.n_head, self.model_dim).permute(0, 2, 1, 3)
        score = q @ k.transpose(2, 3) / math.sqrt(self.model_dim)
        
        if mask is not None:
            score.masked_fill_(mask==0, value=-math.inf)
        
        score = self.softmax(score) @ v
        score = score.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, emb_dim)  # When something goes wrong with permute, try contiguous and see what will happen
        output = self.w_combine(score)
        return output
    
class LayerNorm(nn.Module):
    def __init__(self, embedding_dim, eps=1e-12):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(embedding_dim))
        self.beta = nn.Parameter(torch.zeros(embedding_dim))
        self.eps = eps
    
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim= True)
        x_hat = (x-mean) / torch.sqrt(var+self.eps)
        y = self.gamma * x_hat + self.beta
        return y
    
class PositionwiseFeedForward(nn.Module):
    def __init__(self, embedding_dim, hidden, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(embedding_dim, hidden)
        self.fc2 = nn.Linear(hidden, embedding_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.fc1(x)
        x = f.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
class EncoderLayer(nn.Module):
    def __init__(self, embedding_dim, ffn_hidden, n_head, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(embedding_dim, n_head)
        self.norm1 = LayerNorm(embedding_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.ffn = PositionwiseFeedForward(embedding_dim, ffn_hidden, dropout)
        self.norm2 = LayerNorm(embedding_dim)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        _x = x
        x = self.attention(x, x, x, mask)
        x = self.dropout1(x)
        x = self.norm1(x + _x)
        _x = x
        x = self.ffn(x)
        x = self.dropout2(x)
        x = self.norm2(x + _x)
        return x
    
class Encoder(nn.Module):
    def __init__(self, voc_size, embedding_dim, max_len, n_layers, ffn_hidden, n_head, dropout=0.1, device='cpu'):
        super().__init__()
        self.embedding = TransformerEmbedding(voc_size, embedding_dim, max_len, dropout, device)
        self.layers = nn.ModuleList(
            [
                EncoderLayer(embedding_dim, ffn_hidden, n_head, dropout) for _ in range(n_layers)
            ]
        ).to(device)

    def forward(self, x, mask):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x, mask)
        return x

class DecoderLayer(nn.Module):
    def __init__(self, embedding_dim, ffn_hidden, n_head, dropout):
        super().__init__()
        self.attention1 = MultiHeadAttention(embedding_dim, n_head)
        self.norm1 = LayerNorm(embedding_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.cross_attention = MultiHeadAttention(embedding_dim, n_head)
        self.norm2 = LayerNorm(embedding_dim)
        self.dropout2 = nn.Dropout(dropout)
        self.ffn = PositionwiseFeedForward(embedding_dim, ffn_hidden, dropout)
        self.norm3 = LayerNorm(embedding_dim)
        self.dropout3 = nn.Dropout(dropout)
        
    def forward(self, dec, enc, trg_mask, src_mask): # target mask & source mask
        _x = dec
        x = self.attention1(dec, dec, dec, trg_mask)
        x = self.dropout1(x)
        x = self.norm1(x + _x)
        _x = x
        x = self.cross_attention(x, enc, enc, src_mask)
        x = self.dropout2(x)
        x = self.norm2(x + _x)
        x = self.ffn(x)
        x = self.dropout3(x)
        x = self.norm3(x + _x)
        return x
        
class Decoder(nn.Module):
    def __init__(self, voc_size, embedding_dim, max_len, n_layers, ffn_hidden, n_head, dropout=0.1, device='cpu'):
        super().__init__()
        self.embedding = TransformerEmbedding(voc_size, embedding_dim, max_len, dropout, device)
        self.layers = nn.ModuleList(
            [
                DecoderLayer(embedding_dim, ffn_hidden, n_head, dropout) for _ in range(n_layers)
            ]
        ).to(device)
        self.fc = nn.Linear(embedding_dim, voc_size).to(device)
        
    def forward(self, dec, enc, trg_mask, src_mask):
        dec = self.embedding(dec)
        for layer in self.layers:
            dec = layer(dec, enc, trg_mask, src_mask)
        dec = self.fc(dec)
        return dec