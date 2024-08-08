import torch
from torch import nn
from utls import Encoder, Decoder

class Transformer(nn.Module):
    def __init__(self,
                 src_pad_idx,
                 trg_pad_idx,
                 enc_voc_size,
                 dec_voc_size,
                 embedding_dim,
                 max_len,
                 n_heads,
                 ffn_hidden,
                 n_layers,
                 dropout,
                 device):
        super().__init__()
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device
        self.encoder = Encoder(
            enc_voc_size,
            embedding_dim,
            max_len,
            n_layers,
            ffn_hidden,
            n_heads,
            dropout,
            device
        )
        self.decoder = Decoder(
            dec_voc_size,
            embedding_dim,
            max_len,
            n_layers,
            ffn_hidden,
            n_heads,
            dropout,
            device
        )
    
    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2).to(self.device)
        return src_mask
    
    def make_trg_mask(self, trg):
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(3)
        trg_len = trg.shape[1]
        trg_sub_mask = torch.tril(torch.ones(trg_len, trg_len)).type(torch.ByteTensor).to(self.device)
        trg_mask = trg_pad_mask & trg_sub_mask
        return trg_mask
    
    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_src_mask(trg)
        enc_src = self.encoder(src, src_mask)
        output = self.decoder(trg, enc_src, trg_mask, src_mask)
        return output

if __name__ == "main":
    device='cuda:0'
    
    transformer = Transformer(
        src_pad_idx=0,
        trg_pad_idx=0,
        enc_voc_size=5,
        dec_voc_size=5,
        embedding_dim=512,
        max_len=8,
        n_heads=8,
        ffn_hidden=256,
        n_layers=3,
        dropout=0.1,
        device='cuda:0'
    )
    
    src = torch.tensor([[1, 2, 3, 4, 2, 3, 1, 1],
                        [2, 3, 4, 1, 0, 0, 0, 0]])

    trg = torch.tensor([[1, 2, 3, 2, 0, 0, 0, 0], 
                        [1, 2, 3, 3, 0, 0, 0, 0]])

    trg = transformer(src.to(device), trg.to(device))
    print(trg)