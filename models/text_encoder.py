import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class TextEncoder(nn.Module):
    def __init__(self, vocab_size=4, embed_dim=128, nhead=4, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        encoder_layer = TransformerEncoderLayer(embed_dim, nhead)
        self.transformer = TransformerEncoder(encoder_layer, num_layers)
        self.positional_encoding = 
        
    def forward(self, x):
        # x: [batch_size, seq_len(16)]
        x = self.embedding(x)  # [batch_size, seq_len, embed_dim]
        x = x.transpose(0, 1)  # [seq_len, batch_size, embed_dim]
        x = self.transformer(x)
        x = x.transpose(0, 1)  # [batch_size, seq_len, embed_dim]
        x = nn.layer_norm(x)
        x = x.mean(dim=1)  # Global average pooling
        return F.normalize(x, dim=-1) if normalize else x
