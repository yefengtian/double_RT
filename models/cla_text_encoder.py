import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        return x + self.pe[:x.size(0)]

class TextEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, nhead, num_layers, max_seq_length=512, dropout=0.1):
        super().__init__()
        self.max_seq_length = max_seq_length
        self.embed_dim = embed_dim

        # Token embedding
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(embed_dim, max_seq_length)
        
        # Transformer layers
        encoder_layers = TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=4*embed_dim,
            dropout=dropout,
            activation='gelu'
        )
        self.transformer = TransformerEncoder(encoder_layers, num_layers)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(embed_dim)
        
        # Output projection (optional, uncomment if needed)
        # self.output_projection = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, attention_mask=None, normalize=True):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len]
            attention_mask: Tensor, shape [batch_size, seq_len]
            normalize: bool, whether to L2 normalize the output embeddings
        """
        if x.size(1) > self.max_seq_length:
            raise ValueError(f"Input sequence length {x.size(1)} exceeds maximum allowed length {self.max_seq_length}")

        # Embedding
        x = self.embedding(x)  # [batch_size, seq_len, embed_dim]
        
        # Transpose for positional encoding and transformer
        x = x.transpose(0, 1)  # [seq_len, batch_size, embed_dim]
        
        # Add positional encoding
        x = self.positional_encoding(x)
        
        # Create attention mask for transformer if provided
        if attention_mask is not None:
            attention_mask = attention_mask.transpose(0, 1)
            attention_mask = attention_mask.masked_fill(attention_mask == 0, float('-inf'))
        
        # Apply transformer layers
        x = self.transformer(x, src_key_padding_mask=attention_mask)
        
        # Transpose back
        x = x.transpose(0, 1)  # [batch_size, seq_len, embed_dim]
        
        # Apply layer normalization
        x = self.layer_norm(x)
        
        # Global average pooling
        x = x.mean(dim=1)  # [batch_size, embed_dim]
        
        # Optional output projection
        # x = self.output_projection(x)
        
        # Optional L2 normalization
        if normalize:
            x = F.normalize(x, p=2, dim=-1)
        
        return x

    def get_attention_mask(self, input_ids):
        """Generate attention mask."""
        return (input_ids != 0).long()

# Example usage
if __name__ == "__main__":
    vocab_size = 4
    embed_dim = 128
    nhead = 8
    num_layers = 6
    max_seq_length = 20
    batch_size = 2
    seq_length = 16

    model = TextEncoder(vocab_size, embed_dim, nhead, num_layers, max_seq_length)
    
    # Simulate input
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_length))
    attention_mask = model.get_attention_mask(input_ids)

    # Forward pass
    output = model(input_ids, attention_mask)
    print(f"Output shape: {output.shape}")  # Should be [batch_size, embed_dim]