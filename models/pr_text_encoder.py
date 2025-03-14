import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math

class PositionalEncoding(nn.Module):
    """实现标准的Transformer位置编码"""
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:, :x.size(1)]
        return x

class TextEncoder(nn.Module):
    """专业级文本编码器，包含完整Transformer结构"""
    def __init__(self, 
                 vocab_size=4, 
                 embed_dim=256,
                 nhead=4,
                 num_layers=2,
                 max_seq_len=16,
                 dropout=0.1,
                 layer_norm_eps=1e-5):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        
        # 词嵌入层
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim, max_seq_len)
        
        # Transformer编码器
        encoder_layer = TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dropout=dropout,
            activation='gelu',
            layer_norm_eps=layer_norm_eps,
            batch_first=False,
            norm_first=True  # 使用Pre-LN结构
        )
        self.transformer = TransformerEncoder(encoder_layer, num_layers)
        
        # 正则化层
        self.layer_norm = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        self.dropout = nn.Dropout(p=dropout)
        
        # 初始化参数
        self._reset_parameters()

    def _reset_parameters(self):
        """专业初始化方法"""
        # 词嵌入初始化
        nn.init.xavier_uniform_(self.embedding.weight)
        # Transformer参数初始化
        for p in self.transformer.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
    def forward(self, x: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        """
        Args:
            x: 输入token indices，形状 [batch_size, seq_len]
            normalize: 是否对输出进行L2归一化
        Returns:
            形状 [batch_size, embed_dim]
        """
        # 输入验证
        if x.size(1) > self.max_seq_len:
            raise ValueError(f"输入序列长度{x.size(1)}超过最大限制{self.max_seq_len}")
            
        # 嵌入层
        x = self.embedding(x)  # [batch_size, seq_len, embed_dim]
        
        # 位置编码
        x = self.pos_encoder(x)
        
        # 维度调整 + dropout
        x = x.transpose(0, 1)  # [seq_len, batch_size, embed_dim]
        x = self.dropout(x)
        
        # Transformer编码
        x = self.transformer(x)
        
        # 全局平均池化
        x = x.transpose(0, 1)  # [batch_size, seq_len, embed_dim]
        x = self.layer_norm(x)
        x = x.mean(dim=1)  # [batch_size, embed_dim]
        
        # 可选L2归一化
        if normalize:
            x = F.normalize(x, p=2, dim=-1)
            
        return x

# 测试用例
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建虚拟输入
    batch_size = 2
    seq_len = 16
    vocab_size = 4
    dummy_input = torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)
    
    # 初始化模型
    text_encoder = TextEncoder(
        vocab_size=vocab_size,
        max_seq_len=seq_len
    ).to(device)
    
    # 前向测试
    output = text_encoder(dummy_input)
    print(f"输入形状：{dummy_input.shape}")
    print(f"输出形状：{output.shape}")
    print(f"输出范数：{torch.norm(output, dim=1)}")