import torch
import torch.nn as nn
import torch.nn.functional as F
from .model import create_model
from .pr_text_encoder import TextEncoder


class FFN(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, dropout_rate=0.5):
        super(FFN, self).__init__()
        layers = []
        
        # 第一层
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(nn.LayerNorm(hidden_dims[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))
        
        # 中间层
        for i in range(1, len(hidden_dims)):
            layers.append(nn.Linear(hidden_dims[i-1], hidden_dims[i]))
            layers.append(nn.LayerNorm(hidden_dims[i]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
        
        # 最后一层
        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        
        self.ffn = nn.Sequential(*layers)

    def forward(self, x):
        return self.ffn(x)


class MultiModalFusion(nn.Module):
    def __init__(self,args, num_classes):
        super().__init__()
        # 图像模态编码器
        self.ct_encoder = create_model(args)
        self.dose_encoder = create_model(args)
        self.struct_encoder = create_model(args,in_channels=9)
        
        # 融合层
        input_dim = 256*3  # 三个图像分支和一个文本分支
        hidden_dims = [512, 512, 256, 256] # 隐藏层维度
        
        self.fc = FFN(input_dim, hidden_dims, num_classes)
    
    def forward(self, ct, dose, struct):
        ct_feat = self.ct_encoder(ct)
        dose_feat = self.dose_encoder(dose)
        struct_feat = self.struct_encoder(struct)
        
        fused = torch.cat([
            ct_feat, dose_feat, struct_feat
        ], dim=1)
        
        return self.fc(fused)