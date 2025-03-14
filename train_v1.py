import torch
from torch.utils.data import DataLoader
from models.rtNet import MultiModalFusion
from data.dummy_loader import DummyMedicalDataset
import argparse
import torch.nn as nn
from data.data_sets import get_my_data
import os
import torch.nn.functional as F

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def parse_args():
    parser = argparse.ArgumentParser(description="Train 3D ResNet")
    parser.add_argument('--model_depth', type=int, default=18, help='Depth of ResNet (10 | 18 | 34 | 50 | 101 | 152 | 200)')
    parser.add_argument('--sample_input_D', type=int, default=180, help='Depth of input samples')
    parser.add_argument('--sample_input_H', type=int, default=256, help='Height of input samples')
    parser.add_argument('--sample_input_W', type=int, default=256, help='Width of input samples')
    parser.add_argument('--num_classes', type=int, default=2, help='Number of classes')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs')
    parser.add_argument('--vocab_size', type=int, default=4, help='Vocabulary size')
    parser.add_argument('--max_seq_len', type=int, default=20, help='Maximum sequence length')
    parser.add_argument('--data_dir', type=str, default='/data/darry/data/ESO-data', help='Data directory')
    parser.add_argument('--csv_file', type=str, default='/data/darry/data/val_set_add_train_xiugai_3_add10pos_10neg.csv', help='CSV file')
    parser.add_argument('--workers', type=int, default=2, help='Number of workers')

    # 分布式训练参数
    parser.add_argument('--distributed',type=bool,default=False,help='Distributed training')
    parser.add_argument('--local_rank',type=int,default=0,help='Local rank')
    parser.add_argument('--world_size',type=int,default=1,help='World size')
    parser.add_argument('--rank',type=int,default=0,help='Rank')
    parser.add_argument('--backend',type=str,default='nccl',help='Backend')
    parser.add_argument('--init_method',type=str,default='env://',help='Init method')
    parser.add_argument('--port',type=int,default=29500,help='Port')

    return parser.parse_args()

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        pt = torch.clamp(pt, min=1e-7, max=1-1e-7)  # 避免log(0)或log(1)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def train():
    args = parse_args()
    # 配置参数
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = args.num_classes
    batch_size = args.batch_size
    epochs = args.epochs
    
    # 初始化模型
    model = MultiModalFusion(args,num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = FocalLoss(gamma=2, reduction='mean')
    
    # 虚拟数据加载
    # dataset = DummyMedicalDataset(device=device)
    # train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 真实数据加载
    train_dataloader = get_my_data(args,is_train=True)
    val_dataloader = get_my_data(args,is_train=False)
    
    # 训练循环
    for epoch in range(epochs):
        for batch in train_dataloader:

            # 从虚拟数据加载器中获取数据
            # (ct, dose, structure), texts, labels = batch
            # ct = ct.to(device)
            # dose = dose.to(device)
            # struct = struct.to(device)

            # 从真实数据加载器中获取数据
            ct = batch['ct'].to(device=device)
            dose = batch['dose'].to(device=device)
            structure = batch['structure'].to(device=device)
            texts = batch['text'].to(device=device)
            labels = batch['label'].to(device=device)
            
            # 前向传播
            outputs = model(ct, dose, structure, texts)
            loss = criterion(outputs, labels)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

if __name__ == "__main__":
    train()