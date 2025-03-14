import torch
from torch.utils.data import Dataset

class DummyMedicalDataset(Dataset):
    def __init__(self, num_samples=10, device="cpu"):
        self.num_samples = num_samples
        self.device = device
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # 生成虚拟图像数据（符合您的尺寸要求）
        ct = torch.randn(1, 180, 256, 256).to(self.device)        # [C, D, H, W]
        dose = torch.randn(1, 180, 256, 256).to(self.device)
        struct = torch.randn(9, 180, 256, 256).to(self.device)
        
        # 生成虚拟文本数据（16个指标，每个值0-3）
        text = torch.randint(0, 4, (16,)).to(self.device)
        label = torch.randint(0, 2, (1,)).to(self.device)
        
        return (ct, dose, struct), text, label