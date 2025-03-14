import torch
from torch.utils.data import DataLoader
from models.single_img_net import MultiModalFusion
import argparse
import torch.nn as nn
import os
import torch.nn.functional as F
import logging
from datetime import datetime
import json
import numpy as np
from utils.loss import FocalLoss
from data.data_sets import get_my_data
import math
from torch.optim.lr_scheduler import LambdaLR

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def create_experiment_folder(base_dir):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_folder = os.path.join(base_dir, f"experiment_{timestamp}")
    os.makedirs(exp_folder, exist_ok=True)
    return exp_folder

def setup_logging(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "train.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

# 评估函数
def evaluate(model, dataloader, criterion, device):
    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
 
    with torch.no_grad():
        for batch in dataloader:
            # 数据加载适配不同数据格式
            if isinstance(batch, dict):  # 真实数据格式
                ct = batch['ct'].to(device)
                dose = batch['dose'].to(device)
                structure = batch['structure'].to(device)
                # texts = batch['text'].to(device)
                labels = batch['label'].to(device)
            else:  # 虚拟数据格式
                (ct, dose, structure), texts, labels = batch
                ct, dose, structure = ct.to(device), dose.to(device), structure.to(device)
                texts, labels = texts.to(device), labels.to(device)
            
            outputs = model(ct, dose, structure)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item() * labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    
    avg_loss = total_loss / total
    accuracy = correct / total
    return accuracy, avg_loss

def parse_args():
    parser = argparse.ArgumentParser(description="3D多模态分类训练脚本")
    # 数据参数
    parser.add_argument('--data_dir', type=str, default='/data/darry/data/ESO-data')
    parser.add_argument('--csv_file', type=str, default='/data/darry/data/val_set.csv')
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--target_size', type=int, default=(180, 512, 512))
    
    # 模型参数
    parser.add_argument('--model_depth', type=int, default=50, help='ResNetb版本 Depth of ResNet (10 | 18 | 34 | 50 | 101 | 152 | 200)')
    parser.add_argument('--sample_input_D', type=int, default=180, help='Depth of input samples')
    parser.add_argument('--sample_input_H', type=int, default=512, help='Height of input samples')
    parser.add_argument('--sample_input_W', type=int, default=512, help='Width of input samples')
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--vocab_size', type=int, default=4)
    parser.add_argument('--max_seq_len', type=int, default=20)
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--lr', type=float, default=5e-5)       # 初始学习率
    parser.add_argument('--weight_decay', type=float, default=0.1, help='权重衰减系数')
    parser.add_argument('--warmup_epochs', type=int, default=5, help='Warmup阶段epoch数')
    parser.add_argument('--min_lr', type=float, default=1.0e-8, help='最小学习率')
    # parser.add_argument('--resume', type=str, default='/data/darry/proj/double_RT/experiments/experiment_20250314_161750/checkpoints/model_latest_94.pth', help='Path to checkpoint to resume training')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume training')
    parser.add_argument('--val_interval', type=int, default=2,help='Validation interval in epochs')
    parser.add_argument('--save_interval', type=int, default=2,help='保存最新检查点的间隔epoch数')
    parser.add_argument('--early_stop', action='store_true',help='启用早停机制')
    parser.add_argument('--patience', type=int, default=20,help='早停等待epoch数')
    parser.add_argument('--amp', action='store_true',help='启用混合精度训练')
    
    # 系统参数
    # parser.add_argument('--log_dir', type=str, default='./logs')
    # parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    # 修改base_dir参数
    parser.add_argument('--base_dir', type=str, default='./experiments', help='Base directory for all experiments')
   
    
    # 分布式训练参数
    parser.add_argument('--distributed', type=bool, default=False)
    parser.add_argument('--local_rank', type=int, default=0)
    
    return parser.parse_args()

class Trainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 创建实验文件夹
        self.exp_folder = create_experiment_folder(args.base_dir)
        self.args.log_dir = os.path.join(self.exp_folder, "logs")
        self.args.checkpoint_dir = os.path.join(self.exp_folder, "checkpoints")
        
        # 设置日志
        setup_logging(self.args.log_dir)


        # 保存配置信息
        config_save_path = os.path.join(self.exp_folder, "config.json")
        with open(config_save_path, 'w') as f:
            json.dump(vars(self.args), f, indent=2)
        
        # 初始化模型
        self.model = MultiModalFusion(args, args.num_classes).to(self.device)
        # 优化器为AdamW
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=args.lr,
            weight_decay=args.weight_decay
        )
        # 初始化学习率调度器
        self.scheduler = self._create_scheduler(args)

        self.criterion = FocalLoss(gamma=2, reduction='mean')
        
        # 数据加载
        self.train_loader = get_my_data(args, is_train=True)
        self.val_loader = get_my_data(args, is_train=False)
        
        # 训练状态跟踪
        self.best_acc = 0.0
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_acc': []
        }

        # 恢复训练逻辑
        self.start_epoch = 0
        if args.resume:
            self._resume_checkpoint(args.resume)

        # 混合精度训练
        self.scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
        
        # 早停机制
        self.early_stop = args.early_stop
        self.patience = args.patience
        self.best_loss = float('inf')
        self.no_improve_epochs = 0

    def _resume_checkpoint(self, checkpoint_path):
        """从检查点恢复训练状态"""
        logging.info(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.best_loss = checkpoint.get('best_loss', float('inf'))
        
        # 核心参数加载
        self.model.load_state_dict(checkpoint['state_dict'], strict=False)
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        
        # 训练状态恢复
        self.start_epoch = checkpoint['epoch']
        self.best_acc = checkpoint['best_acc']
        self.no_improve_epochs = checkpoint.get('no_improve_epochs', 0)
        
        # 新增组件兼容
        if hasattr(self, 'scheduler') and 'scheduler' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler'])
        
        # 混合精度恢复
        if self.args.amp and 'scaler' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler'])
        
        # 参数合并策略
        saved_args = checkpoint.get('args', {})
        for k, v in saved_args.items():
            if not hasattr(self.args, k):
                setattr(self.args, k, v)
        
        # 设备同步
        if self.args.distributed:
            self.model = DDP(self.model, device_ids=[self.args.local_rank])
        
        logging.info(f"成功恢复检查点:epoch={checkpoint['epoch']},best_acc={checkpoint['best_acc']:.4f}")

    def _create_scheduler(self, args):
        """创建包含warmup和余弦退火的学习率调度器"""
        def lr_lambda(current_epoch):
            # Warmup阶段
            if current_epoch < args.warmup_epochs:
                return (current_epoch + 1) / args.warmup_epochs
            # 余弦退火阶段
            progress = (current_epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)
            return 0.5 * (1 + math.cos(math.pi * progress)) * (1 - args.min_lr / args.lr) + args.min_lr / args.lr

        return LambdaLR(self.optimizer, lr_lambda=lr_lambda)
        # return max(args.min_lr, 0.5 * args.lr * (1 + math.cos(math.pi * progress)))

    def train_epoch(self, epoch):
        self.model.train()
        epoch_loss = 0.0
        
        for batch_idx, batch in enumerate(self.train_loader):
            # 数据加载
            ct = batch['ct'].to(self.device)
            dose = batch['dose'].to(self.device)
            structure = batch['structure'].to(self.device)
            # texts = batch['text'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()

            # 混合精度训练
            with torch.cuda.amp.autocast(enabled=self.args.amp):
                outputs = self.model(ct, dose, structure)
                loss = self.criterion(outputs, labels)
            
            # 反向传播
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            

            # 统计信息
            epoch_loss += loss.item()
            if batch_idx % 2 == 0:
                logging.info(f"Epoch {epoch+1} [{batch_idx}/{len(self.train_loader)}] Loss: {loss.item():.4f}")
                current_lr = self.optimizer.param_groups[0]['lr']
                grad_norm = sum(p.grad.norm()**2 for p in self.model.parameters() if p.grad is not None)**0.5
                if self.args.amp:
                    grad_norm = self.scaler.get_scale() * grad_norm  # 反缩放
                logging.info(f"LR: {current_lr:.2e} | Unscaled Grad Norm: {grad_norm:.4f}")
            
        
        avg_loss = epoch_loss / len(self.train_loader)
        self.history['train_loss'].append(avg_loss)
        return avg_loss

    def run(self):
        # 修改训练循环范围
        for epoch in range(self.start_epoch, self.args.epochs):
            # 训练阶段
            train_loss = self.train_epoch(epoch)
            self.scheduler.step()

            # 保存最新检查点（条件c）
            if (epoch + 1) % self.args.save_interval == 0:
                save_val_acc = self.history['val_acc'][-1] if self.history['val_acc'] else None
                save_val_loss = self.history['val_loss'][-1] if self.history['val_loss'] else None
                self._save_checkpoint(
                    epoch=epoch+1,
                    val_acc=save_val_acc,
                    val_loss=save_val_loss,
                    is_latest=True
                )


            # 验证和保存逻辑
            if (epoch + 1) % self.args.val_interval == 0 or (epoch + 1) == self.args.epochs:
                logging.info("开始验证...")
                val_acc, val_loss = evaluate(self.model, self.val_loader, self.criterion, self.device)
                self.history['val_acc'].append(val_acc)
                self.history['val_loss'].append(val_loss)

                # 记录当前学习率
                current_lr = self.optimizer.param_groups[0]['lr']
                logging.info(f"当前学习率: {current_lr:.2e}")

                # 日志记录
                logging.info(f"Epoch {epoch+1}/{self.args.epochs} | "
                            f"Train Loss: {train_loss:.4f} | "
                            f"Val Loss: {val_loss:.4f} | "
                            f"Val Acc: {val_acc:.4f} | "
                            f"LR: {current_lr:.2e}")
                            
                
                # 保存最佳模型（条件b）
                is_best_acc = val_acc > self.best_acc
                is_best_loss = val_loss < self.best_loss
                
                if is_best_acc or is_best_loss:
                    tag = "acc" if is_best_acc else "loss"
                    self._save_checkpoint(
                        epoch=epoch+1,
                        val_acc=val_acc,
                        val_loss=val_loss,
                        is_best=True,
                        tag=tag
                    )
                    self.best_acc = max(self.best_acc, val_acc)
                    self.best_loss = min(self.best_loss, val_loss)
                    self.no_improve_epochs = 0
                else:
                    self.no_improve_epochs += 1
                
                # 早停判断
                if self.early_stop and self.no_improve_epochs >= self.patience:
                    logging.info(f"早停触发！在{self.patience}个epoch内无改进")
                    # 保存最终状态
                    self._save_checkpoint(
                        epoch=epoch+1,
                        val_acc=val_acc,
                        val_loss=val_loss,
                        is_latest=True
                    )
                    break

        # 保存训练历史
        history_path = os.path.join(self.exp_folder, "training_history.json")
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)

    def _save_checkpoint(self, epoch, val_acc=None, val_loss=None, is_best=False, is_latest=False, tag=""):
        checkpoint = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'best_acc': self.best_acc,
            'best_loss': self.best_loss,
            'no_improve_epochs': self.no_improve_epochs,
            'args': vars(self.args),
            'scaler': self.scaler.state_dict()
        }
        
        filename = f"checkpoint_epoch{epoch}"
        if val_acc is not None:
            filename += f"_acc{val_acc:.4f}"
        if val_loss is not None:
            filename += f"_loss{val_loss:.4f}"
        filename += f"{tag}.pth"
        
        os.makedirs(self.args.checkpoint_dir, exist_ok=True)
        
        # torch.save(checkpoint, os.path.join(self.args.checkpoint_dir, filename))
        
        if is_best:
            best_path = os.path.join(self.args.checkpoint_dir, f"model_best_{tag}_"+filename)
            torch.save(checkpoint, best_path)
            logging.info(f"保存最佳模型：{best_path}")
            
        if is_latest:
            latest_path = os.path.join(self.args.checkpoint_dir, f"model_latest.pth")
            torch.save(checkpoint, latest_path)


if __name__ == "__main__":
    args = parse_args()
    
    # 训练启动
    trainer = Trainer(args)
    logging.info("开始训练...")
    trainer.run()
    logging.info("训练完成！")