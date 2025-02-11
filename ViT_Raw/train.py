import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from pathlib import Path
from tqdm import tqdm
import yaml
from ViT import VisionTransformer
from data_loader.data_loaders import DefaultDataLoader
import torch.nn.functional as F

class ConfigParser:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
    def __getattr__(self, name):
        if name in self.config:
            return self.config[name]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.multi_gpu = torch.cuda.device_count() > 1
        if self.multi_gpu:
            print(f"Using {torch.cuda.device_count()} GPUs!")
        
        # Initialize best_val_loss
        self.best_val_loss = float('inf')
        
        self.setup_tensorboard()
        self.setup_model()
        self.setup_data()
        self.setup_training()

    def setup_tensorboard(self):
        # Create tensorboard writer
        self.writer = SummaryWriter(Path(self.config.log_dir) / self.config.run_name)
        
        # Log hyperparameters
        hparams = {
            'batch_size': self.config.batch_size,
            'learning_rate': self.config.learning_rate,
            'num_epochs': self.config.num_epochs,
            'embed_dim': self.config.embed_dim,
            'depth': self.config.depth,
            'num_heads': self.config.num_heads,
        }
        self.writer.add_hparams(hparams, {})

    def setup_model(self):
        self.model = VisionTransformer(
            img_size=self.config.img_size,
            patch_size=self.config.patch_size,
            in_channels=self.config.in_channels,
            embed_dim=self.config.embed_dim,
            depth=self.config.depth,
            num_heads=self.config.num_heads,
            mlp_ratio=self.config.mlp_ratio,
            dropout=self.config.dropout
        )
        
        # Move model to device first
        self.model = self.model.to(self.device)
        
        # Wrap model with DataParallel if multiple GPUs are available
        if self.multi_gpu:
            self.model = nn.DataParallel(self.model)

    def setup_data(self):
        # Setup train and validation dataloaders using your custom dataloader
        self.train_loader = DefaultDataLoader(
            data_dir=self.config.train_data_dir,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            training=True,
            previous_days=self.config.previous_days,
            dataset_type=self.config.dataset_type
        )
        
        self.val_loader = DefaultDataLoader(
            data_dir=self.config.val_data_dir,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            training=False,
            previous_days=self.config.previous_days,
            dataset_type=self.config.dataset_type
        )

    def setup_training(self):
        # Use MSE loss for reconstruction with mask
        def masked_mse_loss(output, target, mask):
            target = target.unsqueeze(1)
            # Ensure all tensors have the same size
            if output.shape != target.shape:
                # Resize output to match target size
                output = F.interpolate(output, size=target.shape[-2:], mode='bilinear', align_corners=False)
            if mask.shape[-2:] != target.shape[-2:]:
                # Resize mask to match target size
                mask = F.interpolate(mask, size=target.shape[-2:], mode='nearest')
            
            # Apply mask to both output and target
            loss = F.mse_loss(output * mask, target * mask, reduction='sum')
            # Normalize by the number of valid points in the mask
            valid_points = torch.sum(mask)
            return loss / (valid_points + 1e-8)  # Add small epsilon to avoid division by zero
        
        self.criterion = masked_mse_loss
        
        # Setup optimizer based on config
        if self.config.optimizer.lower() == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=float(self.config.learning_rate),
                betas=(float(self.config.adam_beta1), float(self.config.adam_beta2)),
                eps=float(self.config.adam_epsilon),
                weight_decay=float(self.config.weight_decay)
            )
        elif self.config.optimizer.lower() == 'adamw':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=float(self.config.learning_rate),
                betas=(float(self.config.adam_beta1), float(self.config.adam_beta2)),
                eps=float(self.config.adam_epsilon),
                weight_decay=float(self.config.weight_decay)
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.config.optimizer}")

        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.num_epochs
        )

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        pbar = tqdm(self.train_loader, desc='Training')
        
        for batch_idx, (data, targets) in enumerate(pbar):
            # Get the gulf mask from the last channel of data
            mask = data[:, -1:, :, :].to(self.device)  # Shape: [B, 1, H, W]
            # Remove the mask from input data
            data = data[:, :-1, :, :].to(self.device)
            targets = targets.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(data)
            loss = self.criterion(outputs, targets, mask)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            pbar.set_postfix({
                'loss': total_loss / (batch_idx + 1),
                'gpu_mem': f'{torch.cuda.max_memory_allocated() / 1e9:.2f}GB' if torch.cuda.is_available() else 'N/A'
            })

        return total_loss / len(self.train_loader)

    def validate(self):
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for data, targets in tqdm(self.val_loader, desc='Validating'):
                # Get the gulf mask from the last channel of data
                mask = data[:, -1:, :, :].to(self.device)  # Shape: [B, 1, H, W]
                # Remove the mask from input data
                data = data[:, :-1, :, :].to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.model(data)
                loss = self.criterion(outputs, targets, mask)
                total_loss += loss.item()

        return total_loss / len(self.val_loader)

    def save_checkpoint(self, epoch, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'state_dict': self.model.module.state_dict() if self.multi_gpu else self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss
        }
        
        save_path = Path(self.config.save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save latest checkpoint
        torch.save(checkpoint, save_path / 'latest_checkpoint.pth')
        
        # Save best model
        if is_best:
            torch.save(checkpoint, save_path / 'best_model.pth')

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state
        if self.multi_gpu:
            self.model.module.load_state_dict(checkpoint['state_dict'])
        else:
            self.model.load_state_dict(checkpoint['state_dict'])
            
        # Load optimizer state
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        
        # Load scheduler state if it exists
        if checkpoint['scheduler'] and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler'])
            
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        start_epoch = checkpoint.get('epoch', 0)
        
        return start_epoch

    def train(self):
        start_epoch = 0

        if hasattr(self.config, 'resume_from') and self.config.resume_from:
            print(f"Resuming from checkpoint: {self.config.resume_from}")
            start_epoch = self.load_checkpoint(self.config.resume_from)

        for epoch in range(start_epoch, self.config.num_epochs):
            print(f'\nEpoch: {epoch+1}/{self.config.num_epochs}')
            
            train_loss = self.train_epoch()
            val_loss = self.validate()
            
            if self.scheduler:
                self.scheduler.step()
            
            # Log metrics
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/val', val_loss, epoch)
            self.writer.add_scalar('Learning_rate', 
                self.scheduler.get_last_lr()[0] if self.scheduler else self.config.learning_rate, 
                epoch)
            
            # Save checkpoint
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
            self.save_checkpoint(epoch, is_best)

        # Close tensorboard writer
        self.writer.close()

    def __del__(self):
        # Ensure writer is closed when trainer is destroyed
        if hasattr(self, 'writer'):
            self.writer.close()

def main():
    config = ConfigParser('config.yaml')
    trainer = Trainer(config)
    trainer.train()

if __name__ == '__main__':
    main()
