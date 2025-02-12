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
        
        # Initialize best_val_loss and early stopping parameters
        self.best_val_loss = float('inf')
        self.last_improvement = float('inf')  # For early stopping tracking
        self.patience = getattr(self.config, 'early_stopping_patience', 10)  # Default patience of 10 epochs
        self.min_delta = getattr(self.config, 'early_stopping_min_delta', 1e-4)  # Minimum change to qualify as an improvement
        self.counter = 0  # Counter for patience
        
        self.setup_tensorboard()
        self.setup_model()
        self.setup_data()
        self.setup_training()

    def setup_tensorboard(self):
        # Create root directory for this run
        self.run_dir = Path(self.config.log_dir) / self.config.run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        # Create logs and checkpoints directories
        self.logs_dir = self.run_dir / 'logs'
        self.checkpoints_dir = self.run_dir / 'checkpoints'
        self.logs_dir.mkdir(exist_ok=True)
        self.checkpoints_dir.mkdir(exist_ok=True)
        
        # Create tensorboard writer in logs directory
        self.writer = SummaryWriter(self.logs_dir)
        
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
            dropout=self.config.dropout,
        )

        # Apply Xavier Uniform Initialization
        def init_weights(m):
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

        self.model.apply(init_weights)  # Apply initialization
        
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
        def grad_mse_loss(output, target, mask):
            target = target.unsqueeze(1)  # Shape: [batch_size, 1, height, width]
            
            output_loss = F.mse_loss(output * mask, target * mask, reduction='sum')
            loss =  output_loss

            # Normalize by the number of valid points in the mask
            valid_points = torch.sum(mask) + 1e-8
            return loss / valid_points

        def tv_loss(x):
            return torch.mean(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:])) + \
                              torch.mean(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]))
        
        self.criterion = grad_mse_loss
        self.tv_loss = tv_loss
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
            # Send data and targets to device
            mask = data[:, -1, :, :].to(self.device)  # Shape: [B, 1, H, W]
            # Remove the mask from input data
            data = data[:, :-1, :, :].to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(data)

            loss = self.criterion(outputs, targets, mask) + self.tv_loss(outputs)
            
            loss.backward()
            # Clip gradients to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1, norm_type=2)

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
                loss = self.criterion(outputs, targets, mask) + self.tv_loss(outputs)
                total_loss += loss.item()

        return total_loss / len(self.val_loader)

    def save_checkpoint(self, epoch, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'state_dict': self.model.module.state_dict() if self.multi_gpu else self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'last_improvement': self.last_improvement,
            'counter': self.counter
        }
        
        # Save latest checkpoint
        torch.save(checkpoint, self.checkpoints_dir / 'latest_checkpoint.pth')
        
        # Save best model
        if is_best:
            torch.save(checkpoint, self.checkpoints_dir / 'best_model.pth')

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
            
        # Load training state
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.last_improvement = checkpoint.get('last_improvement', float('inf'))
        self.counter = checkpoint.get('counter', 0)
        start_epoch = checkpoint.get('epoch', 0)
        
        return start_epoch

    def update_best_model(self, val_loss, epoch):
        """
        Handle best model tracking and saving.
        Returns whether this was the best model so far.
        """
        is_best = val_loss < self.best_val_loss
        if is_best:
            self.best_val_loss = val_loss
            print(f"New best validation loss: {val_loss}")
        self.save_checkpoint(epoch, is_best)
        return is_best

    def early_stopping_check(self, val_loss):
        """
        Check if training should stop based on validation loss improvement.
        Uses a separate variable (last_improvement) for early stopping tracking.
        Returns True if training should stop, False otherwise.
        """
        if val_loss < (self.last_improvement - self.min_delta):
            # print(f"Validation loss improved for early stopping from {self.last_improvement} to {val_loss}")
            self.last_improvement = val_loss
            self.counter = 0
            return False
        else:
            self.counter += 1
            # print(f"Early stopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                print(f'\nEarly stopping triggered after {self.patience} epochs without improvement.')
                return True
            return False

    def train(self):
        start_epoch = 0

        if hasattr(self.config, 'resume_from') and self.config.resume_from:
            print(f"Resuming from checkpoint: {self.config.resume_from}")
            start_epoch = self.load_checkpoint(self.config.resume_from)
            self.last_improvement = self.best_val_loss  # Initialize last_improvement from checkpoint

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
            
            # Handle both best model tracking and early stopping independently
            self.update_best_model(val_loss, epoch)
            
            if self.early_stopping_check(val_loss):
                print(f'Training stopped at epoch {epoch+1}')
                break

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
