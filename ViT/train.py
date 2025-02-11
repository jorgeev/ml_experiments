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
from data_loader.data_sets import SimSatelliteDataset
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
            stride_percentage=self.config.stride_percentage
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
        def grad_mse_loss(output, target):

            # Define Sobel kernels for computing gradients along x and y directions
            sobel_kernel_x = torch.tensor([[[-1, 0, 1],
                                            [-2, 0, 2],
                                            [-1, 0, 1]]], dtype=output.dtype, device=output.device)

            sobel_kernel_y = torch.tensor([[[-1, -2, -1],
                                            [ 0,  0,  0],
                                            [ 1,  2,  1]]], dtype=output.dtype, device=output.device)

            # Reshape kernels to match the conv2d weight shape: [out_channels, in_channels, kH, kW]
            sobel_kernel_x = sobel_kernel_x.unsqueeze(1)  # Shape: [1, 1, 3, 3]
            sobel_kernel_y = sobel_kernel_y.unsqueeze(1)  # Shape: [1, 1, 3, 3]

            # output = output.unsqueeze(1)  # Shape: [batch_size, 1, height, width]
            target = target.unsqueeze(1)  # Shape: [batch_size, 1, height, width]
            if output.shape != target.shape:
                # Resize output to match target size
                output = F.interpolate(output, size=target.shape[-2:], mode='bilinear', align_corners=False)

            #print("Inside the grad_mse_loss function")
            #print("output.shape: ", output.shape)
            #print("target.shape: ", target.shape)

            # Compute gradients along x and y directions using conv2d
            grad_output_x = F.conv2d(output, sobel_kernel_x, padding=1)  # Shape: [batch_size, 1, height, width]
            grad_output_y = F.conv2d(output, sobel_kernel_y, padding=1)  # Shape: [batch_size, 1, height, width]

            # Compute gradients of the target
            grad_target_x = F.conv2d(target, sobel_kernel_x, padding=1)  # Shape: [batch_size, 1, height, width]
            grad_target_y = F.conv2d(target, sobel_kernel_y, padding=1)  # Shape: [batch_size, 1, height, width]

            # Compute the gradient magnitude (2D norm) for each element in the batch
            grad_magnitude_output = torch.sqrt(grad_output_x ** 2 + grad_output_y ** 2)  # Shape: [batch_size, 1, height, width]
            grad_magnitude_target = torch.sqrt(grad_target_x ** 2 + grad_target_y ** 2)  # Shape: [batch_size, 1, height, width]

            # Normalize the gradients with mean 0 and std 1
            output_gradient = (grad_magnitude_output - grad_magnitude_output.mean()) / grad_magnitude_output.std()
            target_gradient = (grad_magnitude_target - grad_magnitude_target.mean()) / grad_magnitude_target.std()
            
            output_loss = F.mse_loss(output, target, reduction='sum')
            gradient_loss = F.mse_loss(output_gradient, target_gradient, reduction='sum')
            loss = output_loss + gradient_loss
            
            return loss
        
        self.criterion = grad_mse_loss
        
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
            data = data.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(data)
            # print("Inside the training loop")
            # print("outputs.shape: ", outputs.shape)
            # print("targets.shape: ", targets.shape)
            loss = self.criterion(outputs, targets)
            
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
                loss = self.criterion(outputs, targets)
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
            
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        start_epoch = checkpoint.get('epoch', 0)
        
        return start_epoch

    def early_stopping_check(self, val_loss):
        """
        Check if training should stop based on validation loss improvement.
        Returns True if training should stop, False otherwise.
        """
        if val_loss < (self.best_val_loss - self.min_delta):
            self.best_val_loss = val_loss
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                print(f'\nEarly stopping triggered after {self.patience} epochs without improvement.')
                return True
            return False

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

            # Early stopping check
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
