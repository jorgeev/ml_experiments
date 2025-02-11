import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from tqdm import tqdm
import yaml
from ViT import VisionTransformer
import data_loader.data_loaders as module_data
import h5py
from os.path import join
import pickle
from data_loader.data_loaders import DefaultDataLoader
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
#import cmocean.cm as cmo

class Evaluator:
    def __init__(self, config_path):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.setup_model()
        self.setup_data()
        
        # Create output directory
        self.output_dir = Path(self.config['log_dir']) / self.config['run_name'] / 'evaluation'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Read the scalers from the data_dir
        with open(join(self.config['train_data_dir'], 'scalers.pkl'), 'rb') as f:
            self.scalers = pickle.load(f)
        self.mean_ssh = self.scalers['ssh']['mean']
        self.std_ssh = self.scalers['ssh']['std']
    
    def setup_model(self):
        # Initialize model with same configuration
        self.model = VisionTransformer(
            img_size=self.config['img_size'],
            patch_size=self.config['patch_size'],
            in_channels=self.config['in_channels'],
            embed_dim=self.config['embed_dim'],
            depth=self.config['depth'],
            num_heads=self.config['num_heads'],
            mlp_ratio=self.config['mlp_ratio'],
            dropout=self.config['dropout']
        ).to(self.device)

        # Load best model checkpoint
        checkpoint_path = Path(self.config['save_dir']) / 'best_model.pth'
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['state_dict'])
        print(f"Loaded checkpoint from {checkpoint_path}")
        print(f"Best validation loss: {checkpoint['best_val_loss']:.6f}")

    def setup_data(self):
        # Setup validation dataloader
        self.val_loader = DefaultDataLoader(
            data_dir=self.config['val_data_dir'],
            batch_size=self.config['batch_size'],
            shuffle=False,
            validation_split=0.0,
            num_workers=self.config['num_workers'],
           training=False,
            previous_days=self.config['previous_days'],
            dataset_type=self.config['dataset_type']
        )


    def masked_mse_loss(self, output, target, mask, reduction='mean'):
        target = target.unsqueeze(1)
        # Ensure all tensors have the same size
        if output.shape != target.shape:
            output = F.interpolate(output, size=target.shape[-2:], mode='bilinear', align_corners=False)
        if mask.shape[-2:] != target.shape[-2:]:
            mask = F.interpolate(mask, size=target.shape[-2:], mode='nearest')
        
        # Calculate squared error for each element
        squared_error = (output * mask - target * mask) ** 2
        
        # Sum over spatial dimensions for each sample
        sample_errors = squared_error.sum(dim=(1, 2, 3))  # Sum over channels, height, width
        valid_points = mask.sum(dim=(1, 2, 3))  # Sum over channels, height, width
        
        # Calculate MSE for each sample
        sample_mse = sample_errors / (valid_points + 1e-8)
        
        if reduction == 'mean':
            return sample_mse.mean()
        elif reduction == 'none':
            return sample_mse
        
    def evaluate(self):
        self.model.eval()
        
        # Initialize lists to store results
        all_predictions = []
        all_targets = []
        all_masks = []
        all_mse = []
        all_sample_mse = []  # New list to store individual sample MSEs
        sample_indices = []  # To keep track of sample indices
        current_index = 0    # Counter for samples
        
        print("Starting evaluation...")
        with torch.no_grad():
            for batch_idx, (data, targets) in enumerate(tqdm(self.val_loader)):
                # Get the gulf mask from the last channel of data
                mask = data[:, -1:, :, :].to(self.device)
                # Remove the mask from input data
                data = data[:, :-1, :, :].to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                outputs = self.model(data)
                
                # Inverse transform the outputs and targets
                outputs = self.mean_ssh + outputs * self.std_ssh
                targets = self.mean_ssh + targets * self.std_ssh
                
                # Calculate MSE for each sample in the batch
                sample_mse = self.masked_mse_loss(outputs, targets, mask, reduction='none')
                batch_mse = sample_mse.mean().item()
                
                # Calculate RMSE for each sample
                sample_rmse = torch.sqrt(sample_mse).cpu().numpy()
                
                # Store results
                all_predictions.append(outputs.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
                all_masks.append(mask.cpu().numpy())
                all_mse.append(batch_mse)
                all_sample_mse.extend(sample_rmse)  # Add individual sample RMSEs
                
                # Store sample indices
                batch_size = len(sample_rmse)
                sample_indices.extend(range(current_index, current_index + batch_size))
                current_index += batch_size
                
                # Save batch results periodically
                if (batch_idx + 1) % 10 == 0:
                    print(f"Processed {batch_idx + 1} batches. Current batch MSE: {batch_mse:.6f}")
        
        # Calculate final metrics
        mean_mse = np.mean(all_mse)
        std_mse = np.std(all_mse)
        
        # Save results
        results_file = self.output_dir / 'evaluation_results.h5'
        with h5py.File(results_file, 'w') as f:
            f.create_dataset('predictions', data=np.concatenate(all_predictions, axis=0))
            f.create_dataset('targets', data=np.concatenate(all_targets, axis=0))
            f.create_dataset('masks', data=np.concatenate(all_masks, axis=0))
            f.create_dataset('mse_per_batch', data=np.array(all_mse))
            f.create_dataset('rmse_per_sample', data=np.array(all_sample_mse))  # Save individual RMSEs
            f.attrs['mean_mse'] = mean_mse
            f.attrs['std_mse'] = std_mse
            f.attrs['mean_rmse'] = np.mean(all_sample_mse)
            f.attrs['std_rmse'] = np.std(all_sample_mse)
        
        # Save metrics to text file
        metrics_file = self.output_dir / 'metrics.txt'
        with open(metrics_file, 'w') as f:
            f.write(f"Evaluation Results\n")
            f.write(f"================\n")
            f.write(f"Mean MSE: {mean_mse:.6f}\n")
            f.write(f"Std MSE: {std_mse:.6f}\n")
            f.write(f"Mean RMSE: {np.mean(all_sample_mse):.6f}\n")
            f.write(f"Std RMSE: {np.std(all_sample_mse):.6f}\n")
        
        # Save individual RMSE values to a separate file
        individual_metrics_file = self.output_dir / 'individual_rmse.txt'
        with open(individual_metrics_file, 'w') as f:
            f.write("Individual Sample RMSE Values\n")
            f.write("============================\n")
            f.write("Sample_Index\tRMSE\n")
            
            # Sort RMSE values to identify best and worst cases
            rmse_with_indices = list(zip(sample_indices, all_sample_mse))
            rmse_with_indices.sort(key=lambda x: x[1])  # Sort by RMSE value
            
            # Write all values
            for idx, rmse in rmse_with_indices:
                f.write(f"{idx}\t{rmse:.6f}\n")
            
            # Add summary statistics at the end
            f.write("\nSummary Statistics\n")
            f.write("==================\n")
            f.write(f"Best 5 samples (lowest RMSE):\n")
            for idx, rmse in rmse_with_indices[:5]:
                f.write(f"Sample {idx}: {rmse:.6f}\n")
            
            f.write(f"\nWorst 5 samples (highest RMSE):\n")
            for idx, rmse in rmse_with_indices[-5:]:
                f.write(f"Sample {idx}: {rmse:.6f}\n")
            
            f.write(f"\nQuartile Statistics:\n")
            rmse_array = np.array(all_sample_mse)
            f.write(f"25th percentile: {np.percentile(rmse_array, 25):.6f}\n")
            f.write(f"50th percentile (median): {np.percentile(rmse_array, 50):.6f}\n")
            f.write(f"75th percentile: {np.percentile(rmse_array, 75):.6f}\n")
        
        print("\nEvaluation completed!")
        print(f"Results saved to: {self.output_dir}")
        print(f"Mean MSE: {mean_mse:.6f} ± {std_mse:.6f}")
        print(f"Mean RMSE: {np.mean(all_sample_mse):.6f} ± {np.std(all_sample_mse):.6f}")
        
        # Create scatter plot of RMSE values
        plt.figure(figsize=(12, 8), dpi=300)
        plt.scatter(sample_indices, all_sample_mse, alpha=0.5, color='blue', s=20)
        
        # Add mean RMSE line
        mean_rmse = np.mean(all_sample_mse)
        plt.axhline(y=mean_rmse, color='r', linestyle='--', label=f'Mean RMSE: {mean_rmse:.6f}')
        
        # Customize the plot
        plt.title(f'RMSE per Sample (Mean: {mean_rmse:.6f})', fontsize=14, pad=20)
        plt.xlabel('Sample Index', fontsize=12)
        plt.ylabel('RMSE', fontsize=12)
        plt.ylim(0.008, 0.014)
        
        # Add grid
        plt.grid(True, which='major', linestyle='-', alpha=0.5)
        plt.grid(True, which='minor', linestyle=':', alpha=0.3)
        plt.minorticks_on()
        
        # Add legend
        plt.legend()
        
        # Adjust layout and save
        plt.tight_layout()
        plot_file = self.output_dir / 'rmse_scatter.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight', pad_inches=0.05)
        plt.close()
        
        print(f"Individual RMSE values saved to: {individual_metrics_file}")
        print(f"RMSE scatter plot saved to: {plot_file}")

def main():
    evaluator = Evaluator('config.yaml')
    evaluator.evaluate()

if __name__ == '__main__':
    main() 