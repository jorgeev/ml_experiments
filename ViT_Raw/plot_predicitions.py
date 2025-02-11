#%%
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cmocean.cm as cmo
import torch.nn.functional as F
import torch
h5data = h5py.File('/unity/g2/jvelasco/ai_outs/task21_set1/vitraw_training/evaluation/evaluation_results.h5', 'r')

predictions = h5data['predictions'][:].squeeze()
targets = h5data['targets'][:]
resampeld_predictions = np.zeros_like(targets)
for ii in range(predictions.shape[0]):
    # Add batch and channel dimensions: (H,W) -> (1,1,H,W)
    pred_tensor = torch.from_numpy(predictions[ii]).unsqueeze(0).unsqueeze(0)
    # Interpolate
    pred_tensor = F.interpolate(pred_tensor, 
                              size=(648,712), 
                              mode='bilinear', 
                              align_corners=False)
    # Remove batch and channel dimensions and convert back to numpy
    resampeld_predictions[ii] = pred_tensor.squeeze().numpy()

masks = h5data['masks'][0,0]
print(f"predictions.shape: {predictions.shape}")
print(f"targets.shape: {targets.shape}")
print(f"masks.shape: {masks.shape}")

# Conmpute gradient in x and y direction for each sample
(grad_predictions_t,grad_predictions_x, grad_predictions_y) = np.gradient(resampeld_predictions)
(grad_targets_t,grad_targets_x, grad_targets_y) = np.gradient(targets.squeeze())

## Total gradient
grad_predictions = np.sqrt(grad_predictions_x**2 + grad_predictions_y**2)
grad_targets = np.sqrt(grad_targets_x**2 + grad_targets_y**2)
print(f"grad_predictions.shape: {grad_predictions.shape}")
print(f"grad_targets.shape: {grad_targets.shape}")
#%%
# Difference between predictions and targets
grad_diff = grad_predictions - grad_targets
diff = resampeld_predictions - targets
print(f"diff.shape: {diff.shape}")
print(f"grad_diff.shape: {grad_diff.shape}")

# %%
for ii in range(predictions.shape[0]):
    fig, ax = plt.subplots(2, 3, figsize=(10, 6))

    # Plot the predictions
    i1 = ax[0, 0].imshow(resampeld_predictions[0, :, :] * masks, cmap=cmo.balance, vmin=-0.8, vmax=0.8)
    ax[0, 0].set_title('Prediction')
    ax[0, 0].axis('off')
    plt.colorbar(i1)

    i2 = ax[0, 1].imshow(targets[0, :, :] * masks, cmap=cmo.balance, vmin=-0.8, vmax=0.8)
    ax[0, 1].set_title('Ground Truth')
    ax[0, 1].axis('off')
    plt.colorbar(i2)

    i3 = ax[1, 0].imshow(grad_predictions[0, :, :] * masks, cmap=cmo.balance, vmin=-0, vmax=0.05)
    ax[1, 0].set_title('Gradient Prediction')
    ax[1, 0].axis('off')
    plt.colorbar(i3)

    i4 = ax[1, 1].imshow(grad_targets[0, :, :] * masks, cmap=cmo.balance, vmin=0, vmax=0.05)
    ax[1, 1].set_title('Gradient Ground Truth')
    ax[1, 1].axis('off')
    plt.colorbar(i4)

    i5 = ax[0, 2].imshow(diff[0, :, :] * masks, cmap=cmo.balance, vmin=-0.05, vmax=0.05)
    ax[0, 2].set_title('Prediction - Ground Truth')
    ax[0, 2].axis('off')
    plt.colorbar(i5)

    i6 = ax[1, 2].imshow(grad_diff[0, :, :] * masks, cmap=cmo.balance, vmin=-0.01, vmax=0.01)
    ax[1, 2].set_title('Gradient (Prediction - GTruth)')
    ax[1, 2].axis('off')
    plt.colorbar(i6)

    fig.tight_layout()
    fig.savefig(f'/unity/g2/jvelasco/ai_outs/task21_set1/vitraw_training/plots/results_gradients_{ii}.png')
    fig.clf()
    plt.close()


# %%
