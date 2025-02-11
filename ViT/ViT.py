import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
from einops import rearrange, repeat
import numpy as np
import math

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768, stride_percentage=0.1):
        super().__init__()
        # Handle non-square images
        if isinstance(img_size, int):
            self.img_height = self.img_width = img_size
        else:
            self.img_height, self.img_width = img_size

        self.patch_size = patch_size
        # Calculate stride for desired overlap percentage
        # stride = patch_size * (1 - overlap)
        # For 10% overlap (0.1), we use (1 - 0.1) = 0.9
        self.stride = max(1, int(patch_size * (1 - stride_percentage)))  # For 10% overlap, use 90% of patch_size
        
        #print(f"Image size: {self.img_height}x{self.img_width}")
        #print(f"Patch size: {patch_size}")
        #print(f"Stride: {self.stride}")
        #print(f"Overlap percentage: {stride_percentage*100}%")
        #print(f"Actual overlap pixels: {patch_size - self.stride}")

        # Compute number of patches using Conv2d output size formula
        # For a given input size i, output size = floor((i - k + 2p) / s) + 1
        # where k is kernel size, p is padding (0 in our case), and s is stride
        self.n_patches_h = ((self.img_height - patch_size) // self.stride) + 1
        self.n_patches_w = ((self.img_width - patch_size) // self.stride) + 1
        self.n_patches = self.n_patches_h * self.n_patches_w
        
        #print(f"Number of patches: {self.n_patches} ({self.n_patches_h}x{self.n_patches_w})")

        # Convolution with controlled stride
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=self.stride, padding=0)

    def forward(self, x):
        B = x.shape[0]
        x = self.proj(x)  # Shape: (B, embed_dim, H', W')
        x = x.flatten(2)  # Flatten spatial dimensions into sequence
        x = x.transpose(1, 2)  # (B, num_patches, embed_dim)
        actual_patches = x.shape[1]
        if actual_patches != self.n_patches:
            print(f"Warning: Actual patches ({actual_patches}) differs from computed patches ({self.n_patches})")
            self.n_patches = actual_patches
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, n_patches, embed_dim):
        super().__init__()
        #print(f"Initializing positional encoding for {n_patches} patches (plus 1 CLS token)")
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))
        # Initialize the positional embeddings
        nn.init.normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        B, N, D = x.shape
        if N != self.pos_embed.shape[1]:
            raise ValueError(f"Input sequence length ({N}) doesn't match position embedding size ({self.pos_embed.shape[1]})")
        return x + self.pos_embed

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x
    
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x
    
class EncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        

class Decoder(nn.Module):
    def __init__(self, embed_dim, img_size, patch_size, stride_percentage=0.1):
        super().__init__()
        # Handle non-square images
        if isinstance(img_size, int):
            self.img_height = self.img_width = img_size
        else:
            self.img_height, self.img_width = img_size
            
        self.patch_size = patch_size
        self.stride = max(1, int(patch_size * (1 - stride_percentage)))
        
        # Calculate feature map size using Conv2d output size formula
        self.feat_height = ((self.img_height - patch_size) // self.stride) + 1
        self.feat_width = ((self.img_width - patch_size) // self.stride) + 1
        
        # Calculate intermediate sizes for proper upsampling
        self.sizes = self._calculate_sizes()
        
        # Convolutional decoder layers
        self.decoder = nn.Sequential(
            # Initial projection
            nn.Conv2d(embed_dim, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            
            # Progressive upsampling to ensure exact size matching
            nn.Upsample(size=self.sizes[0], mode='bilinear', align_corners=True),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            nn.Upsample(size=self.sizes[1], mode='bilinear', align_corners=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            nn.Upsample(size=self.sizes[2], mode='bilinear', align_corners=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            # Final upsampling to exact input size
            nn.Upsample(size=(self.img_height, self.img_width), mode='bilinear', align_corners=True),
            nn.Conv2d(64, 1, kernel_size=1)
        )
        
    def _calculate_sizes(self):
        """Calculate intermediate sizes for progressive upsampling"""
        # Calculate three intermediate sizes that progressively approach the target size
        h1 = self.feat_height * 2
        w1 = self.feat_width * 2
        
        h2 = h1 * 2
        w2 = w1 * 2
        
        h3 = h2 * 2
        w3 = w2 * 2
        
        return [
            (h1, w1),
            (h2, w2),
            (h3, w3)
        ]
        
    def forward(self, x):
        B = x.shape[0]
        # Remove CLS token
        x = x[:, 1:]
        # Reshape to match feature map dimensions
        x = x.reshape(B, self.feat_height, self.feat_width, -1)
        # Convert to channel-first format
        x = x.permute(0, 3, 1, 2)
        # Decode
        x = self.decoder(x)
        return x  # No need for final interpolation as it's handled in the decoder

class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, dropout=0.1, stride_percentage=0.1):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim, stride_percentage)
        self.pos_embed = PositionalEncoding(self.patch_embed.n_patches, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.decoder = Decoder(embed_dim, img_size, patch_size, stride_percentage)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=B)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.pos_embed(x)
        x = self.dropout(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        x = self.decoder(x)
        return x

# Update these hyperparameters
img_size = (652, 714)  # Specify both height and width
patch_size = 16
in_channels = 30
embed_dim = 768
depth = 12
num_heads = 12
mlp_ratio = 4.0
dropout = 0.1
stride_percentage = 0.1

# Instantiate the model
model = VisionTransformer(
    img_size=img_size,
    patch_size=patch_size,
    in_channels=in_channels,
    embed_dim=embed_dim,
    depth=depth,
    num_heads=num_heads,
    mlp_ratio=mlp_ratio,
    dropout=dropout,
    stride_percentage=stride_percentage
)

# Example input tensor (batch_size, channels, height, width)
x = torch.randn(1, 30, 652, 714)

# Forward pass
output = model(x)
print(output.shape)  # Should be (1, 1, 652, 714) - single channel output