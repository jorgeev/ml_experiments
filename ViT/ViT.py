import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        # Handle non-square images
        if isinstance(img_size, int):
            self.img_height = self.img_width = img_size
        else:
            self.img_height, self.img_width = img_size
            
        self.patch_size = patch_size
        self.n_patches = (self.img_height // patch_size) * (self.img_width // patch_size)
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # (B, embed_dim, H/patch_size, W/patch_size)
        x = x.flatten(2)  # (B, embed_dim, n_patches)
        x = x.transpose(1, 2)  # (B, n_patches, embed_dim)
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, n_patches, embed_dim):
        super().__init__()
        #self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches, embed_dim))

    def forward(self, x):
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
        attn = attn - attn.mean(dim=-1, keepdim=True)[0]
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x
    
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.attn = MultiHeadAttention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim, eps=1e-6)
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
        self.norm1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.attn = MultiHeadAttention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim, eps=1e-6)
        

class Decoder(nn.Module):
    def __init__(self, embed_dim, img_size, patch_size):
        super().__init__()
        # Handle non-square images
        if isinstance(img_size, int):
            self.img_height = self.img_width = img_size
        else:
            self.img_height, self.img_width = img_size
            
        self.patch_size = patch_size
        self.n_patches = (self.img_height // patch_size) * (self.img_width // patch_size)
        
        # Calculate the feature map size after patch embedding
        self.feat_height = self.img_height // patch_size
        self.feat_width = self.img_width // patch_size
        
        # Convolutional decoder layers to restore original resolution
        self.decoder = nn.Sequential(
            nn.Conv2d(embed_dim, 512, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),  # Smooth Upsampling
            
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            
            nn.Conv2d(64, 1, kernel_size=1)  # Single output channel
        )
        
    def forward(self, x):
        B = x.shape[0]
        # Remove CLS token
        # x = x[:, 1:]
        # Reshape to (B, H/patch_size, W/patch_size, embed_dim)
        x = x.reshape(B, self.feat_height, self.feat_width, -1)
        # Convert to (B, embed_dim, H/patch_size, W/patch_size)
        x = x.permute(0, 3, 1, 2)
        # Pass through decoder to restore original resolution
        x = self.decoder(x)
        # Ensure output size matches input size exactly
        if x.shape[-2:] != (self.img_height, self.img_width):
            x = F.interpolate(x, size=(self.img_height, self.img_width), mode='bilinear', align_corners=False)
        
        return x

class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.pos_embed = PositionalEncoding(self.patch_embed.n_patches, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.decoder = Decoder(embed_dim, img_size, patch_size)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        x = self.pos_embed(x)
        x = self.dropout(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        # Pass through decoder to get reconstructed image
        x = self.decoder(x)
        return x

# Update these hyperparameters
img_size = (648, 712)  # Specify both height and width
patch_size = 16
in_channels = 30
embed_dim = 768
depth = 12
num_heads = 12
mlp_ratio = 4.0
dropout = 0.1

# Instantiate the model
model = VisionTransformer(
    img_size=img_size,
    patch_size=patch_size,
    in_channels=in_channels,
    embed_dim=embed_dim,
    depth=depth,
    num_heads=num_heads,
    mlp_ratio=mlp_ratio,
    dropout=dropout
)

# Example input tensor (batch_size, channels, height, width)
x = torch.randn(1, 30, 652, 714)

# Forward pass
output = model(x)
print(output.shape)  # Should be (1, 1, 652, 714) - single channel output