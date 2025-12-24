import torch
import torch.nn as nn
import torch.nn.functional as F

class SiameseNetwork(nn.Module):
    """Improved Siamese Network with BatchNorm and Multi-Channel Similarity Head."""
    
    def __init__(self, embedding_dim=512):
        super(SiameseNetwork, self).__init__()
        
        self.embedding_dim = embedding_dim
        
        # Convolutional layers with BatchNorm (shared between twins)
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=10, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=7, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=4, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=4, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # Calculate the flattened feature size dynamically
        self._to_linear = None
        self._get_conv_output((3, 105, 105))
        
        # Embedding layer with BatchNorm and Dropout
        self.embedding = nn.Sequential(
            nn.Linear(self._to_linear, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, embedding_dim),
            nn.BatchNorm1d(embedding_dim)
        )
        
        # Multi-channel similarity fusion head
        fusion_input_dim = embedding_dim * 2 + 2
        
        self.similarity_head = nn.Sequential(
            nn.Linear(fusion_input_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(64, 1)
        )
    
    def _get_conv_output(self, shape):
        """Helper to calculate the output size after conv layers"""
        with torch.no_grad():
            dummy_input = torch.zeros(1, *shape)
            x = self.conv1(dummy_input)
            x = self.conv2(x)
            x = self.conv3(x)
            x = self.conv4(x)
            self._to_linear = x.view(1, -1).size(1)
        
    def forward_once(self, x):
        """Forward pass for one image - returns embedding"""
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)
        x = self.embedding(x)
        return x
    
    def forward(self, img1, img2):
        """Forward pass for a pair of images with multi-channel similarity"""
        z1 = self.forward_once(img1)
        z2 = self.forward_once(img2)
        
        d_l1 = torch.abs(z1 - z2)
        d_l2 = (z1 - z2) ** 2
        cos_sim = F.cosine_similarity(z1, z2, dim=1, eps=1e-8).unsqueeze(1)
        dot_prod = (z1 * z2).sum(dim=1, keepdim=True)
        
        similarity_features = torch.cat([d_l1, d_l2, cos_sim, dot_prod], dim=1)
        logits = self.similarity_head(similarity_features)
        
        return logits

# Initialize model
model = SiameseNetwork(embedding_dim=512)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print("=" * 80)
print("SIAMESE NETWORK ARCHITECTURE DETAILS")
print("=" * 80)
print()

# Test with dummy input to get intermediate dimensions
dummy_input = torch.zeros(1, 3, 105, 105)

print("INPUT DIMENSIONS")
print("-" * 80)
print(f"Input image size: 3 × 105 × 105 (RGB)")
print()

print("CONVOLUTIONAL LAYERS (Shared Twin Network)")
print("-" * 80)
x = dummy_input
print(f"Input: {list(x.shape[1:])}")

x = model.conv1[0](x)  # Conv
print(f"After Conv1 (10×10, 64 filters, stride=1): {list(x.shape[1:])}")
x = model.conv1[3](model.conv1[2](model.conv1[1](x)))  # BN, ReLU, MaxPool
print(f"After MaxPool1 (2×2, stride=2): {list(x.shape[1:])}")
conv1_params = sum(p.numel() for p in model.conv1.parameters())
print(f"Parameters: {conv1_params:,}")
print()

x = model.conv2[0](x)  # Conv
print(f"After Conv2 (7×7, 128 filters, stride=1): {list(x.shape[1:])}")
x = model.conv2[3](model.conv2[2](model.conv2[1](x)))  # BN, ReLU, MaxPool
print(f"After MaxPool2 (2×2, stride=2): {list(x.shape[1:])}")
conv2_params = sum(p.numel() for p in model.conv2.parameters())
print(f"Parameters: {conv2_params:,}")
print()

x = model.conv3[0](x)  # Conv
print(f"After Conv3 (4×4, 256 filters, stride=1): {list(x.shape[1:])}")
x = model.conv3[3](model.conv3[2](model.conv3[1](x)))  # BN, ReLU, MaxPool
print(f"After MaxPool3 (2×2, stride=2): {list(x.shape[1:])}")
conv3_params = sum(p.numel() for p in model.conv3.parameters())
print(f"Parameters: {conv3_params:,}")
print()

x = model.conv4[0](x)  # Conv
print(f"After Conv4 (4×4, 256 filters, stride=1): {list(x.shape[1:])}")
x = model.conv4[2](model.conv4[1](x))  # BN, ReLU (no pooling)
print(f"After ReLU: {list(x.shape[1:])}")
conv4_params = sum(p.numel() for p in model.conv4.parameters())
print(f"Parameters: {conv4_params:,}")
print()

print("EMBEDDING LAYERS")
print("-" * 80)
x_flat = x.view(x.size(0), -1)
print(f"Flattened features: {x_flat.shape[1]}")

x = model.embedding[0](x_flat)  # FC1
print(f"After FC1 (Linear {x_flat.shape[1]} → 1024): {list(x.shape[1:])}")
fc1_params = sum(p.numel() for p in [model.embedding[0].weight, model.embedding[0].bias])
print(f"Parameters: {fc1_params:,}")

x = model.embedding[4](x)  # FC2 (after BN, ReLU, Dropout)
print(f"After FC2 (Linear 1024 → 512): {list(x.shape[1:])}")
fc2_params = sum(p.numel() for p in [model.embedding[4].weight, model.embedding[4].bias])
print(f"Parameters: {fc2_params:,}")
print(f"Final embedding dimension: {model.embedding_dim}")

embedding_params = sum(p.numel() for p in model.embedding.parameters())
print(f"Total embedding parameters: {embedding_params:,}")
print()

print("MULTI-CHANNEL SIMILARITY HEAD")
print("-" * 80)
fusion_input_dim = model.embedding_dim * 2 + 2
print(f"Similarity channels:")
print(f"  - L1 distance: {model.embedding_dim} dimensions")
print(f"  - L2 distance: {model.embedding_dim} dimensions")
print(f"  - Cosine similarity: 1 dimension")
print(f"  - Dot product: 1 dimension")
print(f"Concatenated features: {fusion_input_dim} dimensions")
print()

print(f"Fusion Network:")
print(f"  FC1: Linear {fusion_input_dim} → 256, ReLU, Dropout(0.5)")
fc1_sim_params = sum(p.numel() for p in [model.similarity_head[0].weight, model.similarity_head[0].bias])
print(f"  Parameters: {fc1_sim_params:,}")

print(f"  FC2: Linear 256 → 64, ReLU, Dropout(0.4)")
fc2_sim_params = sum(p.numel() for p in [model.similarity_head[3].weight, model.similarity_head[3].bias])
print(f"  Parameters: {fc2_sim_params:,}")

print(f"  FC3: Linear 64 → 1 (logits output)")
fc3_sim_params = sum(p.numel() for p in [model.similarity_head[6].weight, model.similarity_head[6].bias])
print(f"  Parameters: {fc3_sim_params:,}")

similarity_params = sum(p.numel() for p in model.similarity_head.parameters())
print(f"Total similarity head parameters: {similarity_params:,}")
print()

print("PARAMETER SUMMARY")
print("-" * 80)
print(f"Convolutional layers: {conv1_params + conv2_params + conv3_params + conv4_params:,}")
print(f"Embedding layers: {embedding_params:,}")
print(f"Similarity head: {similarity_params:,}")
print(f"Total trainable parameters: {trainable_params:,}")
print(f"Total parameters: {total_params:,}")
print()

print("REGULARIZATION")
print("-" * 80)
print(f"BatchNorm2d: After each convolutional layer")
print(f"BatchNorm1d: After each fully connected layer in embedding")
print(f"Dropout: 0.5 (after FC1 in embedding), 0.5 (after FC1 in similarity), 0.4 (after FC2 in similarity)")
print()

print("=" * 80)
