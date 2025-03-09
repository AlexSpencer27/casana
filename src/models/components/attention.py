"""
Attention mechanisms for feature weighting and selection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class AttentionModule(nn.Module):
    def __init__(
        self,
        feature_dim,
        attention_dim=64,
        num_heads=1,
        attention_type='self'
    ):
        """
        Apply attention mechanisms to features.
        
        Args:
            feature_dim: Feature dimension
            attention_dim: Dimension of attention calculation (default: 64)
            num_heads: Number of attention heads (default: 1)
            attention_type: Type of attention ('self', 'channel', 'spatial') (default: 'self')
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.attention_dim = attention_dim
        self.num_heads = num_heads
        self.attention_type = attention_type
        
        if attention_type == 'self':
            # Self-attention mechanism
            self.query = nn.Linear(feature_dim, attention_dim * num_heads)
            self.key = nn.Linear(feature_dim, attention_dim * num_heads)
            self.value = nn.Linear(feature_dim, attention_dim * num_heads)
            self.output = nn.Linear(attention_dim * num_heads, feature_dim)
            
        elif attention_type == 'channel':
            # Channel attention (squeeze and excitation style)
            self.fc1 = nn.Linear(feature_dim, attention_dim)
            self.fc2 = nn.Linear(attention_dim, feature_dim)
            
        elif attention_type == 'spatial':
            # Spatial attention for 1D signals
            self.conv1 = nn.Conv1d(1, num_heads, kernel_size=7, padding=3)
            self.conv2 = nn.Conv1d(num_heads, 1, kernel_size=7, padding=3)
            
        else:
            raise ValueError(f"Unsupported attention type: {attention_type}")
            
    def forward(self, x):
        """
        Apply attention to input features.
        
        Args:
            x: Input tensor, shape depends on attention_type:
               - 'self': [batch_size, feature_dim]
               - 'channel': [batch_size, feature_dim]
               - 'spatial': [batch_size, channels, length]
               
        Returns:
            Tensor with same shape as input but with attention applied
        """
        if self.attention_type == 'self':
            batch_size = x.size(0)
            
            # Reshape for multi-head attention
            query = self.query(x).view(batch_size, -1, self.num_heads, self.attention_dim)
            key = self.key(x).view(batch_size, -1, self.num_heads, self.attention_dim)
            value = self.value(x).view(batch_size, -1, self.num_heads, self.attention_dim)
            
            # Transpose to [batch_size, num_heads, seq_len, attention_dim]
            query = query.transpose(1, 2)
            key = key.transpose(1, 2)
            value = value.transpose(1, 2)
            
            # Compute attention scores
            scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.attention_dim)
            attention_weights = F.softmax(scores, dim=-1)
            
            # Apply attention weights
            context = torch.matmul(attention_weights, value)
            
            # Reshape back
            context = context.transpose(1, 2).contiguous().view(batch_size, -1)
            
            # Output projection
            output = self.output(context)
            
            return output
            
        elif self.attention_type == 'channel':
            # Channel attention (squeeze and excitation)
            weights = F.relu(self.fc1(x))
            weights = torch.sigmoid(self.fc2(weights))
            
            # Apply channel-wise weights
            return x * weights
            
        elif self.attention_type == 'spatial':
            # Ensure input is 3D for Conv1d
            if x.dim() == 2:
                x = x.unsqueeze(1)  # [batch_size, 1, feature_dim]
                
            # Compute spatial attention weights
            weights = F.relu(self.conv1(x))
            weights = torch.sigmoid(self.conv2(weights))
            
            # Apply spatial attention
            return x * weights


class CrossPathwayAttention(nn.Module):
    def __init__(self, time_features, freq_features):
        """
        Cross-pathway attention for fusing time and frequency domain features.
        
        Args:
            time_features: Number of time domain features
            freq_features: Number of frequency domain features
        """
        super().__init__()
        
        # Attention mechanism for feature fusion
        self.attention = nn.Sequential(
            nn.Linear(time_features + freq_features, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Softmax(dim=1)
        )
        
    def forward(self, time_features, freq_features):
        """
        Apply cross-pathway attention to fuse time and frequency features.
        
        Args:
            time_features: Time domain features [batch_size, time_features]
            freq_features: Frequency domain features [batch_size, freq_features]
            
        Returns:
            Fused features [batch_size, time_features + freq_features]
        """
        batch_size = time_features.size(0)
        
        # Combine features for attention calculation
        combined_features = torch.cat([time_features, freq_features], dim=1)
        
        # Calculate attention weights
        attention_weights = self.attention(combined_features)
        
        # Apply attention weights
        time_weighted = time_features * attention_weights[:, 0].unsqueeze(1)
        freq_weighted = freq_features * attention_weights[:, 1].unsqueeze(1)
        
        # Recombine weighted features
        weighted_features = torch.cat([time_weighted, freq_weighted], dim=1)
        
        return weighted_features 