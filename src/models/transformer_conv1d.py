import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from src.config.config import config
from src.models import register_model
from src.models.base_model import BaseModel
from src.models.components import PeakOrderingLayer

@register_model("transformer_conv1d")
class TransformerConv1D(BaseModel):
    def __init__(self) -> None:
        super().__init__()
        
        # Initial convolution layer to reduce sequence length
        self.conv1 = nn.Conv1d(1, 32, kernel_size=7, padding=3, stride=2)
        self.pool1 = nn.MaxPool1d(2)
        
        # Second convolution to further reduce dimensionality
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2, stride=2)
        
        # Calculate sequence length after convolutions and pooling
        # Original: 2048 -> Conv1(stride 2): 1024 -> Pool1: 512 -> Conv2(stride 2): 256
        seq_len = 256
        embed_dim = 64
        
        # Position encoding
        self.pos_encoder = PositionalEncoding(embed_dim, dropout=0.1, max_len=seq_len)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=4, 
            dim_feedforward=128,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # Global attention pooling
        self.attention_pool = nn.Sequential(
            nn.Linear(embed_dim, 1),
            nn.Softmax(dim=1)
        )
        
        # Final MLP layers
        self.fc1 = nn.Linear(embed_dim, 128)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 64)
        self.output = nn.Linear(64, 3)
        
        # Peak ordering layer
        self.peak_ordering = PeakOrderingLayer(softness=0.1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        
        # Initial convolution feature extraction
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        
        # Reshape for transformer: [batch, channels, seq_len] -> [batch, seq_len, channels]
        x = x.transpose(1, 2)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Apply transformer encoder
        x = self.transformer_encoder(x)
        
        # Global attention pooling
        attention_weights = self.attention_pool(x)
        x = torch.sum(x * attention_weights, dim=1)
        
        # MLP prediction head
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.output(x)
        
        # Ensure peak1 < midpoint < peak2 using component
        x = self.peak_ordering(x)
        
        return x
        
        
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)