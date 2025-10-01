
"""
KANFormer - Knowledge-Aware Transformer Model
=============================================
Complete implementation of the KANFormer architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class CNNFeatureExtractor(nn.Module):
    """1D CNN for local feature extraction"""
    
    def __init__(self, input_dim, n_filters=128, kernel_size=3):
        super(CNNFeatureExtractor, self).__init__()
        
        self.conv1 = nn.Conv1d(1, n_filters, kernel_size, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(n_filters)
        self.pool1 = nn.MaxPool1d(2)
        
        self.conv2 = nn.Conv1d(n_filters, n_filters, kernel_size, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(n_filters)
        self.pool2 = nn.AdaptiveMaxPool1d(1)
        
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        """
        Forward pass
        
        Parameters:
        -----------
        x : torch.Tensor [batch_size, input_dim]
        
        Returns:
        --------
        out : torch.Tensor [batch_size, n_filters]
        """
        # Add channel dimension: [batch, 1, features]
        x = x.unsqueeze(1)
        
        # First conv block
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.pool1(out)
        out = self.dropout(out)
        
        # Second conv block
        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)
        out = self.pool2(out)
        
        # Flatten
        out = out.squeeze(-1)
        
        return out


class BiGRUEncoder(nn.Module):
    """Bidirectional GRU for temporal modeling"""
    
    def __init__(self, input_dim, hidden_dim=64, num_layers=2):
        super(BiGRUEncoder, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.3 if num_layers > 1 else 0
        )
        
    def forward(self, x):
        """
        Forward pass
        
        Parameters:
        -----------
        x : torch.Tensor [batch_size, input_dim]
        
        Returns:
        --------
        out : torch.Tensor [batch_size, hidden_dim * 2]
        """
        # Add sequence dimension: [batch, seq_len=1, features]
        x = x.unsqueeze(1)
        
        # GRU forward
        out, hidden = self.gru(x)
        
        # Take last hidden state from both directions
        # hidden shape: [num_layers * 2, batch, hidden_dim]
        forward_hidden = hidden[-2, :, :]
        backward_hidden = hidden[-1, :, :]
        
        # Concatenate forward and backward
        out = torch.cat([forward_hidden, backward_hidden], dim=1)
        
        return out


class KnowledgeAwareAttention(nn.Module):
    """Knowledge-aware multi-head attention mechanism"""
    
    def __init__(self, d_model, n_heads=8, dropout=0.3):
        super(KnowledgeAwareAttention, self).__init__()
        
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # Query, Key, Value projections
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        
        # Knowledge graph key projection
        self.W_kg = nn.Linear(d_model, d_model)
        
        # Output projection
        self.W_o = nn.Linear(d_model, d_model)
        
        # Lambda parameter for knowledge weighting
        self.lambda_kg = nn.Parameter(torch.tensor(0.42))
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x, kg_embeddings=None):
        """
        Forward pass
        
        Parameters:
        -----------
        x : torch.Tensor [batch_size, d_model]
            Input features
        kg_embeddings : torch.Tensor [n_features, d_model]
            Knowledge graph embeddings
        
        Returns:
        --------
        out : torch.Tensor [batch_size, d_model]
        """
        batch_size = x.size(0)
        
        # Add sequence dimension
        x = x.unsqueeze(1)  # [batch, 1, d_model]
        
        # Linear projections
        Q = self.W_q(x)  # [batch, 1, d_model]
        K = self.W_k(x)  # [batch, 1, d_model]
        V = self.W_v(x)  # [batch, 1, d_model]
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, 1, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, 1, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, 1, self.n_heads, self.d_k).transpose(1, 2)
        
        # Standard attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Add knowledge graph attention if available
        if kg_embeddings is not None:
            # Project KG embeddings
            K_kg = self.W_kg(kg_embeddings)  # [n_features, d_model]
            K_kg = K_kg.view(1, kg_embeddings.size(0), self.n_heads, self.d_k)
            K_kg = K_kg.transpose(1, 2)  # [1, n_heads, n_features, d_k]
            
            # Expand for batch
            K_kg = K_kg.expand(batch_size, -1, -1, -1)
            
            # Compute KG attention scores
            kg_scores = torch.matmul(Q, K_kg.transpose(-2, -1)) / math.sqrt(self.d_k)
            kg_scores = kg_scores.mean(dim=-1, keepdim=True)  # Average over features
            
            # Combine standard and KG attention
            scores = scores + self.lambda_kg * kg_scores
        
        # Apply softmax
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        out = torch.matmul(attention_weights, V)
        
        # Reshape back
        out = out.transpose(1, 2).contiguous()
        out = out.view(batch_size, 1, self.d_model)
        
        # Output projection
        out = self.W_o(out)
        out = out.squeeze(1)  # [batch, d_model]
        
        # Residual connection and layer norm
        out = self.layer_norm(x.squeeze(1) + self.dropout(out))
        
        return out


class PositionwiseFeedForward(nn.Module):
    """Position-wise feed-forward network"""
    
    def __init__(self, d_model, d_ff=2048, dropout=0.3):
        super(PositionwiseFeedForward, self).__init__()
        
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        """
        Forward pass
        
        Parameters:
        -----------
        x : torch.Tensor [batch_size, d_model]
        
        Returns:
        --------
        out : torch.Tensor [batch_size, d_model]
        """
        residual = x
        
        out = self.w_1(x)
        out = F.relu(out)
        out = self.dropout(out)
        out = self.w_2(out)
        out = self.dropout(out)
        
        # Residual connection and layer norm
        out = self.layer_norm(residual + out)
        
        return out


class KANFormer(nn.Module):
    """
    Knowledge-Aware Transformer for Academic Orientation Prediction
    """
    
    def __init__(self, input_dim, n_classes, d_model=256, n_heads=8, 
                 n_cnn_filters=128, n_gru_hidden=64, dropout=0.3):
        """
        Initialize KANFormer
        
        Parameters:
        -----------
        input_dim : int
            Number of input features
        n_classes : int
            Number of output classes
        d_model : int
            Model dimension
        n_heads : int
            Number of attention heads
        n_cnn_filters : int
            Number of CNN filters
        n_gru_hidden : int
            GRU hidden dimension
        dropout : float
            Dropout rate
        """
        super(KANFormer, self).__init__()
        
        self.input_dim = input_dim
        self.n_classes = n_classes
        self.d_model = d_model
        
        # Feature extraction modules
        self.cnn_extractor = CNNFeatureExtractor(input_dim, n_cnn_filters)
        self.gru_encoder = BiGRUEncoder(input_dim, n_gru_hidden)
        
        # Projection to common dimension
        cnn_out_dim = n_cnn_filters
        gru_out_dim = n_gru_hidden * 2  # Bidirectional
        combined_dim = cnn_out_dim + gru_out_dim
        
        self.projection = nn.Linear(combined_dim, d_model)
        
        # Embedding layer for input features
        self.feature_embedding = nn.Linear(input_dim, d_model)
        
        # Knowledge-aware attention
        self.attention = KnowledgeAwareAttention(d_model, n_heads, dropout)
        
        # Feed-forward network
        self.ffn = PositionwiseFeedForward(d_model, d_model * 4, dropout)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_classes)
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, kg_embeddings=None):
        """
        Forward pass
        
        Parameters:
        -----------
        x : torch.Tensor [batch_size, input_dim]
            Input features
        kg_embeddings : torch.Tensor [n_features, d_model] or None
            Knowledge graph embeddings
        
        Returns:
        --------
        logits : torch.Tensor [batch_size, n_classes]
            Class logits
        """
        # Feature extraction pathways
        cnn_features = self.cnn_extractor(x)
        gru_features = self.gru_encoder(x)
        
        # Concatenate CNN and GRU features
        combined_features = torch.cat([cnn_features, gru_features], dim=1)
        
        # Project to model dimension
        h = self.projection(combined_features)
        h = self.dropout(h)
        
        # Knowledge-aware attention
        h = self.attention(h, kg_embeddings)
        
        # Feed-forward network
        h = self.ffn(h)
        
        # Classification
        logits = self.classifier(h)
        
        return logits
    
    def get_attention_weights(self, x, kg_embeddings=None):
        """Get attention weights for interpretability"""
        # This is a simplified version for visualization
        # In practice, you'd need to modify the attention module to return weights
        with torch.no_grad():
            cnn_features = self.cnn_extractor(x)
            gru_features = self.gru_encoder(x)
            combined_features = torch.cat([cnn_features, gru_features], dim=1)
            h = self.projection(combined_features)
            
            # Get attention weights (simplified)
            # In full implementation, modify KnowledgeAwareAttention to return weights
            return None  # Placeholder


# Model initialization helper
def create_kanformer(input_dim, n_classes, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Create and initialize KANFormer model
    
    Parameters:
    -----------
    input_dim : int
        Number of input features
    n_classes : int
        Number of classes
    device : str
        Device to use
    
    Returns:
    --------
    model : KANFormer
        Initialized model
    """
    model = KANFormer(
        input_dim=input_dim,
        n_classes=n_classes,
        d_model=256,
        n_heads=8,
        n_cnn_filters=128,
        n_gru_hidden=64,
        dropout=0.3
    )
    
    model = model.to(device)
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nKANFormer Model Created:")
    print(f"  Input dimension: {input_dim}")
    print(f"  Number of classes: {n_classes}")
    print(f"  Total parameters: {n_params:,}")
    print(f"  Device: {device}")
    
    return model


# Usage example
if __name__ == "__main__":
    # Test model
    print("\n" + "="*70)
    print("KANFORMER MODEL TEST")
    print("="*70)
    
    # Dummy data
    batch_size = 32
    input_dim = 20
    n_classes = 4
    
    # Create model
    model = create_kanformer(input_dim, n_classes)
    
    # Test forward pass
    x = torch.randn(batch_size, input_dim)
    
    # Without KG embeddings
    print("\nTesting forward pass without KG embeddings...")
    logits = model(x)
    print(f"  Output shape: {logits.shape}")
    print(f"  Expected: [{batch_size}, {n_classes}]")
    
    # With KG embeddings
    print("\nTesting forward pass with KG embeddings...")
    kg_emb = torch.randn(input_dim, 256)  # d_model = 256
    logits = model(x, kg_emb)
    print(f"  Output shape: {logits.shape}")
    
    # Test predictions
    probs = F.softmax(logits, dim=1)
    predictions = torch.argmax(probs, dim=1)
    print(f"\nSample predictions: {predictions[:10]}")
    
    print("\n" + "="*70)
    print("Model test complete!")
    print("="*70)
