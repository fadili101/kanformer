"""
KANFormer - Baseline Models Comparison
======================================
Compare KANFormer with classical ML and DL baselines
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score)
from xgboost import XGBClassifier
import time
import pandas as pd
from tqdm import tqdm


# =====================================================================
# CLASSICAL MACHINE LEARNING MODELS
# =====================================================================

def train_random_forest(X_train, y_train, X_test, y_test):
    """Train and evaluate Random Forest"""
    print("\nTraining Random Forest...")
    start_time = time.time()
    
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    train_time = time.time() - start_time
    
    # Predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    
    return compute_metrics(y_test, y_pred, y_proba, train_time)


def train_xgboost(X_train, y_train, X_test, y_test):
    """Train and evaluate XGBoost"""
    print("\nTraining XGBoost...")
    start_time = time.time()
    
    model = XGBClassifier(
        n_estimators=150,
        learning_rate=0.1,
        max_depth=10,
        random_state=42,
        n_jobs=-1,
        use_label_encoder=False,
        eval_metric='mlogloss'
    )
    model.fit(X_train, y_train)
    
    train_time = time.time() - start_time
    
    # Predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    
    return compute_metrics(y_test, y_pred, y_proba, train_time)


def train_svm(X_train, y_train, X_test, y_test):
    """Train and evaluate SVM"""
    print("\nTraining SVM...")
    start_time = time.time()
    
    model = SVC(
        C=10,
        kernel='rbf',
        gamma=0.001,
        random_state=42,
        probability=True
    )
    model.fit(X_train, y_train)
    
    train_time = time.time() - start_time
    
    # Predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    
    return compute_metrics(y_test, y_pred, y_proba, train_time)


# =====================================================================
# DEEP LEARNING BASELINE MODELS
# =====================================================================

class SimpleANN(nn.Module):
    """Simple Artificial Neural Network"""
    def __init__(self, input_dim, n_classes):
        super(SimpleANN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, n_classes)
        )
    
    def forward(self, x):
        return self.network(x)


class SimpleCNN(nn.Module):
    """Simple 1D CNN"""
    def __init__(self, input_dim, n_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(128, 128, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, n_classes)
        )
    
    def forward(self, x):
        x = x.unsqueeze(1)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = self.pool(x).squeeze(-1)
        return self.fc(x)


class SimpleLSTM(nn.Module):
    """Simple LSTM"""
    def __init__(self, input_dim, n_classes, hidden_dim=128):
        super(SimpleLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=2, 
                           batch_first=True, dropout=0.3)
        self.fc = nn.Linear(hidden_dim, n_classes)
    
    def forward(self, x):
        x = x.unsqueeze(1)
        _, (hidden, _) = self.lstm(x)
        return self.fc(hidden[-1])


class SimpleBiLSTM(nn.Module):
    """Simple Bidirectional LSTM"""
    def __init__(self, input_dim, n_classes, hidden_dim=128):
        super(SimpleBiLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=2,
                           batch_first=True, bidirectional=True, dropout=0.3)
        self.fc = nn.Linear(hidden_dim * 2, n_classes)
    
    def forward(self, x):
        x = x.unsqueeze(1)
        _, (hidden, _) = self.lstm(x)
        # Concatenate forward and backward
        hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        return self.fc(hidden)


class AttentionLSTM(nn.Module):
    """LSTM with Attention"""
    def __init__(self, input_dim, n_classes, hidden_dim=128):
        super(AttentionLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=2,
                           batch_first=True, dropout=0.3)
        self.attention = nn.Linear(hidden_dim, 1)
        self.fc = nn.Linear(hidden_dim, n_classes)
    
    def forward(self, x):
        x = x.unsqueeze(1)
        lstm_out, _ = self.lstm(x)
        
        # Attention weights
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
        
        # Weighted sum
        context = torch.sum(attn_weights * lstm_out, dim=1)
        
        return self.fc(context)


def train_dl_model(model, train_loader, val_loader, test_loader, 
                   n_epochs=100, device='cuda'):
    """Generic training function for DL models"""
    
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=10, factor=0.5
    )
    
    start_time = time.time()
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 15
    
    for epoch in range(n_epochs):
        # Training
        model.train()
        train_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
    
    train_time = time.time() - start_time
    
    # Test evaluation
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            outputs = model(batch_x)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_y.numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    return compute_metrics(all_labels, all_preds, all_probs, train_time)


# =====================================================================
# METRICS COMPUTATION
# =====================================================================

def compute_metrics(y_true, y_pred, y_proba, train_time):
    """Compute all evaluation metrics"""
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    # AUC
    try:
        auc = roc_auc_score(y_true, y_proba, multi_class='ovr', average='macro')
    except:
        auc = None
    
    return {
        'accuracy': accuracy * 100,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc': auc,
        'train_time': train_time
    }


# =====================================================================
# MAIN COMPARISON FUNCTION
# =====================================================================

def compare_all_models(X_train, X_val, X_test, y_train, y_val, y_test,
                       input_dim, n_classes, device='cuda'):
    """
    Compare all baseline models
    
    Returns:
    --------
    results_df : pandas.DataFrame
        Comparison results
    """
    
    print("\n" + "="*70)
    print("BASELINE MODELS COMPARISON")
    print("="*70)
    
    results = {}
    
    # ===== CLASSICAL ML MODELS =====
    print("\n" + "-"*70)
    print("CLASSICAL MACHINE LEARNING MODELS")
    print("-"*70)
    
    # Random Forest
    results['Random Forest'] = train_random_forest(X_train, y_train, X_test, y_test)
    
    # XGBoost
    results['XGBoost'] = train_xgboost(X_train, y_train, X_test, y_test)
    
    # SVM
    results['SVM'] = train_svm(X_train, y_train, X_test, y_test)
    
    # ===== DEEP LEARNING MODELS =====
    print("\n" + "-"*70)
    print("DEEP LEARNING MODELS")
    print("-"*70)
    
    # Prepare data loaders
    from training_pipeline import create_data_loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        X_train, X_val, X_test, y_train, y_val, y_test, batch_size=64
    )
    
    # ANN
    print("\nTraining ANN...")
    ann = SimpleANN(input_dim, n_classes)
    results['ANN'] = train_dl_model(ann, train_loader, val_loader, test_loader,
                                    n_epochs=100, device=device)
    
    # CNN
    print("\nTraining CNN...")
    cnn = SimpleCNN(input_dim, n_classes)
    results['CNN'] = train_dl_model(cnn, train_loader, val_loader, test_loader,
                                    n_epochs=100, device=device)
    
    # LSTM
    print("\nTraining LSTM...")
    lstm = SimpleLSTM(input_dim, n_classes)
    results['LSTM'] = train_dl_model(lstm, train_loader, val_loader, test_loader,
                                     n_epochs=100, device=device)
    
    # BiLSTM
    print("\nTraining BiLSTM...")
    bilstm = SimpleBiLSTM(input_dim, n_classes)
    results['BiLSTM'] = train_dl_model(bilstm, train_loader, val_loader, test_loader,
                                       n_epochs=100, device=device)
    
    # Attention-LSTM
    print("\nTraining Attention-LSTM...")
    att_lstm = AttentionLSTM(input_dim, n_classes)
    results['Att-LSTM'] = train_dl_model(att_lstm, train_loader, val_loader, test_loader,
                                         n_epochs=100, device=device)
    
    # ===== CREATE RESULTS DATAFRAME =====
    results_df = pd.DataFrame(results).T
    results_df = results_df.round(4)
    
    # Sort by accuracy
    results_df = results_df.sort_values('accuracy', ascending=False)
    
    print("\n" + "="*70)
    print("COMPARISON RESULTS")
    print("="*70)
    print(results_df.to_string())
    print("="*70)
    
    # Save to CSV
    results_df.to_csv('baseline_comparison_results.csv')
    print("\nResults saved to baseline_comparison_results.csv")
    
    return results_df


def visualize_comparison(results_df, save_path='baseline_comparison.png'):
    """Visualize comparison results"""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    metrics = ['accuracy', 'f1_score', 'auc', 'train_time']
    titles = ['Accuracy (%)', 'F1-Score', 'AUC', 'Training Time (s)']
    
    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[idx // 2, idx % 2]
        
        if metric in results_df.columns:
            data = results_df[metric].dropna()
            data.plot(kind='barh', ax=ax, color='steelblue')
            ax.set_xlabel(title, fontsize=12, fontweight='bold')
            ax.set_ylabel('Model', fontsize=12, fontweight='bold')
            ax.set_title(f'Comparison: {title}', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='x')
            
            # Add value labels
            for i, v in enumerate(data):
                ax.text(v, i, f' {v:.2f}', va='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nComparison plot saved to {save_path}")
    plt.close()


# =====================================================================
# USAGE EXAMPLE
# =====================================================================

if __name__ == "__main__":
    from dataset_preparation import DatasetLoader
    
    print("\n" + "="*70)
    print("BASELINE COMPARISON TEST")
    print("="*70)
    
    # Load data
    loader = DatasetLoader()
    X, y = loader.create_synthetic_dataset(n_samples=2000)
    
    # Preprocess
    X_train, X_val, X_test, y_train, y_val, y_test = loader.preprocess_data(
        X, y, test_size=0.2, apply_smote=True
    )
    
    input_dim = X_train.shape[1]
    n_classes = len(np.unique(y_train))
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Compare all models
    results_df = compare_all_models(
        X_train, X_val, X_test, y_train, y_val, y_test,
        input_dim, n_classes, device=device
    )
    
    # Visualize
    visualize_comparison(results_df)
    
    print("\n" + "="*70)
    print("Baseline comparison complete!")
    print("="*70)
