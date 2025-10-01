"""
KANFormer - Training and Evaluation Pipeline
============================================
Complete training, evaluation, and testing pipeline
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score, confusion_matrix,
                            classification_report)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import time
import os


class KANFormerTrainer:
    """Trainer class for KANFormer"""
    
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize trainer
        
        Parameters:
        -----------
        model : KANFormer
            Model to train
        device : str
            Device to use
        """
        self.model = model
        self.device = device
        self.model.to(device)
        
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
    def train_epoch(self, train_loader, optimizer, criterion, kg_embeddings=None):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            
            # Forward pass
            logits = self.model(batch_x, kg_embeddings)
            loss = criterion(logits, batch_y)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=1)
            correct += (predictions == batch_y).sum().item()
            total += batch_y.size(0)
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100 * correct / total
        
        return avg_loss, accuracy
    
    def validate(self, val_loader, criterion, kg_embeddings=None):
        """Validate model"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                # Forward pass
                logits = self.model(batch_x, kg_embeddings)
                loss = criterion(logits, batch_y)
                
                # Statistics
                total_loss += loss.item()
                predictions = torch.argmax(logits, dim=1)
                correct += (predictions == batch_y).sum().item()
                total += batch_y.size(0)
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100 * correct / total
        
        return avg_loss, accuracy
    
    def train(self, train_loader, val_loader, n_epochs=200, learning_rate=1e-4,
              kg_embeddings=None, patience=20, save_best=True, save_path='best_model.pth'):
        """
        Train model with early stopping
        
        Parameters:
        -----------
        train_loader : DataLoader
            Training data loader
        val_loader : DataLoader
            Validation data loader
        n_epochs : int
            Maximum number of epochs
        learning_rate : float
            Learning rate
        kg_embeddings : torch.Tensor or None
            Knowledge graph embeddings
        patience : int
            Early stopping patience
        save_best : bool
            Whether to save best model
        save_path : str
            Path to save best model
        """
        print("\n" + "="*70)
        print("TRAINING KANFORMER")
        print("="*70)
        print(f"Epochs: {n_epochs}")
        print(f"Learning rate: {learning_rate}")
        print(f"Early stopping patience: {patience}")
        print(f"Device: {self.device}")
        print("="*70 + "\n")
        
        # Move KG embeddings to device if provided
        if kg_embeddings is not None:
            kg_embeddings = kg_embeddings.to(self.device)
        
        # Loss function with class weights
        criterion = nn.CrossEntropyLoss()
        
        # Optimizer
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=1e-4
        )
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=n_epochs, eta_min=1e-6
        )
        
        # Early stopping
        best_val_loss = float('inf')
        patience_counter = 0
        
        start_time = time.time()
        
        for epoch in range(n_epochs):
            # Train
            train_loss, train_acc = self.train_epoch(
                train_loader, optimizer, criterion, kg_embeddings
            )
            
            # Validate
            val_loss, val_acc = self.validate(
                val_loader, criterion, kg_embeddings
            )
            
            # Update scheduler
            scheduler.step()
            
            # Save metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            
            # Print progress
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch+1}/{n_epochs}")
                print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
                print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
                print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
                print("-" * 50)
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                
                if save_best:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'train_loss': train_loss,
                        'val_loss': val_loss,
                        'val_accuracy': val_acc
                    }, save_path)
                    print(f"  ✓ Best model saved (Val Loss: {val_loss:.4f})")
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break
        
        training_time = time.time() - start_time
        
        print("\n" + "="*70)
        print(f"Training Complete!")
        print(f"Total time: {training_time/60:.2f} minutes")
        print(f"Best validation loss: {best_val_loss:.4f}")
        print("="*70 + "\n")
        
        # Load best model
        if save_best and os.path.exists(save_path):
            checkpoint = torch.load(save_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded best model from epoch {checkpoint['epoch']+1}")
    
    def plot_training_history(self, save_path='training_history.png'):
        """Plot training history"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        axes[0].plot(self.train_losses, label='Train Loss', linewidth=2)
        axes[0].plot(self.val_losses, label='Val Loss', linewidth=2)
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Loss', fontsize=12)
        axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        axes[0].legend(fontsize=11)
        axes[0].grid(True, alpha=0.3)
        
        # Accuracy plot
        axes[1].plot(self.train_accuracies, label='Train Acc', linewidth=2)
        axes[1].plot(self.val_accuracies, label='Val Acc', linewidth=2)
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Accuracy (%)', fontsize=12)
        axes[1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
        axes[1].legend(fontsize=11)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to {save_path}")
        plt.close()


class KANFormerEvaluator:
    """Evaluator class for comprehensive model assessment"""
    
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize evaluator
        
        Parameters:
        -----------
        model : KANFormer
            Trained model
        device : str
            Device to use
        """
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
    
    def predict(self, test_loader, kg_embeddings=None):
        """Get predictions on test set"""
        all_predictions = []
        all_labels = []
        all_probs = []
        
        if kg_embeddings is not None:
            kg_embeddings = kg_embeddings.to(self.device)
        
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(self.device)
                
                logits = self.model(batch_x, kg_embeddings)
                probs = torch.softmax(logits, dim=1)
                predictions = torch.argmax(probs, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(batch_y.numpy())
                all_probs.extend(probs.cpu().numpy())
        
        return np.array(all_predictions), np.array(all_labels), np.array(all_probs)
    
    def evaluate(self, test_loader, kg_embeddings=None, class_names=None):
        """
        Comprehensive evaluation
        
        Parameters:
        -----------
        test_loader : DataLoader
            Test data loader
        kg_embeddings : torch.Tensor or None
            Knowledge graph embeddings
        class_names : list or None
            Class names for display
        
        Returns:
        --------
        metrics : dict
            Dictionary of metrics
        """
        print("\n" + "="*70)
        print("EVALUATING KANFORMER")
        print("="*70)
        
        # Get predictions
        predictions, labels, probs = self.predict(test_loader, kg_embeddings)
        
        # Calculate metrics
        accuracy = accuracy_score(labels, predictions)
        precision = precision_score(labels, predictions, average='macro')
        recall = recall_score(labels, predictions, average='macro')
        f1 = f1_score(labels, predictions, average='macro')
        
        # Calculate AUC (one-vs-rest)
        try:
            auc = roc_auc_score(labels, probs, multi_class='ovr', average='macro')
        except:
            auc = None
        
        # Top-k accuracy
        top_k_acc = self._top_k_accuracy(probs, labels, k=3)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc': auc,
            'top_3_accuracy': top_k_acc
        }
        
        # Print results
        print(f"\nTest Set Results:")
        print(f"  Accuracy:      {accuracy*100:.2f}%")
        print(f"  Precision:     {precision:.4f}")
        print(f"  Recall:        {recall:.4f}")
        print(f"  F1-Score:      {f1:.4f}")
        if auc is not None:
            print(f"  AUC:           {auc:.4f}")
        print(f"  Top-3 Acc:     {top_k_acc*100:.2f}%")
        
        # Classification report
        print("\n" + "-"*70)
        print("Classification Report:")
        print("-"*70)
        if class_names is None:
            class_names = [f"Class_{i}" for i in range(len(np.unique(labels)))]
        print(classification_report(labels, predictions, target_names=class_names))
        
        # Confusion matrix
        cm = confusion_matrix(labels, predictions)
        
        print("="*70 + "\n")
        
        return metrics, cm, predictions, labels, probs
    
    def _top_k_accuracy(self, probs, labels, k=3):
        """Calculate top-k accuracy"""
        top_k_preds = np.argsort(probs, axis=1)[:, -k:]
        correct = 0
        for i, label in enumerate(labels):
            if label in top_k_preds[i]:
                correct += 1
        return correct / len(labels)
    
    def plot_confusion_matrix(self, cm, class_names=None, save_path='confusion_matrix.png'):
        """Plot confusion matrix"""
        if class_names is None:
            class_names = [f"Class_{i}" for i in range(cm.shape[0])]
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names,
                   cbar_kws={'label': 'Count'})
        plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
        plt.ylabel('True Label', fontsize=12, fontweight='bold')
        plt.title('Confusion Matrix - KANFormer', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
        plt.close()
    
    def plot_roc_curves(self, probs, labels, class_names=None, save_path='roc_curves.png'):
        """Plot ROC curves for each class"""
        from sklearn.preprocessing import label_binarize
        from sklearn.metrics import roc_curve, auc
        
        n_classes = probs.shape[1]
        
        if class_names is None:
            class_names = [f"Class_{i}" for i in range(n_classes)]
        
        # Binarize labels
        labels_bin = label_binarize(labels, classes=range(n_classes))
        
        plt.figure(figsize=(10, 8))
        
        # Plot ROC curve for each class
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(labels_bin[:, i], probs[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, linewidth=2, 
                    label=f'{class_names[i]} (AUC = {roc_auc:.3f})')
        
        # Plot diagonal
        plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Baseline')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
        plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
        plt.title('ROC Curves - Multi-Class Classification', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC curves saved to {save_path}")
        plt.close()


def create_data_loaders(X_train, X_val, X_test, y_train, y_val, y_test, 
                       batch_size=64):
    """Create PyTorch data loaders"""
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.LongTensor(y_train)
    
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.LongTensor(y_val)
    
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.LongTensor(y_test)
    
    # Create datasets
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    # Create loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                             shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                           shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, 
                            shuffle=False, num_workers=0)
    
    print(f"\nData Loaders Created:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    print(f"  Batch size: {batch_size}")
    
    return train_loader, val_loader, test_loader


# Usage example
if __name__ == "__main__":
    # This would be used with actual data from the dataset loader
    print("\n" + "="*70)
    print("TRAINING PIPELINE TEST")
    print("="*70)
    
    # Dummy data
    n_samples = 1000
    input_dim = 20
    n_classes = 4
    
    X_train = np.random.randn(int(n_samples * 0.64), input_dim)
    X_val = np.random.randn(int(n_samples * 0.16), input_dim)
    X_test = np.random.randn(int(n_samples * 0.20), input_dim)
    
    y_train = np.random.randint(0, n_classes, int(n_samples * 0.64))
    y_val = np.random.randint(0, n_classes, int(n_samples * 0.16))
    y_test = np.random.randint(0, n_classes, int(n_samples * 0.20))
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        X_train, X_val, X_test, y_train, y_val, y_test, batch_size=32
    )
    
    # Create model (import from kanformer_model.py)
    from kanformer_model import create_kanformer
    model = create_kanformer(input_dim, n_classes)
    
    # Create trainer
    trainer = KANFormerTrainer(model)
    
    # Train (reduced epochs for demo)
    print("\nStarting training (demo with 20 epochs)...")
    trainer.train(train_loader, val_loader, n_epochs=20, patience=10)
    
    # Plot training history
    trainer.plot_training_history('demo_training_history.png')
    
    # Evaluate
    evaluator = KANFormerEvaluator(model)
    class_names = ['Science', 'Arts', 'Business', 'Engineering']
    metrics, cm, preds, labels, probs = evaluator.evaluate(
        test_loader, class_names=class_names
    )
    
    # Plot confusion matrix
    evaluator.plot_confusion_matrix(cm, class_names, 'demo_confusion_matrix.png')
    
    # Plot ROC curves
    evaluator.plot_roc_curves(probs, labels, class_names, 'demo_roc_curves.png')
    
    print("\n" + "="*70)
    print("Training pipeline test complete!")
    print("="*70)
