import torch
import numpy as np
import pandas as pd
import seaborn as sns
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split


# -------------------------
# 1. Dataset Definition
# -------------------------
class MultimodalEmotionDataset(Dataset):
    def __init__(self, text_embs, video_embs, labels):
        self.text_embs = torch.from_numpy(text_embs).float()
        self.video_embs = torch.from_numpy(video_embs).float()
        self.labels = torch.from_numpy(labels).long()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "text": self.text_embs[idx],
            "video": self.video_embs[idx],
            "label": self.labels[idx],
        }


# -------------------------
# 2. Model Definition - Improved
# -------------------------
class ImprovedCollaborativeGatingModel(nn.Module):
    def __init__(
        self,
        dim_text=100,
        dim_video=2048,
        proj_dim=512,  # Increased from 256
        hidden_dim=512,  # Increased from 128
        num_classes=4,
        dropout_rate=0.2,  # Reduced dropout
    ):
        super().__init__()
        
        # Projection layers with batch norm
        self.proj_text = nn.Sequential(
            nn.Linear(dim_text, proj_dim),
            nn.BatchNorm1d(proj_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        self.proj_video = nn.Sequential(
            nn.Linear(dim_video, proj_dim),
            nn.BatchNorm1d(proj_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Pairwise scoring with residual connections
        self.g_text_video = nn.Sequential(
            nn.Linear(2 * proj_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, proj_dim),
        )
        
        self.g_video_text = nn.Sequential(
            nn.Linear(2 * proj_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, proj_dim),
        )
        
        # Attention transform with better regularization
        self.h_text = nn.Sequential(
            nn.Linear(proj_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, proj_dim),
        )
        
        self.h_video = nn.Sequential(
            nn.Linear(proj_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, proj_dim),
        )
        
        # Improved classification head
        self.classifier = nn.Sequential(
            nn.Linear(2 * proj_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, num_classes),
        )
        
        # Additional modality-specific classifiers for auxiliary losses
        self.text_classifier = nn.Linear(proj_dim, num_classes)
        self.video_classifier = nn.Linear(proj_dim, num_classes)

    def forward(self, text_emb, video_emb):
        # 1) Project and normalize
        psi_t = self.proj_text(text_emb)
        psi_v = self.proj_video(video_emb)
        
        # 2) Pairwise scoring with residual connections
        m_t = self.g_text_video(torch.cat([psi_t, psi_v], dim=1)) + psi_t  # Residual connection
        m_v = self.g_video_text(torch.cat([psi_v, psi_t], dim=1)) + psi_v  # Residual connection
        
        # 3) Attention vectors
        T_t = self.h_text(m_t)
        T_v = self.h_video(m_v)
        
        # 4) Gating with scaled sigmoid to prevent vanishing gradients
        sigma_t = 2 * torch.sigmoid(T_t)  # Scale to (0,2) range
        sigma_v = 2 * torch.sigmoid(T_v)  # Scale to (0,2) range
        tilde_t = psi_t * sigma_t
        tilde_v = psi_v * sigma_v
        
        # 5) Additional auxiliary predictions for regularization
        aux_text_logits = self.text_classifier(tilde_t)
        aux_video_logits = self.video_classifier(tilde_v)
        
        # 6) Fuse & classify
        fused = torch.cat([tilde_t, tilde_v], dim=1)
        main_logits = self.classifier(fused)
        
        return main_logits, aux_text_logits, aux_video_logits


# -------------------------
# 3. Training & Evaluation - Improved
# -------------------------
def mixup_data(x_t, x_v, y, alpha=0.2, device='cpu'):
    """Apply mixup augmentation to the batch data"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x_t.size()[0]
    index = torch.randperm(batch_size).to(device)

    mixed_x_t = lam * x_t + (1 - lam) * x_t[index, :]
    mixed_x_v = lam * x_v + (1 - lam) * x_v[index, :]
    y_a, y_b = y, y[index]
    
    return mixed_x_t, mixed_x_v, y_a, y_b, lam

def train_epoch(model, loader, optimizer, criterion, device, use_mixup=True, aux_weight=0.2):
    model.train()
    total_loss = 0.0
    
    for batch in loader:
        optimizer.zero_grad()
        x_t = batch["text"].to(device)
        x_v = batch["video"].to(device)
        y = batch["label"].to(device)
        
        if use_mixup:
            # Apply mixup data augmentation
            mixed_x_t, mixed_x_v, y_a, y_b, lam = mixup_data(x_t, x_v, y, alpha=0.2, device=device)
            
            # Forward pass with mixed inputs
            main_out, aux_t_out, aux_v_out = model(mixed_x_t, mixed_x_v)
            
            # Compute mixed loss
            main_loss = lam * criterion(main_out, y_a) + (1 - lam) * criterion(main_out, y_b)
            aux_t_loss = lam * criterion(aux_t_out, y_a) + (1 - lam) * criterion(aux_t_out, y_b)
            aux_v_loss = lam * criterion(aux_v_out, y_a) + (1 - lam) * criterion(aux_v_out, y_b)
        else:
            # Regular forward pass
            main_out, aux_t_out, aux_v_out = model(x_t, x_v)
            
            # Compute standard loss
            main_loss = criterion(main_out, y)
            aux_t_loss = criterion(aux_t_out, y)
            aux_v_loss = criterion(aux_v_out, y)
        
        # Combined loss with auxiliary tasks
        loss = main_loss + aux_weight * (aux_t_loss + aux_v_loss)
        
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        total_loss += loss.item()
        
    return total_loss / len(loader)

def eval_model(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for batch in loader:
            x_t = batch["text"].to(device)
            x_v = batch["video"].to(device)
            y = batch["label"].to(device)
            
            main_out, _, _ = model(x_t, x_v)
            loss = criterion(main_out, y)
            total_loss += loss.item()
            
            pred = torch.argmax(main_out, dim=1)
            all_preds.append(pred.cpu().numpy())
            all_labels.append(y.cpu().numpy())
    
    val_loss = total_loss / len(loader)
    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    
    accuracy = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='weighted')
    
    return preds, labels, accuracy, f1, val_loss


# -------------------------
# 4. Main Script - Improved
# -------------------------
if __name__ == "__main__":
    # Hyperparams - Improved
    BATCH_SIZE = 32
    LR = 3e-4  # Reduced from 1e-3
    EPOCHS = 50  # Increased from 10
    PATIENCE = 10  # For early stopping
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    # Load your pre-split arrays
    print("Loading data...")
    df1 = pd.read_csv("Embeddings/Text/train_emb.csv")
    text_train = df1.values

    df2 = pd.read_csv("Embeddings/Videos/train_final_emb.csv")
    video_train = df2.values

    df3 = pd.read_csv("Embeddings/Labels/train_y_labels.csv")
    labels_train = df3.values.squeeze()

    df4 = pd.read_csv("Embeddings/Text/test_emb1.csv")
    text_test = df4.values

    df5 = pd.read_csv("Embeddings/Videos/test_final_emb.csv")
    video_test = df5.values

    df6 = pd.read_csv("Embeddings/Labels/test_y_labels1.csv")
    labels_test = df6.values.squeeze()
    
    # Create validation set from training data
    print("Creating validation split...")
    text_train_final, text_val, video_train_final, video_val, labels_train_final, labels_val = train_test_split(
        text_train, video_train, labels_train, test_size=0.15, random_state=42, stratify=labels_train
    )
    
    # Check class distribution
    print("Class distribution:")
    print(f"Train: {np.bincount(labels_train_final)}")
    print(f"Val: {np.bincount(labels_val)}")
    print(f"Test: {np.bincount(labels_test)}")
    
    # Normalize embedding features
    print("Normalizing features...")
    # Simple standardization for text embeddings
    text_mean = np.mean(text_train_final, axis=0, keepdims=True)
    text_std = np.std(text_train_final, axis=0, keepdims=True) + 1e-6
    text_train_final = (text_train_final - text_mean) / text_std
    text_val = (text_val - text_mean) / text_std
    text_test = (text_test - text_mean) / text_std
    
    # Simple standardization for video embeddings
    video_mean = np.mean(video_train_final, axis=0, keepdims=True)
    video_std = np.std(video_train_final, axis=0, keepdims=True) + 1e-6
    video_train_final = (video_train_final - video_mean) / video_std
    video_val = (video_val - video_mean) / video_std
    video_test = (video_test - video_mean) / video_std
    
    print("Setting up datasets and loaders...")
    # Datasets & loaders
    train_ds = MultimodalEmotionDataset(text_train_final, video_train_final, labels_train_final)
    val_ds = MultimodalEmotionDataset(text_val, video_val, labels_val)
    test_ds = MultimodalEmotionDataset(text_test, video_test, labels_test)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    # Calculate class weights for handling imbalance
    class_counts = np.bincount(labels_train_final)
    print(f"Computing class weights from distribution: {class_counts}")
    class_weights = torch.FloatTensor(1.0 / (class_counts + 1e-5))
    class_weights = class_weights / class_weights.sum() * len(class_weights)
    print(f"Class weights: {class_weights}")
    
    # Model, optimizer, loss
    print("Initializing model...")
    model = ImprovedCollaborativeGatingModel().to(DEVICE)
    
    # Use weighted loss for imbalanced classes
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))
    
    # Initialize optimizer with weight decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6
    )
    
    # Setup for early stopping
    best_val_f1 = 0
    patience_counter = 0
    best_model_state = None
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_acc': [],
        'val_f1': []
    }
    
    # Train with early stopping
    print("Starting training...")
    for epoch in range(1, EPOCHS + 1):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, DEVICE)
        
        # Evaluate on validation set
        _, _, val_acc, val_f1, val_loss = eval_model(model, val_loader, DEVICE)
        
        # Update learning rate based on validation loss
        scheduler.step(val_loss)
        
        # Record history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        
        # Print progress
        print(f"Epoch {epoch}/{EPOCHS} — Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}, Val acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")
        
        # Check for improvement
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            print(f"✓ New best model saved (F1: {val_f1:.4f})")
        else:
            patience_counter += 1
            print(f"× No improvement for {patience_counter} epochs")
            
        # Early stopping check
        if patience_counter >= PATIENCE:
            print(f"Early stopping triggered after {epoch} epochs")
            break
    
    # Load best model
    print("Loading best model for final evaluation...")
    model.load_state_dict(best_model_state)
    
    # Final evaluation on test set
    test_preds, test_labels, test_acc, test_f1, _ = eval_model(model, test_loader, DEVICE)
    
    print(f"Final Test Results:")
    print(f"Accuracy: {test_acc:.4f}")
    print(f"F1 Score (weighted): {test_f1:.4f}")
    
    # Classification report
    target_names = ["anger", "surprise", "ridicule", "sad"]
    print("\nClassification Report:")
    print(classification_report(test_labels, test_preds, target_names=target_names))
    
    # Confusion matrix plotting
    cm = confusion_matrix(test_labels, test_preds)
    df_cm = pd.DataFrame(cm, index=target_names, columns=target_names)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(df_cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig("Models/Model_1/outputs/Iter_1/Confusion_Matrix.png")
    
    # Save confusion matrix
    df_cm.to_csv("Models/Model_1/outputs/Iter_1/confusion_matrix.csv")
    
    # Predictions file
    df_out = pd.DataFrame({"true": test_labels, "pred": test_preds})
    df_out.to_csv("Models/Model_1/outputs/Iter_1/test_predictions.csv", index=False)
    
    # Plot training history
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Loss History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['val_acc'], label='Accuracy')
    plt.plot(history['val_f1'], label='F1 Score')
    plt.title('Validation Metrics')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("Models/Model_1/outputs/Iter_1/Training_History.png")
    
    print("Done! All outputs saved.")

# _____________________________________________________________________________
