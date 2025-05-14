import os
import torch
import argparse
import numpy as np
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader, random_split

# ─── HYPERPARAMETERS ─────────────────────────────────────────────────────────
SEED            = 42
TEXT_DIM        = 100
VIDEO_DIM       = 2048
REDUCED_TEXT_DIM= 64
REDUCED_VIDEO_DIM= 256
TEXT_HID_DIM    = 32
VIDEO_HID_DIM   = 64
FUSION_OUT_DIM  = 32
DROPOUT_RATE    = 0.6
WEIGHT_DECAY    = 5e-3
BATCH_SIZE      = 16
LR              = 5e-4
PATIENCE        = 7
MAX_EPOCHS      = 50
STRATIFY_VAL    = True

# Set seed for reproducibility
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ─── DATA PREPROCESSING ───────────────────────────────────────────────────

def preprocess_features(train_text_path, val_text_path, test_text_path,
                        train_video_path, val_video_path, test_video_path):
    train_text = pd.read_csv(train_text_path).values
    val_text = pd.read_csv(val_text_path).values
    test_text = pd.read_csv(test_text_path).values

    train_video = pd.read_csv(train_video_path).values
    val_video = pd.read_csv(val_video_path).values
    test_video = pd.read_csv(test_video_path).values

    text_scaler = StandardScaler().fit(train_text)
    video_scaler = StandardScaler().fit(train_video)

    train_text_scaled = text_scaler.transform(train_text)
    val_text_scaled = text_scaler.transform(val_text)
    test_text_scaled = text_scaler.transform(test_text)

    train_video_scaled = video_scaler.transform(train_video)
    val_video_scaled = video_scaler.transform(val_video)
    test_video_scaled = video_scaler.transform(test_video)

    text_pca = PCA(n_components=REDUCED_TEXT_DIM).fit(train_text_scaled)
    video_pca = PCA(n_components=REDUCED_VIDEO_DIM).fit(train_video_scaled)

    train_text_reduced = text_pca.transform(train_text_scaled)
    val_text_reduced = text_pca.transform(val_text_scaled)
    test_text_reduced = text_pca.transform(test_text_scaled)

    train_video_reduced = video_pca.transform(train_video_scaled)
    val_video_reduced = video_pca.transform(val_video_scaled)
    test_video_reduced = video_pca.transform(test_video_scaled)

    print(f"Text features reduced from {train_text.shape[1]} to {train_text_reduced.shape[1]}")
    print(f"Video features reduced from {train_video.shape[1]} to {train_video_reduced.shape[1]}")
    print(f"Text PCA explained variance: {sum(text_pca.explained_variance_ratio_):.2f}")
    print(f"Video PCA explained variance: {sum(video_pca.explained_variance_ratio_):.2f}")

    return (train_text_reduced, val_text_reduced, test_text_reduced,
            train_video_reduced, val_video_reduced, test_video_reduced)

# ─── DATASET ─────────────────────────────────────────────────

class MultimodalDataset(Dataset):
    def __init__(self, text_features, video_features, labels=None):
        self.text_features = torch.tensor(text_features, dtype=torch.float32)
        self.video_features = torch.tensor(video_features, dtype=torch.float32)
        self.has_labels = labels is not None
        if self.has_labels:
            self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.text_features)

    def __getitem__(self, idx):
        if self.has_labels:
            return self.text_features[idx], self.video_features[idx], self.labels[idx]
        else:
            return self.text_features[idx], self.video_features[idx]

# ─── MODEL COMPONENTS ─────────────────────────────────────────

class SimplifiedTextEncoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, TEXT_HID_DIM)
        self.bn1 = nn.BatchNorm1d(TEXT_HID_DIM)
        self.dropout = nn.Dropout(DROPOUT_RATE)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        return self.dropout(x)

class SimplifiedVideoEncoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, VIDEO_HID_DIM)
        self.bn1 = nn.BatchNorm1d(VIDEO_HID_DIM)
        self.dropout = nn.Dropout(DROPOUT_RATE)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        return self.dropout(x)

class AttentionFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.text_projection = nn.Linear(TEXT_HID_DIM, FUSION_OUT_DIM)
        self.video_projection = nn.Linear(VIDEO_HID_DIM, FUSION_OUT_DIM)
        self.attention = nn.Linear(FUSION_OUT_DIM * 2, 2)

    def forward(self, text_features, video_features):
        text_proj = torch.relu(self.text_projection(text_features))
        video_proj = torch.relu(self.video_projection(video_features))
        combined = torch.cat([text_proj, video_proj], dim=1)
        attention_weights = torch.softmax(self.attention(combined), dim=1)
        fused = attention_weights[:, 0].unsqueeze(1) * text_proj + attention_weights[:, 1].unsqueeze(1) * video_proj
        return fused

class SimplifiedModel(nn.Module):
    def __init__(self, text_dim, video_dim, num_classes):
        super().__init__()
        self.text_encoder = SimplifiedTextEncoder(text_dim)
        self.video_encoder = SimplifiedVideoEncoder(video_dim)
        self.fusion = AttentionFusion()
        self.classifier = nn.Linear(FUSION_OUT_DIM, num_classes)
        self.dropout = nn.Dropout(DROPOUT_RATE)

    def forward(self, text_input, video_input):
        text_features = self.text_encoder(text_input)
        video_features = self.video_encoder(video_input)
        fused_features = self.fusion(text_features, video_features)
        fused_features = self.dropout(fused_features)
        return self.classifier(fused_features)

# ─── TRAINING AND EVALUATION ────────────────────────────────────────────────

def train_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []
    
    for text, video, labels in tqdm(train_loader, desc="Training", leave=False):
        text, video, labels = text.to(device), video.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(text, video)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = accuracy_score(all_labels, all_preds)
    return epoch_loss, epoch_acc

def evaluate(model, data_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for text, video, labels in tqdm(data_loader, desc="Evaluating", leave=False):
            text, video, labels = text.to(device), video.to(device), labels.to(device)
            
            outputs = model(text, video)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(data_loader)
    epoch_acc = accuracy_score(all_labels, all_preds)
    return epoch_loss, epoch_acc, all_preds, all_labels

def get_class_weights(labels):
    """Calculate class weights inversely proportional to class frequency"""
    class_counts = np.bincount(labels)
    total = len(labels)
    weights = total / (len(class_counts) * class_counts)
    return torch.tensor(weights, dtype=torch.float32)

# ─── PLOTTING ────────────────────────────────────────────────────────────────
def plot_training_history(train_losses, val_losses, train_accs, val_accs, save_path="training_history.png"):
    """Plot training and validation metrics"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot losses
    ax1.plot(train_losses, label='Training Loss')
    ax1.plot(val_losses, label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    
    # Plot accuracies
    ax2.plot(train_accs, label='Training Accuracy')
    ax2.plot(val_accs, label='Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_confusion_matrix(cm, class_names, save_path="confusion_matrix.png"):
    """Plot confusion matrix"""
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    # Set labels
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names, yticklabels=class_names,
           ylabel='True label',
           xlabel='Predicted label')
    
    # Rotate xtick labels and set alignment
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Loop over data dimensions and create text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    fig.tight_layout()
    plt.savefig(save_path)
    plt.close()

# ─── ENSEMBLE MODEL ─────────────────────────────────────────────────────────
class EnsembleModel:
    def __init__(self, models, device):
        self.models = models
        self.device = device
    
    def predict(self, text_loader):
        all_predictions = []
        
        for model in self.models:
            model.eval()
            model_preds = []
            
            with torch.no_grad():
                for text, video in text_loader:
                    text, video = text.to(self.device), video.to(self.device)
                    outputs = model(text, video)
                    probs = torch.softmax(outputs, dim=1)
                    model_preds.append(probs.cpu().numpy())
            
            all_predictions.append(np.vstack(model_preds))
        
        # Average predictions from all models
        ensemble_preds = np.mean(all_predictions, axis=0)
        return np.argmax(ensemble_preds, axis=1)

# ─── MAIN FUNCTION ────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_text', default='Embeddings/Text/train_text.csv')
    parser.add_argument('--train_video', default='Embeddings/Videos/train/train_embeddings.csv')
    parser.add_argument('--train_labels', default='Embeddings/Labels/train_labels.csv')
    parser.add_argument('--test_text', default='Embeddings/Text/test_text.csv')
    parser.add_argument('--test_video', default='Embeddings/Videos/test/test_embeddings.csv')
    parser.add_argument('--test_labels', default='Embeddings/Labels/test_labels.csv')
    parser.add_argument('--output_dir', default='Models/Model_3/Outputs')
    parser.add_argument('--n_splits', default=5, type=int)
    parser.add_argument('--ensemble', action='store_true')
    parser.add_argument('--val_text',    default='Embeddings/Text/val_text.csv')
    parser.add_argument('--val_video',   default='Embeddings/Videos/val/val_embeddings.csv')
    parser.add_argument('--val_labels',  default='Embeddings/Labels/val_labels.csv')
    args = parser.parse_args()
    
    # Use the file path utils/scripts/Video/cnn_embeddings.py for custom CNN video embeddings
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Preprocess data with PCA
    train_text, val_text, test_text, train_video, val_video, test_video = preprocess_features(
        args.train_text, args.val_text, args.test_text, 
        args.train_video, args.val_video, args.test_video
    )
    
    # Load labels
    train_labels = pd.read_csv(args.train_labels)['emotion'].values
    test_labels = pd.read_csv(args.test_labels)['emotion'].values
    
    # Create datasets
    full_train_dataset = MultimodalDataset(train_text, train_video, train_labels)
    test_dataset = MultimodalDataset(test_text, test_video, test_labels)
    
    # Calculate class weights for weighted loss
    class_weights = get_class_weights(train_labels).to(device)
    print(f"Class weights: {class_weights}")
    
    # Create test dataloader
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Calculate split sizes
    val_size = int(0.15 * len(full_train_dataset))
    train_size = len(full_train_dataset) - val_size
    
    # For ensemble modeling
    models = []
    
    # Train multiple models for ensemble (or just one if not using ensemble)
    num_models = args.n_splits if args.ensemble else 1
    
    for model_idx in range(num_models):
        print(f"\n{'='*30} Model {model_idx+1}/{num_models} {'='*30}")
        
        # Split dataset with different seed for each model in the ensemble
        if args.ensemble:
            generator = torch.Generator().manual_seed(SEED + model_idx)
        else:
            generator = torch.Generator().manual_seed(SEED)
        
        # Split with stratification if possible
        if STRATIFY_VAL:
            # Create stratified split
            labels = full_train_dataset.labels.numpy()
            unique_labels = np.unique(labels)
            train_indices, val_indices = [], []
            
            for label in unique_labels:
                label_indices = np.where(labels == label)[0]
                np.random.shuffle(label_indices)
                split_idx = int(0.85 * len(label_indices))
                train_indices.extend(label_indices[:split_idx])
                val_indices.extend(label_indices[split_idx:])
            
            train_dataset = torch.utils.data.Subset(full_train_dataset, train_indices)
            val_dataset = torch.utils.data.Subset(full_train_dataset, val_indices)
        else:
            # Random split
            train_dataset, val_dataset = random_split(
                full_train_dataset, [train_size, val_size], generator=generator
            )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False
        )
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        # Create model
        num_classes = len(np.unique(train_labels))
        model = SimplifiedModel(REDUCED_TEXT_DIM, REDUCED_VIDEO_DIM, num_classes).to(device)
        
        # Loss function with class weighting
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        # Optimizer with weight decay
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
        )
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3
        )
        
        # Training history
        train_losses, val_losses = [], []
        train_accs, val_accs = [], []
        
        # Early stopping variables
        best_val_loss = float('inf')
        best_val_acc = 0.0
        early_stop_counter = 0
        best_model_path = os.path.join(args.output_dir, f'Emotion_model_{model_idx}.pt')
        
        # Training loop
        for epoch in range(MAX_EPOCHS):
            print(f"\nEpoch {epoch+1}/{MAX_EPOCHS}")
            
            # Train
            train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
            
            # Validate
            val_loss, val_acc, val_preds, val_labels = evaluate(model, val_loader, criterion, device)
            
            # Update training history
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accs.append(train_acc)
            val_accs.append(val_acc)
            
            # Print metrics
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # Learning rate scheduler step
            scheduler.step(val_loss)
            
            # Check for improvement
            if val_acc > best_val_acc:
                print("Validation accuracy improved! Saving model...")
                best_val_acc = val_acc
                best_val_loss = val_loss
                early_stop_counter = 0
                
                # Save model
                torch.save(model.state_dict(), best_model_path)
            else:
                early_stop_counter += 1
                print(f"No improvement for {early_stop_counter}/{PATIENCE} epochs")
                
                if early_stop_counter >= PATIENCE:
                    print("Early stopping triggered")
                    break
        
        # Plot training history
        plot_training_history(
            train_losses, val_losses, train_accs, val_accs,
            save_path=os.path.join(args.output_dir, f'training_history_{model_idx}.png')
        )
        
        # Load best model for evaluation
        model.load_state_dict(torch.load(best_model_path))
        
        # Add to ensemble if using ensemble method
        if args.ensemble:
            models.append(model)
    
    # Evaluate on test set
    if args.ensemble:
        # Create ensemble
        ensemble = EnsembleModel(models, device)
        
        # Create test loader for ensemble prediction
        test_loader_no_labels = DataLoader(
            MultimodalDataset(test_text, test_video),
            batch_size=BATCH_SIZE, shuffle=False
        )
        
        # Get ensemble predictions
        test_preds = ensemble.predict(test_loader_no_labels)
        print("\n--- Ensemble Model Evaluation ---")
    else:
        # Evaluate single model
        _, _, test_preds, _ = evaluate(model, test_loader, criterion, device)
        print("\n--- Model Evaluation ---")
    
    # Calculate metrics
    test_acc = accuracy_score(test_labels, test_preds)
    test_report = classification_report(test_labels, test_preds)
    test_cm = confusion_matrix(test_labels, test_preds)
    
    # Print results
    print(f"Test Accuracy: {test_acc:.4f}")
    print("\nClassification Report:")
    print(test_report)
    print("\nConfusion Matrix:")
    print(test_cm)
    
    # Plot confusion matrix
    plot_confusion_matrix(
        test_cm, 
        class_names=[f"Class {i}" for i in range(len(np.unique(test_labels)))],
        save_path=os.path.join(args.output_dir, 'confusion_matrix.png')
    )
    
    # Save results to file
    with open(os.path.join(args.output_dir, 'results.txt'), 'w') as f:
        f.write(f"Test Accuracy: {test_acc:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(test_report)
        f.write("\nConfusion Matrix:\n")
        f.write(str(test_cm))

if __name__ == '__main__':
    main()