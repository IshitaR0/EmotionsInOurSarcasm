# full_pipeline_flat_with_oversample_earlystop.py

import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
from sklearn.metrics import classification_report, confusion_matrix

# ─── HYPERPARAMETERS ──────────────────────────────────────────────────────────
TEXT_UTTER_DIM   = 100    # per-utterance FastText dim (100 or 300)
TEXT_HID_DIM     = 128
VIDEO_HID_DIM    = 128
FUSION_OUT_DIM   = 256
CLASS_HID_DIM    = 256

# ─── DATASET ──────────────────────────────────────────────────────────────────
class FlatPipelineDataset(Dataset):
    """
    text_csv  : each row is one sample, flat sequence = seq_len * TEXT_UTTER_DIM columns
    vid_csv   : each row is one sample, video_dim columns
    label_csv : one column 'emotion' with ints 0..C-1
    Rows aligned across files.
    """
    def __init__(self, text_csv, vid_csv, label_csv):
        df_txt = pd.read_csv(text_csv)
        df_vid = pd.read_csv(vid_csv)
        df_lbl = pd.read_csv(label_csv)

        assert len(df_txt) == len(df_vid) == len(df_lbl), "Row counts must match!"
        self.X_txt = torch.tensor(df_txt.values, dtype=torch.float32)
        self.X_vid = torch.tensor(df_vid.values, dtype=torch.float32)
        self.y     = torch.tensor(df_lbl['emotion'].values, dtype=torch.long)

        total_txt_dim = self.X_txt.shape[1]
        assert total_txt_dim % TEXT_UTTER_DIM == 0, \
            f"Text dim {total_txt_dim} not divisible by {TEXT_UTTER_DIM}"
        self.seq_len = total_txt_dim // TEXT_UTTER_DIM

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        t = self.X_txt[i].view(self.seq_len, TEXT_UTTER_DIM)   # (seq_len, D)
        v = self.X_vid[i].unsqueeze(0)                        # (1, V)
        return t, v, self.y[i]


# ─── MODEL COMPONENTS ─────────────────────────────────────────────────────────
class TextEncoder(nn.Module):
    def __init__(self, in_dim, hid_dim, dropout=0.3):
        super().__init__()
        self.gru     = nn.GRU(in_dim, hid_dim, batch_first=True, bidirectional=True)
        self.attn    = nn.Linear(hid_dim * 2, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out, _ = self.gru(x)                    # (B, seq_len, 2H)
        out     = self.dropout(out)
        scores  = self.attn(out).squeeze(-1)    # (B, seq_len)
        weights = torch.softmax(scores, dim=-1) # (B, seq_len)
        return (out * weights.unsqueeze(-1)).sum(1)  # (B, 2H)

class VideoEncoder(nn.Module):
    def __init__(self, in_dim, hid_dim, dropout=0.3):
        super().__init__()
        self.fc      = nn.Linear(in_dim, hid_dim)
        self.attn    = nn.Linear(hid_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h      = torch.relu(self.fc(x))        # (B, 1, H)
        h      = self.dropout(h)
        scores = self.attn(h).squeeze(-1)      # (B, 1)
        weights= torch.softmax(scores, dim=-1) # (B, 1)
        return (h * weights.unsqueeze(-1)).sum(1)  # (B, H)

class Fusion(nn.Module):
    def __init__(self, t_dim, v_dim, out_dim):
        super().__init__()
        self.tp    = nn.Linear(t_dim, out_dim)
        self.vp    = nn.Linear(v_dim, out_dim)
        self.inter = nn.Linear(out_dim * 2, 1)

    def forward(self, t_feat, v_feat):
        t_p  = torch.relu(self.tp(t_feat))          # (B, out_dim)
        v_p  = torch.relu(self.vp(v_feat))          # (B, out_dim)
        cat  = torch.cat([t_p, v_p], dim=-1)        # (B, 2*out_dim)
        gate = torch.sigmoid(self.inter(cat))       # (B,1)
        return gate * t_p + (1 - gate) * v_p        # (B, out_dim)

class Classifier(nn.Module):
    def __init__(self, in_dim, hid_dim, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hid_dim)
        self.drop= nn.Dropout(0.5)
        self.fc2 = nn.Linear(hid_dim, num_classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.drop(x)
        return self.fc2(x)

class FullModel(nn.Module):
    def __init__(self, text_utter_dim, video_dim, num_classes):
        super().__init__()
        t_enc_dim = TEXT_HID_DIM * 2
        v_enc_dim = VIDEO_HID_DIM

        self.text_enc  = TextEncoder(text_utter_dim, TEXT_HID_DIM)
        self.video_enc = VideoEncoder(video_dim, VIDEO_HID_DIM)
        self.fusion    = Fusion(t_enc_dim, v_enc_dim, FUSION_OUT_DIM)
        self.classif   = Classifier(FUSION_OUT_DIM, CLASS_HID_DIM, num_classes)

    def forward(self, t_seq, v_seq):
        t_feat = self.text_enc(t_seq)   # (B, 2H)
        v_feat = self.video_enc(v_seq)  # (B, H)
        f      = self.fusion(t_feat, v_feat)
        return self.classif(f)


# ─── TRAIN / EVAL LOOPS ───────────────────────────────────────────────────────
def train_one_epoch(model, loader, opt, crit, device):
    model.train()
    total_loss = 0.0
    for t_seq, v_seq, y in loader:
        t_seq, v_seq, y = t_seq.to(device), v_seq.to(device), y.to(device)
        opt.zero_grad()
        logits = model(t_seq, v_seq)
        loss   = crit(logits, y)
        loss.backward()
        opt.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def get_preds_labels(model, loader, device):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for t_seq, v_seq, y in loader:
            t_seq, v_seq = t_seq.to(device), v_seq.to(device)
            out = model(t_seq, v_seq)
            p   = out.argmax(dim=-1).cpu().tolist()
            preds.extend(p)
            labels.extend(y.tolist())
    return preds, labels


# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--train_text',   required=True)
    p.add_argument('--train_video',  required=True)
    p.add_argument('--train_labels', required=True)
    p.add_argument('--test_text',    required=True)
    p.add_argument('--test_video',   required=True)
    p.add_argument('--test_labels',  required=True)
    p.add_argument('--bs',    type=int,   default=32)
    p.add_argument('--lr',    type=float, default=1e-3)
    p.add_argument('--epochs',type=int,   default=100)
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load datasets
    train_ds = FlatPipelineDataset(args.train_text,
                                   args.train_video,
                                   args.train_labels)
    test_ds  = FlatPipelineDataset(args.test_text,
                                   args.test_video,
                                   args.test_labels)

    # split train → train/val
    n_val = int(0.1 * len(train_ds))
    n_tr  = len(train_ds) - n_val
    tr_ds, vl_ds = random_split(train_ds, [n_tr, n_val])

    # oversample minority in tr_ds
    train_labels_np = train_ds.y.numpy()
    tr_indices      = tr_ds.indices
    class_counts    = np.bincount(train_labels_np)
    inv_counts      = 1.0 / class_counts
    sample_wts      = inv_counts[train_labels_np[tr_indices]]
    sampler         = WeightedRandomSampler(sample_wts,
                                           num_samples=len(sample_wts),
                                           replacement=True)

    tr_ld = DataLoader(tr_ds,
                       batch_size=args.bs,
                       sampler=sampler,
                       drop_last=True)
    vl_ld = DataLoader(vl_ds, batch_size=args.bs)
    ts_ld = DataLoader(test_ds, batch_size=args.bs)

    # class weights for loss
    all_counts      = np.bincount(train_labels_np)
    weights_for_loss= all_counts.sum() / (len(all_counts) * all_counts)
    class_weights   = torch.tensor(weights_for_loss, dtype=torch.float32).to(device)

    # model, loss, optimizer, scheduler
    video_dim = train_ds.X_vid.shape[1]
    num_classes = len(np.unique(train_labels_np))
    model     = FullModel(TEXT_UTTER_DIM, video_dim, num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    # training with early stopping
    best_val_acc = 0.0
    no_improve   = 0
    patience     = 10

    for epoch in range(1, args.epochs+1):
        tr_loss = train_one_epoch(model, tr_ld, optimizer, criterion, device)
        vl_preds, vl_labels = get_preds_labels(model, vl_ld, device)
        val_acc = sum(p==l for p,l in zip(vl_preds, vl_labels)) / len(vl_labels)

        print(f"Epoch {epoch:03d}  train_loss={tr_loss:.4f}  val_acc={val_acc:.4f}")
        scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            no_improve   = 0
            torch.save(model.state_dict(), 'best_full_model.pt')
            print(" → new best model saved")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
                break

    # test evaluation
    model.load_state_dict(torch.load('best_full_model.pt', map_location=device))
    ts_preds, ts_labels = get_preds_labels(model, ts_ld, device)

    test_acc = sum(p==l for p,l in zip(ts_preds, ts_labels)) / len(ts_labels)
    print(f"\nTest Accuracy: {test_acc:.4f}\n")
    print("Classification Report:")
    print(classification_report(ts_labels, ts_preds))
    print("Confusion Matrix:")
    print(confusion_matrix(ts_labels, ts_preds))


if __name__ == '__main__':
    main()
    
# ________________________________________________________________________________________________________________
