import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np

# Define the custom CNN model
class CustomCNN2048(nn.Module):
    def __init__(self):
        super(CustomCNN2048, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(256, 2048)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Instantiate the model
model = CustomCNN2048()
model.eval()

# Define transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Feature extraction function
def extract_features_custom_2048(root_folder):
    feature_dict = {}
    for subfolder in os.listdir(root_folder):
        subfolder_path = os.path.join(root_folder, subfolder)
        if not os.path.isdir(subfolder_path) or subfolder.startswith(".ipynb_checkpoints"):
            continue

        for img_name in os.listdir(subfolder_path):
            img_path = os.path.join(subfolder_path, img_name)

            try:
                img = Image.open(img_path).convert("RGB")
                img_tensor = transform(img).unsqueeze(0)

                with torch.no_grad():
                    features = model(img_tensor).squeeze().numpy()

                feature_dict[f"{subfolder}/{img_name}"] = features

            except Exception as e:
                print(f"Skipping {img_path} due to error: {e}")
    return feature_dict

# Base input/output path
base_input_path = r"C:\Users\SUHAANI SACHDEVA\OneDrive - Plaksha University\MLPR raw data(video)"
sets = ["1", "2", "3", "4", "val", "test"]

# Loop through all sets and extract features
for dataset in sets:
    print(f"\nProcessing {dataset}...")
    keyframe_folder = os.path.join(base_input_path, f"keyframe_output{dataset}")
    output_npy = os.path.join(base_input_path, f"keyframe_features_customcnn2048_{dataset}.npy")

    features = extract_features_custom_2048(keyframe_folder)

    if features:
        np.save(output_npy, features)
        print(f"Saved {len(features)} features to {output_npy}")
    else:
        print(f"No features extracted for {dataset}.")
import os
import numpy as np
import pandas as pd

# -----------------------------
# Step 1: Average Features Per Video
# -----------------------------
def average_features_per_video(features_dict):
    video_features = {}
    video_groups = {}

    for key, value in features_dict.items():
        video_name = key.split('/')[0]  # e.g., train_1_0
        if video_name not in video_groups:
            video_groups[video_name] = []
        video_groups[video_name].append(value)

    for video_name, features in video_groups.items():
        avg_feature = np.mean(features, axis=0)
        video_features[video_name] = avg_feature

    return video_features

# -----------------------------
# Step 2: Average Per Utterance (e.g., train_1_0, train_1_1 → train_1)
# -----------------------------
def average_features_per_utterance(video_features):
    utterance_groups = {}
    for video_name, feature_vector in video_features.items():
        utterance = '_'.join(video_name.split('_')[:2])  # Extract "train_1", "val_2", etc.
        if utterance not in utterance_groups:
            utterance_groups[utterance] = []
        utterance_groups[utterance].append(feature_vector)

    averaged_utterances = {}
    for utterance, vectors in utterance_groups.items():
        avg_vector = np.mean(vectors, axis=0)
        averaged_utterances[utterance] = avg_vector

    return averaged_utterances

# -----------------------------
# Step 3: Save to CSV
# -----------------------------
def save_utterance_features_to_csv(utterance_features, save_path):
    data = []
    for utterance, features in utterance_features.items():
        data.append([utterance] + features.tolist())

    df = pd.DataFrame(data)
    df.columns = ['Utterance'] + [f'Feature_{i}' for i in range(len(features))]
    df.to_csv(save_path, index=False)
    print(f"Saved: {save_path}")

# -----------------------------
# Step 4: Load all npy files → average → save to CSV
# -----------------------------
base_path = r"C:\Users\SUHAANI SACHDEVA\OneDrive - Plaksha University\MLPR raw data(video)"
sets = ["1", "2", "3", "4", "val", "test"]

for dataset in sets:
    dataset_label = f"train_{dataset}" if dataset in ["1", "2", "3", "4"] else dataset
    input_npy = os.path.join(base_path, f"keyframe_features_customcnn2048_{dataset}.npy")
    output_csv = os.path.join(base_path, f"final_utterance_features_customcnn2048_{dataset_label}.csv")

    if not os.path.exists(input_npy):
        print(f"Missing file: {input_npy}")
        continue

    print(f"\nProcessing: {dataset_label}")
    features_dict = np.load(input_npy, allow_pickle=True).item()

    video_features = average_features_per_video(features_dict)
    utterance_features = average_features_per_utterance(video_features)
    save_utterance_features_to_csv(utterance_features, output_csv)