# Train 1
import cv2
import numpy as np
import os
from skimage.metrics import structural_similarity as ssim

def calculate_ssim(frame1, frame2):
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    return ssim(gray1, gray2)

def extract_keyframes(video_path, output_folder, ssim_threshold=0.02):
    cap = cv2.VideoCapture(video_path)
    ret, prev_frame = cap.read()
    
    if not ret:
        print(f"Error reading {video_path}")
        return
    
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    video_output_path = os.path.join(output_folder, video_name)
    os.makedirs(video_output_path, exist_ok=True)

    keyframes = []
    prev_ssim = None
    frame_idx = 0

    while ret:
        ret, curr_frame = cap.read()
        if not ret:
            break

        ssim_value = calculate_ssim(prev_frame, curr_frame)

        if prev_ssim is not None:
            diff = abs(ssim_value - prev_ssim)
            if diff > ssim_threshold:  # Significant change detected
                keyframes.append((frame_idx, curr_frame))
                cv2.imwrite(f"{video_output_path}/keyframe_{frame_idx}.jpg", curr_frame)

        prev_ssim = ssim_value
        prev_frame = curr_frame
        frame_idx += 1

    cap.release()
    print(f"Extracted {len(keyframes)} keyframes from {video_name}")

def process_videos_in_folder(folder_path, output_folder, ssim_threshold=0.02):
    os.makedirs(output_folder, exist_ok=True)

    video_files = [f for f in os.listdir(folder_path) if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]
    
    for video_file in video_files:
        video_path = os.path.join(folder_path, video_file)
        extract_keyframes(video_path, output_folder, ssim_threshold)

input_folder = r"C:\Users\SUHAANI SACHDEVA\OneDrive - Plaksha University\MLPR raw data(video)\train_1"
output_folder = r"C:\Users\SUHAANI SACHDEVA\OneDrive - Plaksha University\MLPR raw data(video)\keyframe_output1"

process_videos_in_folder(input_folder, output_folder, ssim_threshold=0.02)

import os
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision.models import resnet50

model = resnet50(pretrained=True)
model = torch.nn.Sequential(*list(model.children())[:-1])  # Remove final classification layer
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def extract_features_from_folder(root_folder):
    feature_dict = {}

    # Loop through each subfolder (one per video)
    for subfolder in os.listdir(root_folder):
        subfolder_path = os.path.join(root_folder, subfolder)

        # Skip if not a folder or if it's a Jupyter checkpoint
        if not os.path.isdir(subfolder_path) or subfolder.startswith(".ipynb_checkpoints"):
            print(f"Skipping {subfolder_path}")
            continue  

        # Loop through images inside the subfolder
        for img_name in os.listdir(subfolder_path):
            img_path = os.path.join(subfolder_path, img_name)

            try:
                img = Image.open(img_path).convert("RGB")
                img_tensor = transform(img).unsqueeze(0)

                with torch.no_grad():
                    features = model(img_tensor).squeeze().numpy()

                feature_dict[f"{subfolder}/{img_name}"] = features  # Store with subfolder name

            except Exception as e:
                print(f"Skipping {img_path} due to error: {e}")

    return feature_dict

root_folder = r"C:\Users\SUHAANI SACHDEVA\OneDrive - Plaksha University\MLPR raw data(video)\keyframe_output1"  # The folder containing subfolders for each video
extracted_features = extract_features_from_folder(root_folder)

if extracted_features:
    np.save(r"C:\Users\SUHAANI SACHDEVA\OneDrive - Plaksha University\MLPR raw data(video)\keyframe_features.npy", extracted_features)
    print(f"Saved {len(extracted_features)} keyframes.")
else:
    print("No features extracted. Check your dataset.")

import numpy as np
features1 = np.load(r"C:\Users\SUHAANI SACHDEVA\OneDrive - Plaksha University\MLPR raw data(video)\keyframe_features.npy", allow_pickle=True).item()

print(f"Loaded {len(features1)} keyframes.")
print("Sample filenames:", list(features1.keys())[:5])  # Show some filenames
print(f"Feature vector shape: {features1[list(features1.keys())[0]].shape}")

print(f"Number of keyframes stored: {len(features1)}")

print("Sample filenames:", list(features1.keys())[:5])  # Show first 5 filenames
sample_filename = list(features1.keys())[0]  
print(f"Feature vector shape for {sample_filename}: {features1[sample_filename].shape}")
print(f"Feature vector sample: {features1[sample_filename][:10]}")

import pandas as pd
if features1:
    # Convert to a DataFrame
    df = pd.DataFrame(features1)
    
    # Save to CSV
    csv_path = r"C:\Users\SUHAANI SACHDEVA\OneDrive - Plaksha University\MLPR raw data(video)\keyframe_features.csv"
    df.to_csv(csv_path, index=False, header=False)

    print(f"Saved {len(features1)} keyframes to CSV.")
else:
    print("No features extracted. Check your dataset.")

import numpy as np

def average_features(features_dict):
    video_features = {}
    
    # Group by video
    video_groups = {}
    for key, value in features_dict.items():
        video_name = key.split('/')[0]  # Extract video name from path
        if video_name not in video_groups:
            video_groups[video_name] = []
        video_groups[video_name].append(value)
    
    # Calculate average for each video
    for video_name, features in video_groups.items():
        avg_feature = np.mean(features, axis=0)
        video_features[video_name] = avg_feature
    
    return video_features

averaged_features = average_features(features1)
print(f"Generated averaged features for {len(averaged_features)} videos.")
import pandas as pd

def save_to_csv(video_features, save_path):
    # Convert the dictionary to a DataFrame
    data = []
    for video_name, features in video_features.items():
        data.append([video_name] + features.tolist())
    
    # Create DataFrame with video names and feature columns
    df = pd.DataFrame(data)
    df.columns = ['Video_Name'] + [f'Feature_{i}' for i in range(len(features))]
    
    # Save to CSV
    df.to_csv(save_path, index=False)
    print(f"Averaged video features saved to {save_path}")

# Provide the save path
csv_path = r"C:\Users\SUHAANI SACHDEVA\OneDrive - Plaksha University\MLPR raw data(video)\averaged_video_features.csv"
save_to_csv(averaged_features, csv_path)
import pandas as pd

# Load CSV
csv_path = r"C:\Users\SUHAANI SACHDEVA\OneDrive - Plaksha University\MLPR raw data(video)\averaged_video_features.csv"
df = pd.read_csv(csv_path)

# Print sample data
print(df.head())
print(f"Shape of DataFrame: {df.shape}")
import pandas as pd

# Load the CSV
csv_path = r"C:\Users\SUHAANI SACHDEVA\OneDrive - Plaksha University\MLPR raw data(video)\averaged_video_features.csv"
df = pd.read_csv(csv_path)

# Extract utterance name by removing the suffix (e.g., train_10_0 → train_10)
df['Utterance'] = df['Video_Name'].apply(lambda x: '_'.join(x.split('_')[:2]))

# Group by Utterance and calculate the mean for numeric columns only
numeric_cols = df.drop(columns=['Video_Name', 'Utterance']).columns
averaged_df = df.groupby('Utterance')[numeric_cols].mean().reset_index()

# Save to CSV
output_path = r"C:\Users\SUHAANI SACHDEVA\OneDrive - Plaksha University\MLPR raw data(video)\final_averaged_features.csv"
averaged_df.to_csv(output_path, index=False)

print(f"Saved averaged vectors for {len(averaged_df)} utterances to {output_path}")
print(f"Size of the averaged dataframe: {averaged_df.shape}")


# Train 2

import cv2
import numpy as np
import os
from skimage.metrics import structural_similarity as ssim

def calculate_ssim(frame1, frame2):
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    return ssim(gray1, gray2)

def extract_keyframes(video_path, output_folder, ssim_threshold=0.02):
    cap = cv2.VideoCapture(video_path)
    ret, prev_frame = cap.read()
    
    if not ret:
        print(f"Error reading {video_path}")
        return
    
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    video_output_path = os.path.join(output_folder, video_name)
    os.makedirs(video_output_path, exist_ok=True)

    keyframes = []
    prev_ssim = None
    frame_idx = 0

    while ret:
        ret, curr_frame = cap.read()
        if not ret:
            break

        ssim_value = calculate_ssim(prev_frame, curr_frame)

        if prev_ssim is not None:
            diff = abs(ssim_value - prev_ssim)
            if diff > ssim_threshold:  # Significant change detected
                keyframes.append((frame_idx, curr_frame))
                cv2.imwrite(f"{video_output_path}/keyframe_{frame_idx}.jpg", curr_frame)

        prev_ssim = ssim_value
        prev_frame = curr_frame
        frame_idx += 1

    cap.release()
    print(f"Extracted {len(keyframes)} keyframes from {video_name}")

def process_videos_in_folder(folder_path, output_folder, ssim_threshold=0.02):
    os.makedirs(output_folder, exist_ok=True)

    video_files = [f for f in os.listdir(folder_path) if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]
    
    for video_file in video_files:
        video_path = os.path.join(folder_path, video_file)
        extract_keyframes(video_path, output_folder, ssim_threshold)

input_folder = r"C:\Users\SUHAANI SACHDEVA\OneDrive - Plaksha University\MLPR raw data(video)\train_2"
output_folder = r"C:\Users\SUHAANI SACHDEVA\OneDrive - Plaksha University\MLPR raw data(video)\keyframe_output2"

process_videos_in_folder(input_folder, output_folder, ssim_threshold=0.02)
import os
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision.models import resnet50

model = resnet50(pretrained=True)
model = torch.nn.Sequential(*list(model.children())[:-1])  # Remove final classification layer
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def extract_features_from_folder(root_folder):
    feature_dict = {}

    # Loop through each subfolder (one per video)
    for subfolder in os.listdir(root_folder):
        subfolder_path = os.path.join(root_folder, subfolder)

        # Skip if not a folder or if it's a Jupyter checkpoint
        if not os.path.isdir(subfolder_path) or subfolder.startswith(".ipynb_checkpoints"):
            print(f"Skipping {subfolder_path}")
            continue  

        # Loop through images inside the subfolder
        for img_name in os.listdir(subfolder_path):
            img_path = os.path.join(subfolder_path, img_name)

            try:
                img = Image.open(img_path).convert("RGB")
                img_tensor = transform(img).unsqueeze(0)

                with torch.no_grad():
                    features = model(img_tensor).squeeze().numpy()

                feature_dict[f"{subfolder}/{img_name}"] = features  # Store with subfolder name

            except Exception as e:
                print(f"Skipping {img_path} due to error: {e}")

    return feature_dict

root_folder = r"C:\Users\SUHAANI SACHDEVA\OneDrive - Plaksha University\MLPR raw data(video)\keyframe_output2"  # The folder containing subfolders for each video
extracted_features = extract_features_from_folder(root_folder)

if extracted_features:
    np.save(r"C:\Users\SUHAANI SACHDEVA\OneDrive - Plaksha University\MLPR raw data(video)\keyframe_features2.npy", extracted_features)
    print(f"Saved {len(extracted_features)} keyframes.")
else:
    print("No features extracted. Check your dataset.")
import numpy as np
features2 = np.load(r"C:\Users\SUHAANI SACHDEVA\OneDrive - Plaksha University\MLPR raw data(video)\keyframe_features2.npy", allow_pickle=True).item()

print(f"Loaded {len(features2)} keyframes.")
print("Sample filenames:", list(features2.keys())[:5])  # Show some filenames
print(f"Feature vector shape: {features2[list(features2.keys())[0]].shape}")
print(f"Number of keyframes stored: {len(features2)}")
import pandas as pd
if features2:
    # Convert to a DataFrame
    df = pd.DataFrame(features2)
    
    # Save to CSV
    csv_path = r"C:\Users\SUHAANI SACHDEVA\OneDrive - Plaksha University\MLPR raw data(video)\keyframe_features2.csv"
    df.to_csv(csv_path, index=False, header=False)

    print(f"Saved {len(features2)} keyframes to CSV.")
else:
    print("No features extracted. Check your dataset.")

import numpy as np

def average_features(features_dict):
    video_features = {}
    
    # Group by video
    video_groups = {}
    for key, value in features_dict.items():
        video_name = key.split('/')[0]  # Extract video name from path
        if video_name not in video_groups:
            video_groups[video_name] = []
        video_groups[video_name].append(value)
    
    # Calculate average for each video
    for video_name, features in video_groups.items():
        avg_feature = np.mean(features, axis=0)
        video_features[video_name] = avg_feature
    
    return video_features

averaged_features = average_features(features2)
print(f"Generated averaged features for {len(averaged_features)} videos.")
import pandas as pd

def save_to_csv(video_features, save_path):
    # Convert the dictionary to a DataFrame
    data = []
    for video_name, features in video_features.items():
        data.append([video_name] + features.tolist())
    
    # Create DataFrame with video names and feature columns
    df = pd.DataFrame(data)
    df.columns = ['Video_Name'] + [f'Feature_{i}' for i in range(len(features))]
    
    # Save to CSV
    df.to_csv(save_path, index=False)
    print(f"Averaged video features saved to {save_path}")

# Provide the save path
csv_path = r"C:\Users\SUHAANI SACHDEVA\OneDrive - Plaksha University\MLPR raw data(video)\averaged_video_features2.csv"
save_to_csv(averaged_features, csv_path)
import pandas as pd

# Load CSV
csv_path = r"C:\Users\SUHAANI SACHDEVA\OneDrive - Plaksha University\MLPR raw data(video)\averaged_video_features2.csv"
df = pd.read_csv(csv_path)

# Print sample data
print(df.head())
print(f"Shape of DataFrame: {df.shape}")
import pandas as pd

# Load the CSV
csv_path = r"C:\Users\SUHAANI SACHDEVA\OneDrive - Plaksha University\MLPR raw data(video)\averaged_video_features2.csv"
df = pd.read_csv(csv_path)

# Extract utterance name by removing the suffix (e.g., train_10_0 → train_10)
df['Utterance'] = df['Video_Name'].apply(lambda x: '_'.join(x.split('_')[:2]))

# Group by Utterance and calculate the mean for numeric columns only
numeric_cols = df.drop(columns=['Video_Name', 'Utterance']).columns
averaged_df = df.groupby('Utterance')[numeric_cols].mean().reset_index()

# Save to CSV
output_path = r"C:\Users\SUHAANI SACHDEVA\OneDrive - Plaksha University\MLPR raw data(video)\final_averaged_features2.csv"
averaged_df.to_csv(output_path, index=False)

print(f"Saved averaged vectors for {len(averaged_df)} utterances to {output_path}")
print(f"Size of the averaged dataframe: {averaged_df.shape}")


# Train 3
import cv2
import numpy as np
import os
from skimage.metrics import structural_similarity as ssim

def calculate_ssim(frame1, frame2):
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    return ssim(gray1, gray2)

def extract_keyframes(video_path, output_folder, ssim_threshold=0.02):
    cap = cv2.VideoCapture(video_path)
    ret, prev_frame = cap.read()
    
    if not ret:
        print(f"Error reading {video_path}")
        return
    
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    video_output_path = os.path.join(output_folder, video_name)
    os.makedirs(video_output_path, exist_ok=True)

    keyframes = []
    prev_ssim = None
    frame_idx = 0

    while ret:
        ret, curr_frame = cap.read()
        if not ret:
            break

        ssim_value = calculate_ssim(prev_frame, curr_frame)

        if prev_ssim is not None:
            diff = abs(ssim_value - prev_ssim)
            if diff > ssim_threshold:  # Significant change detected
                keyframes.append((frame_idx, curr_frame))
                cv2.imwrite(f"{video_output_path}/keyframe_{frame_idx}.jpg", curr_frame)

        prev_ssim = ssim_value
        prev_frame = curr_frame
        frame_idx += 1

    cap.release()
    print(f"Extracted {len(keyframes)} keyframes from {video_name}")

def process_videos_in_folder(folder_path, output_folder, ssim_threshold=0.02):
    os.makedirs(output_folder, exist_ok=True)

    video_files = [f for f in os.listdir(folder_path) if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]
    
    for video_file in video_files:
        video_path = os.path.join(folder_path, video_file)
        extract_keyframes(video_path, output_folder, ssim_threshold)

input_folder = r"C:\Users\SUHAANI SACHDEVA\OneDrive - Plaksha University\MLPR raw data(video)\train_3"
output_folder = r"C:\Users\SUHAANI SACHDEVA\OneDrive - Plaksha University\MLPR raw data(video)\keyframe_output3"

process_videos_in_folder(input_folder, output_folder, ssim_threshold=0.02)
import os
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision.models import resnet50

model = resnet50(pretrained=True)
model = torch.nn.Sequential(*list(model.children())[:-1])  # Remove final classification layer
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def extract_features_from_folder(root_folder):
    feature_dict = {}

    # Loop through each subfolder (one per video)
    for subfolder in os.listdir(root_folder):
        subfolder_path = os.path.join(root_folder, subfolder)

        # Skip if not a folder or if it's a Jupyter checkpoint
        if not os.path.isdir(subfolder_path) or subfolder.startswith(".ipynb_checkpoints"):
            print(f"Skipping {subfolder_path}")
            continue  

        # Loop through images inside the subfolder
        for img_name in os.listdir(subfolder_path):
            img_path = os.path.join(subfolder_path, img_name)

            try:
                img = Image.open(img_path).convert("RGB")
                img_tensor = transform(img).unsqueeze(0)

                with torch.no_grad():
                    features = model(img_tensor).squeeze().numpy()

                feature_dict[f"{subfolder}/{img_name}"] = features  # Store with subfolder name

            except Exception as e:
                print(f"Skipping {img_path} due to error: {e}")

    return feature_dict

root_folder = r"C:\Users\SUHAANI SACHDEVA\OneDrive - Plaksha University\MLPR raw data(video)\keyframe_output3"  # The folder containing subfolders for each video
extracted_features = extract_features_from_folder(root_folder)

if extracted_features:
    np.save(r"C:\Users\SUHAANI SACHDEVA\OneDrive - Plaksha University\MLPR raw data(video)\keyframe_features3.npy", extracted_features)
    print(f"Saved {len(extracted_features)} keyframes.")
else:
    print("No features extracted. Check your dataset.")
import numpy as np
features3 = np.load(r"C:\Users\SUHAANI SACHDEVA\OneDrive - Plaksha University\MLPR raw data(video)\keyframe_features3.npy", allow_pickle=True).item()

print(f"Loaded {len(features3)} keyframes.")
print("Sample filenames:", list(features3.keys())[:5])  # Show some filenames
print(f"Feature vector shape: {features3[list(features3.keys())[0]].shape}")
import pandas as pd
if features3:
    # Convert to a DataFrame
    df = pd.DataFrame(features3)
    
    # Save to CSV
    csv_path = r"C:\Users\SUHAANI SACHDEVA\OneDrive - Plaksha University\MLPR raw data(video)\keyframe_features3.csv"
    df.to_csv(csv_path, index=False, header=False)

    print(f"Saved {len(features3)} keyframes to CSV.")
else:
    print("No features extracted. Check your dataset.")

import numpy as np

def average_features(features_dict):
    video_features = {}
    
    # Group by video
    video_groups = {}
    for key, value in features_dict.items():
        video_name = key.split('/')[0]  # Extract video name from path
        if video_name not in video_groups:
            video_groups[video_name] = []
        video_groups[video_name].append(value)
    
    # Calculate average for each video
    for video_name, features in video_groups.items():
        avg_feature = np.mean(features, axis=0)
        video_features[video_name] = avg_feature
    
    return video_features

averaged_features = average_features(features3)
print(f"Generated averaged features for {len(averaged_features)} videos.")
import pandas as pd

def save_to_csv(video_features, save_path):
    # Convert the dictionary to a DataFrame
    data = []
    for video_name, features in video_features.items():
        data.append([video_name] + features.tolist())
    
    # Create DataFrame with video names and feature columns
    df = pd.DataFrame(data)
    df.columns = ['Video_Name'] + [f'Feature_{i}' for i in range(len(features))]
    
    # Save to CSV
    df.to_csv(save_path, index=False)
    print(f"Averaged video features saved to {save_path}")

# Provide the save path
csv_path = r"C:\Users\SUHAANI SACHDEVA\OneDrive - Plaksha University\MLPR raw data(video)\averaged_video_features3.csv"
save_to_csv(averaged_features, csv_path)
import pandas as pd

# Load CSV
csv_path = r"C:\Users\SUHAANI SACHDEVA\OneDrive - Plaksha University\MLPR raw data(video)\averaged_video_features3.csv"
df = pd.read_csv(csv_path)

# Print sample data
print(df.head())
print(f"Shape of DataFrame: {df.shape}")
import pandas as pd

# Load the CSV
csv_path = r"C:\Users\SUHAANI SACHDEVA\OneDrive - Plaksha University\MLPR raw data(video)\averaged_video_features3.csv"
df = pd.read_csv(csv_path)

# Extract utterance name by removing the suffix (e.g., train_10_0 → train_10)
df['Utterance'] = df['Video_Name'].apply(lambda x: '_'.join(x.split('_')[:2]))

# Group by Utterance and calculate the mean for numeric columns only
numeric_cols = df.drop(columns=['Video_Name', 'Utterance']).columns
averaged_df = df.groupby('Utterance')[numeric_cols].mean().reset_index()

# Save to CSV
output_path = r"C:\Users\SUHAANI SACHDEVA\OneDrive - Plaksha University\MLPR raw data(video)\final_averaged_features3.csv"
averaged_df.to_csv(output_path, index=False)

print(f"Saved averaged vectors for {len(averaged_df)} utterances to {output_path}")
print(f"Size of the averaged dataframe: {averaged_df.shape}")

#Train 4
import cv2
import numpy as np
import os
from skimage.metrics import structural_similarity as ssim

def calculate_ssim(frame1, frame2):
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    return ssim(gray1, gray2)

def extract_keyframes(video_path, output_folder, ssim_threshold=0.02):
    cap = cv2.VideoCapture(video_path)
    ret, prev_frame = cap.read()
    
    if not ret:
        print(f"Error reading {video_path}")
        return
    
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    video_output_path = os.path.join(output_folder, video_name)
    os.makedirs(video_output_path, exist_ok=True)

    keyframes = []
    prev_ssim = None
    frame_idx = 0

    while ret:
        ret, curr_frame = cap.read()
        if not ret:
            break

        ssim_value = calculate_ssim(prev_frame, curr_frame)

        if prev_ssim is not None:
            diff = abs(ssim_value - prev_ssim)
            if diff > ssim_threshold:  # Significant change detected
                keyframes.append((frame_idx, curr_frame))
                cv2.imwrite(f"{video_output_path}/keyframe_{frame_idx}.jpg", curr_frame)

        prev_ssim = ssim_value
        prev_frame = curr_frame
        frame_idx += 1

    cap.release()
    print(f"Extracted {len(keyframes)} keyframes from {video_name}")

def process_videos_in_folder(folder_path, output_folder, ssim_threshold=0.02):
    os.makedirs(output_folder, exist_ok=True)

    video_files = [f for f in os.listdir(folder_path) if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]
    
    for video_file in video_files:
        video_path = os.path.join(folder_path, video_file)
        extract_keyframes(video_path, output_folder, ssim_threshold)

input_folder = r"C:\Users\SUHAANI SACHDEVA\OneDrive - Plaksha University\MLPR raw data(video)\train_4"
output_folder = r"C:\Users\SUHAANI SACHDEVA\OneDrive - Plaksha University\MLPR raw data(video)\keyframe_output4"

process_videos_in_folder(input_folder, output_folder, ssim_threshold=0.02)
import os
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision.models import resnet50

model = resnet50(pretrained=True)
model = torch.nn.Sequential(*list(model.children())[:-1])  # Remove final classification layer
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def extract_features_from_folder(root_folder):
    feature_dict = {}

    # Loop through each subfolder (one per video)
    for subfolder in os.listdir(root_folder):
        subfolder_path = os.path.join(root_folder, subfolder)

        # Skip if not a folder or if it's a Jupyter checkpoint
        if not os.path.isdir(subfolder_path) or subfolder.startswith(".ipynb_checkpoints"):
            print(f"Skipping {subfolder_path}")
            continue  

        # Loop through images inside the subfolder
        for img_name in os.listdir(subfolder_path):
            img_path = os.path.join(subfolder_path, img_name)

            try:
                img = Image.open(img_path).convert("RGB")
                img_tensor = transform(img).unsqueeze(0)

                with torch.no_grad():
                    features = model(img_tensor).squeeze().numpy()

                feature_dict[f"{subfolder}/{img_name}"] = features  # Store with subfolder name

            except Exception as e:
                print(f"Skipping {img_path} due to error: {e}")

    return feature_dict

root_folder = r"C:\Users\SUHAANI SACHDEVA\OneDrive - Plaksha University\MLPR raw data(video)\keyframe_output4"  # The folder containing subfolders for each video
extracted_features = extract_features_from_folder(root_folder)

if extracted_features:
    np.save(r"C:\Users\SUHAANI SACHDEVA\OneDrive - Plaksha University\MLPR raw data(video)\keyframe_features4.npy", extracted_features)
    print(f"Saved {len(extracted_features)} keyframes.")
else:
    print("No features extracted. Check your dataset.")
import numpy as np
features4 = np.load(r"C:\Users\SUHAANI SACHDEVA\OneDrive - Plaksha University\MLPR raw data(video)\keyframe_features4.npy", allow_pickle=True).item()

print(f"Loaded {len(features4)} keyframes.")
print("Sample filenames:", list(features4.keys())[:5])  # Show some filenames
print(f"Feature vector shape: {features4[list(features4.keys())[0]].shape}")
import pandas as pd
if features4:
    # Convert to a DataFrame
    df = pd.DataFrame(features4)
    
    # Save to CSV
    csv_path = r"C:\Users\SUHAANI SACHDEVA\OneDrive - Plaksha University\MLPR raw data(video)\keyframe_features4.csv"
    df.to_csv(csv_path, index=False, header=False)

    print(f"Saved {len(features4)} keyframes to CSV.")
else:
    print("No features extracted. Check your dataset.")

import numpy as np

def average_features(features_dict):
    video_features = {}
    
    # Group by video
    video_groups = {}
    for key, value in features_dict.items():
        video_name = key.split('/')[0]  # Extract video name from path
        if video_name not in video_groups:
            video_groups[video_name] = []
        video_groups[video_name].append(value)
    
    # Calculate average for each video
    for video_name, features in video_groups.items():
        avg_feature = np.mean(features, axis=0)
        video_features[video_name] = avg_feature
    
    return video_features

averaged_features = average_features(features4)
print(f"Generated averaged features for {len(averaged_features)} videos.")
import pandas as pd

def save_to_csv(video_features, save_path):
    # Convert the dictionary to a DataFrame
    data = []
    for video_name, features in video_features.items():
        data.append([video_name] + features.tolist())
    
    # Create DataFrame with video names and feature columns
    df = pd.DataFrame(data)
    df.columns = ['Video_Name'] + [f'Feature_{i}' for i in range(len(features))]
    
    # Save to CSV
    df.to_csv(save_path, index=False)
    print(f"Averaged video features saved to {save_path}")

# Provide the save path
csv_path = r"C:\Users\SUHAANI SACHDEVA\OneDrive - Plaksha University\MLPR raw data(video)\averaged_video_features4.csv"
save_to_csv(averaged_features, csv_path)
import pandas as pd

# Load CSV
csv_path = r"C:\Users\SUHAANI SACHDEVA\OneDrive - Plaksha University\MLPR raw data(video)\averaged_video_features4.csv"
df = pd.read_csv(csv_path)

# Print sample data
print(df.head())
print(f"Shape of DataFrame: {df.shape}")
import pandas as pd

# Load the CSV
csv_path = r"C:\Users\SUHAANI SACHDEVA\OneDrive - Plaksha University\MLPR raw data(video)\averaged_video_features4.csv"
df = pd.read_csv(csv_path)

# Extract utterance name by removing the suffix (e.g., train_10_0 → train_10)
df['Utterance'] = df['Video_Name'].apply(lambda x: '_'.join(x.split('_')[:2]))

# Group by Utterance and calculate the mean for numeric columns only
numeric_cols = df.drop(columns=['Video_Name', 'Utterance']).columns
averaged_df = df.groupby('Utterance')[numeric_cols].mean().reset_index()

# Save to CSV
output_path = r"C:\Users\SUHAANI SACHDEVA\OneDrive - Plaksha University\MLPR raw data(video)\final_averaged_features4.csv"
averaged_df.to_csv(output_path, index=False)

print(f"Saved averaged vectors for {len(averaged_df)} utterances to {output_path}")
print(f"Size of the averaged dataframe: {averaged_df.shape}")


# Test
import cv2
import numpy as np
import os
from skimage.metrics import structural_similarity as ssim

def calculate_ssim(frame1, frame2):
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    return ssim(gray1, gray2)

def extract_keyframes(video_path, output_folder, ssim_threshold=0.02):
    cap = cv2.VideoCapture(video_path)
    ret, prev_frame = cap.read()
    
    if not ret:
        print(f"Error reading {video_path}")
        return
    
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    video_output_path = os.path.join(output_folder, video_name)
    os.makedirs(video_output_path, exist_ok=True)

    keyframes = []
    prev_ssim = None
    frame_idx = 0

    while ret:
        ret, curr_frame = cap.read()
        if not ret:
            break

        ssim_value = calculate_ssim(prev_frame, curr_frame)

        if prev_ssim is not None:
            diff = abs(ssim_value - prev_ssim)
            if diff > ssim_threshold:  # Significant change detected
                keyframes.append((frame_idx, curr_frame))
                cv2.imwrite(f"{video_output_path}/keyframe_{frame_idx}.jpg", curr_frame)

        prev_ssim = ssim_value
        prev_frame = curr_frame
        frame_idx += 1

    cap.release()
    print(f"Extracted {len(keyframes)} keyframes from {video_name}")

def process_videos_in_folder(folder_path, output_folder, ssim_threshold=0.02):
    os.makedirs(output_folder, exist_ok=True)

    video_files = [f for f in os.listdir(folder_path) if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]
    
    for video_file in video_files:
        video_path = os.path.join(folder_path, video_file)
        extract_keyframes(video_path, output_folder, ssim_threshold)

input_folder = r"C:\Users\SUHAANI SACHDEVA\OneDrive - Plaksha University\MLPR raw data(video)\test"
output_folder = r"C:\Users\SUHAANI SACHDEVA\OneDrive - Plaksha University\MLPR raw data(video)\keyframe_outputtest"

process_videos_in_folder(input_folder, output_folder, ssim_threshold=0.02)
import os
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision.models import resnet50

input_folder = r"C:\Users\SUHAANI SACHDEVA\OneDrive - Plaksha University\MLPR raw data(video)\test"
output_folder = r"C:\Users\SUHAANI SACHDEVA\OneDrive - Plaksha University\MLPR raw data(video)\keyframe_outputtest"

model = resnet50(pretrained=True)
model = torch.nn.Sequential(*list(model.children())[:-1])  # Remove final classification layer
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def extract_features_from_folder(root_folder):
    feature_dict = {}

    # Loop through each subfolder (one per video)
    for subfolder in os.listdir(root_folder):
        subfolder_path = os.path.join(root_folder, subfolder)

        # Skip if not a folder or if it's a Jupyter checkpoint
        if not os.path.isdir(subfolder_path) or subfolder.startswith(".ipynb_checkpoints"):
            print(f"Skipping {subfolder_path}")
            continue  

        # Loop through images inside the subfolder
        for img_name in os.listdir(subfolder_path):
            img_path = os.path.join(subfolder_path, img_name)

            try:
                img = Image.open(img_path).convert("RGB")
                img_tensor = transform(img).unsqueeze(0)

                with torch.no_grad():
                    features = model(img_tensor).squeeze().numpy()

                feature_dict[f"{subfolder}/{img_name}"] = features  # Store with subfolder name

            except Exception as e:
                print(f"Skipping {img_path} due to error: {e}")

    return feature_dict

root_folder = r"C:\Users\SUHAANI SACHDEVA\OneDrive - Plaksha University\MLPR raw data(video)\keyframe_outputtest"  # The folder containing subfolders for each video
extracted_features = extract_features_from_folder(root_folder)

if extracted_features:
    np.save(r"C:\Users\SUHAANI SACHDEVA\OneDrive - Plaksha University\MLPR raw data(video)\keyframe_featurestest.npy", extracted_features)
    print(f"Saved {len(extracted_features)} keyframes.")
else:
    print("No features extracted. Check your dataset.")
import numpy as np
featurest = np.load(r"C:\Users\SUHAANI SACHDEVA\OneDrive - Plaksha University\MLPR raw data(video)\keyframe_featurestest.npy", allow_pickle=True).item()

print(f"Loaded {len(featurest)} keyframes.")
print("Sample filenames:", list(featurest.keys())[:5])  # Show some filenames
print(f"Feature vector shape: {featurest[list(featurest.keys())[0]].shape}")
import pandas as pd
if featurest:
    # Convert to a DataFrame
    df = pd.DataFrame(featurest)
    
    # Save to CSV
    csv_path = r"C:\Users\SUHAANI SACHDEVA\OneDrive - Plaksha University\MLPR raw data(video)\keyframe_featurestest.csv"
    df.to_csv(csv_path, index=False, header=False)

    print(f"Saved {len(featurest)} keyframes to CSV.")
else:
    print("No features extracted. Check your dataset.")

import numpy as np

def average_features(features_dict):
    video_features = {}
    
    # Group by video
    video_groups = {}
    for key, value in features_dict.items():
        video_name = key.split('/')[0]  # Extract video name from path
        if video_name not in video_groups:
            video_groups[video_name] = []
        video_groups[video_name].append(value)
    
    # Calculate average for each video
    for video_name, features in video_groups.items():
        avg_feature = np.mean(features, axis=0)
        video_features[video_name] = avg_feature
    
    return video_features

averaged_features = average_features(featurest)
print(f"Generated averaged features for {len(averaged_features)} videos.")
import pandas as pd

def save_to_csv(video_features, save_path):
    # Convert the dictionary to a DataFrame
    data = []
    for video_name, features in video_features.items():
        data.append([video_name] + features.tolist())
    
    # Create DataFrame with video names and feature columns
    df = pd.DataFrame(data)
    df.columns = ['Video_Name'] + [f'Feature_{i}' for i in range(len(features))]
    
    # Save to CSV
    df.to_csv(save_path, index=False)
    print(f"Averaged video features saved to {save_path}")

# Provide the save path
csv_path = r"C:\Users\SUHAANI SACHDEVA\OneDrive - Plaksha University\MLPR raw data(video)\averaged_video_featurestest.csv"
save_to_csv(averaged_features, csv_path)
import pandas as pd

# Load CSV
csv_path = r"C:\Users\SUHAANI SACHDEVA\OneDrive - Plaksha University\MLPR raw data(video)\averaged_video_featurestest.csv"
df = pd.read_csv(csv_path)

# Print sample data
print(df.head())
print(f"Shape of DataFrame: {df.shape}")
import pandas as pd

# Load the CSV
csv_path = r"C:\Users\SUHAANI SACHDEVA\OneDrive - Plaksha University\MLPR raw data(video)\averaged_video_featurestest.csv"
df = pd.read_csv(csv_path)

# Extract utterance name by removing the suffix (e.g., train_10_0 → train_10)
df['Utterance'] = df['Video_Name'].apply(lambda x: '_'.join(x.split('_')[:2]))

# Group by Utterance and calculate the mean for numeric columns only
numeric_cols = df.drop(columns=['Video_Name', 'Utterance']).columns
averaged_df = df.groupby('Utterance')[numeric_cols].mean().reset_index()

# Save to CSV
output_path = r"C:\Users\SUHAANI SACHDEVA\OneDrive - Plaksha University\MLPR raw data(video)\final_averaged_featurestest.csv"
averaged_df.to_csv(output_path, index=False)

print(f"Saved averaged vectors for {len(averaged_df)} utterances to {output_path}")
print(f"Size of the averaged dataframe: {averaged_df.shape}")

# val
import cv2
import numpy as np
import os
from skimage.metrics import structural_similarity as ssim

def calculate_ssim(frame1, frame2):
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    return ssim(gray1, gray2)

def extract_keyframes(video_path, output_folder, ssim_threshold=0.02):
    cap = cv2.VideoCapture(video_path)
    ret, prev_frame = cap.read()
    
    if not ret:
        print(f"Error reading {video_path}")
        return
    
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    video_output_path = os.path.join(output_folder, video_name)
    os.makedirs(video_output_path, exist_ok=True)

    keyframes = []
    prev_ssim = None
    frame_idx = 0

    while ret:
        ret, curr_frame = cap.read()
        if not ret:
            break

        ssim_value = calculate_ssim(prev_frame, curr_frame)

        if prev_ssim is not None:
            diff = abs(ssim_value - prev_ssim)
            if diff > ssim_threshold:  # Significant change detected
                keyframes.append((frame_idx, curr_frame))
                cv2.imwrite(f"{video_output_path}/keyframe_{frame_idx}.jpg", curr_frame)

        prev_ssim = ssim_value
        prev_frame = curr_frame
        frame_idx += 1

    cap.release()
    print(f"Extracted {len(keyframes)} keyframes from {video_name}")

def process_videos_in_folder(folder_path, output_folder, ssim_threshold=0.02):
    os.makedirs(output_folder, exist_ok=True)

    video_files = [f for f in os.listdir(folder_path) if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]
    
    for video_file in video_files:
        video_path = os.path.join(folder_path, video_file)
        extract_keyframes(video_path, output_folder, ssim_threshold)

input_folder = r"C:\Users\SUHAANI SACHDEVA\OneDrive - Plaksha University\MLPR raw data(video)\val"
output_folder = r"C:\Users\SUHAANI SACHDEVA\OneDrive - Plaksha University\MLPR raw data(video)\keyframe_outputval"

process_videos_in_folder(input_folder, output_folder, ssim_threshold=0.02)

import os
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision.models import resnet50

input_folder = r"C:\Users\SUHAANI SACHDEVA\OneDrive - Plaksha University\MLPR raw data(video)\val"
output_folder = r"C:\Users\SUHAANI SACHDEVA\OneDrive - Plaksha University\MLPR raw data(video)\keyframe_outputval"

model = resnet50(pretrained=True)
model = torch.nn.Sequential(*list(model.children())[:-1])  # Remove final classification layer
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def extract_features_from_folder(root_folder):
    feature_dict = {}

    # Loop through each subfolder (one per video)
    for subfolder in os.listdir(root_folder):
        subfolder_path = os.path.join(root_folder, subfolder)

        # Skip if not a folder or if it's a Jupyter checkpoint
        if not os.path.isdir(subfolder_path) or subfolder.startswith(".ipynb_checkpoints"):
            print(f"Skipping {subfolder_path}")
            continue  

        # Loop through images inside the subfolder
        for img_name in os.listdir(subfolder_path):
            img_path = os.path.join(subfolder_path, img_name)

            try:
                img = Image.open(img_path).convert("RGB")
                img_tensor = transform(img).unsqueeze(0)

                with torch.no_grad():
                    features = model(img_tensor).squeeze().numpy()

                feature_dict[f"{subfolder}/{img_name}"] = features  # Store with subfolder name

            except Exception as e:
                print(f"Skipping {img_path} due to error: {e}")

    return feature_dict

root_folder = r"C:\Users\SUHAANI SACHDEVA\OneDrive - Plaksha University\MLPR raw data(video)\keyframe_outputval"  # The folder containing subfolders for each video
extracted_features = extract_features_from_folder(root_folder)

if extracted_features:
    np.save(r"C:\Users\SUHAANI SACHDEVA\OneDrive - Plaksha University\MLPR raw data(video)\keyframe_featuresval.npy", extracted_features)
    print(f"Saved {len(extracted_features)} keyframes.")
else:
    print("No features extracted. Check your dataset.")
import numpy as np
featuresv = np.load(r"C:\Users\SUHAANI SACHDEVA\OneDrive - Plaksha University\MLPR raw data(video)\keyframe_featuresval.npy", allow_pickle=True).item()

print(f"Loaded {len(featuresv)} keyframes.")
print("Sample filenames:", list(featuresv.keys())[:5])  # Show some filenames
print(f"Feature vector shape: {featuresv[list(featuresv.keys())[0]].shape}")
import pandas as pd
if featuresv:
    # Convert to a DataFrame
    df = pd.DataFrame(featuresv)
    
    # Save to CSV
    csv_path = r"C:\Users\SUHAANI SACHDEVA\OneDrive - Plaksha University\MLPR raw data(video)\keyframe_featuresval.csv"
    df.to_csv(csv_path, index=False, header=False)

    print(f"Saved {len(featuresv)} keyframes to CSV.")
else:
    print("No features extracted. Check your dataset.")

import numpy as np

def average_features(features_dict):
    video_features = {}
    
    # Group by video
    video_groups = {}
    for key, value in features_dict.items():
        video_name = key.split('/')[0]  # Extract video name from path
        if video_name not in video_groups:
            video_groups[video_name] = []
        video_groups[video_name].append(value)
    
    # Calculate average for each video
    for video_name, features in video_groups.items():
        avg_feature = np.mean(features, axis=0)
        video_features[video_name] = avg_feature
    
    return video_features

averaged_features = average_features(featuresv)
print(f"Generated averaged features for {len(averaged_features)} videos.")
import pandas as pd

def save_to_csv(video_features, save_path):
    # Convert the dictionary to a DataFrame
    data = []
    for video_name, features in video_features.items():
        data.append([video_name] + features.tolist())
    
    # Create DataFrame with video names and feature columns
    df = pd.DataFrame(data)
    df.columns = ['Video_Name'] + [f'Feature_{i}' for i in range(len(features))]
    
    # Save to CSV
    df.to_csv(save_path, index=False)
    print(f"Averaged video features saved to {save_path}")

# Provide the save path
csv_path = r"C:\Users\SUHAANI SACHDEVA\OneDrive - Plaksha University\MLPR raw data(video)\averaged_video_featuresval.csv"
save_to_csv(averaged_features, csv_path)
import pandas as pd

# Load CSV
csv_path = r"C:\Users\SUHAANI SACHDEVA\OneDrive - Plaksha University\MLPR raw data(video)\averaged_video_featuresval.csv"
df = pd.read_csv(csv_path)

# Print sample data
print(df.head())
print(f"Shape of DataFrame: {df.shape}")
import pandas as pd

# Load the CSV
csv_path = r"C:\Users\SUHAANI SACHDEVA\OneDrive - Plaksha University\MLPR raw data(video)\averaged_video_featuresval.csv"
df = pd.read_csv(csv_path)

# Extract utterance name by removing the suffix (e.g., train_10_0 → train_10)
df['Utterance'] = df['Video_Name'].apply(lambda x: '_'.join(x.split('_')[:2]))

# Group by Utterance and calculate the mean for numeric columns only
numeric_cols = df.drop(columns=['Video_Name', 'Utterance']).columns
averaged_df = df.groupby('Utterance')[numeric_cols].mean().reset_index()

# Save to CSV
output_path = r"C:\Users\SUHAANI SACHDEVA\OneDrive - Plaksha University\MLPR raw data(video)\final_averaged_featuresval.csv"
averaged_df.to_csv(output_path, index=False)

print(f"Saved averaged vectors for {len(averaged_df)} utterances to {output_path}")
print(f"Size of the averaged dataframe: {averaged_df.shape}")
