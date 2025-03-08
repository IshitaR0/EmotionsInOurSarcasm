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

# Set folder paths
input_folder = r"C:\Users\SUHAANI SACHDEVA\OneDrive - Plaksha University\MLPR raw data(video)\test_1"
output_folder = r"C:\Users\SUHAANI SACHDEVA\OneDrive - Plaksha University\MLPR raw data(video)\keyframe_output1"


# Process all videos in the folder
process_videos_in_folder(input_folder, output_folder, ssim_threshold=0.02)

################################################################################################################################################################################################

import os
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision.models import resnet50

# Load ResNet model
model = resnet50(pretrained=True)
model = torch.nn.Sequential(*list(model.children())[:-1])  # Remove final classification layer
model.eval()

# Define transform
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

# Run extraction
root_folder = "keyframes_output"  # The folder containing subfolders for each video
extracted_features = extract_features_from_folder(root_folder)

# Save properly
if extracted_features:
    np.save("keyframe_features.npy", extracted_features)
    print(f"Saved {len(extracted_features)} keyframes.")
else:
    print("No features extracted. Check your dataset.")


################################################################################################################################################################################################


features = np.load("keyframe_features.npy", allow_pickle=True).item()

print(f"Loaded {len(features)} keyframes.")
print("Sample filenames:", list(features.keys())[:5])  # Show some filenames
print(f"Feature vector shape: {features[list(features.keys())[0]].shape}")

import numpy as np

# Load the saved feature file
features = np.load("keyframe_features.npy", allow_pickle=True)

# Check the type
print(f"Type of loaded data: {type(features)}")

# If it's a dictionary, convert it to a dictionary
if isinstance(features, np.ndarray):
    features = features.item()


sample_filename = list(features.keys())[0]  # Pick the first keyframe
print(f"Feature vector shape for {sample_filename}: {features[sample_filename].shape}")

# Print first 10 feature values
print(f"Feature vector sample: {features[sample_filename][:10]}")


