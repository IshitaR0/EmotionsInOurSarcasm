import json
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import os

# File paths for train, test, and val datasets
files = {
    "train": "/labeled_train_datafinal.json",
    "test": "/test_data.json",
    "val": '/val_data.json'
}

# Create a main output folder
main_output_folder = "emotion_analysis"
os.makedirs(main_output_folder, exist_ok=True)

# Process each file
for dataset_type, file_path in files.items():
    # Load the JSON data
    with open(file_path, "r") as file:
        data = json.load(file)

    # Extract the emotions
    emotions = [item["emotion"] for item in data.values()]

    # Count the occurrences of each emotion
    emotion_counts = Counter(emotions)

    # Create a subfolder for each dataset
    output_folder = os.path.join(main_output_folder, dataset_type)
    os.makedirs(output_folder, exist_ok=True)

    # Save the emotion counts to a text file
    txt_file_path = os.path.join(output_folder, f"emotion_counts_{dataset_type}.txt")
    with open(txt_file_path, "w") as txt_file:
        for emotion, count in emotion_counts.items():
            txt_file.write(f"{emotion}: {count}\n")

    # Plot the distribution
    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(emotion_counts.keys()), y=list(emotion_counts.values()), palette="viridis")
    plt.title(f"Emotion Distribution for {dataset_type.capitalize()}")
    plt.xlabel("Emotions")
    plt.ylabel("Frequency")
    plt.xticks(rotation=45)

    # Save the plot as an image
    png_file_path = os.path.join(output_folder, f"emotion_distribution_{dataset_type}.png")
    plt.savefig(png_file_path, dpi=300, bbox_inches="tight")
    plt.close()  # Close the figure to free memory

    print(f"Plot saved at: {png_file_path}")
    print(f"Emotion counts saved at: {txt_file_path}")
