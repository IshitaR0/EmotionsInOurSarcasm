import pandas as pd

file_path = "/Users/anjelica/Downloads/final_utterance_features_customcnn2048_train_4.csv"
df = pd.read_csv(file_path)

df['Utterance_Number'] = df['Utterance'].str.extract(r'(\d+)').astype(int)

# Sort by the numeric part
df_sorted = df.sort_values(by='Utterance_Number').drop(columns='Utterance_Number')

# Save to a new CSV file
sorted_file_path = "/Users/anjelica/Project-MLPR/EmotionsInOurSarcasm/Model_1/data/embeddings/cnn/train/train4.csv"
df_sorted.to_csv(sorted_file_path, index=False)

sorted_file_path


