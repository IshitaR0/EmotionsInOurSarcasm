import pandas as pd

# Load the uploaded Excel/CSV file
file_path = "MLPR raw data(video)/final_averaged_features2.csv"
df = pd.read_csv(file_path)

# Check the first few rows to inspect the data
# df.head()
# Extract the numeric part of the 'Utterance' for sorting
df['Utterance_Number'] = df['Utterance'].str.extract(r'(\d+)').astype(int)

# Sort by the numeric part
df_sorted = df.sort_values(by='Utterance_Number').drop(columns='Utterance_Number')

# Save to a new CSV file
sorted_file_path = "EmotionsInOurSarcasm/Embeddings/Video/sorted_video_embeddings2.csv"
df_sorted.to_csv(sorted_file_path, index=False)

sorted_file_path
