import pandas as pd

# Load the CSV file
df = pd.read_csv("Embeddings\Video\sorted_video_embeddings2.csv")

# Check if 'Utterance' column exists
if 'Utterance' in df.columns:
    df = df.drop(columns=['Utterance'])

# Save the updated CSV
df.to_csv("video_embeddings_no_utterance.csv", index=False)
