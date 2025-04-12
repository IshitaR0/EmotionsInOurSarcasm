import pandas as pd

# Load the CSV file
df = pd.read_csv("/Users/anjelica/EmotionsInOurSarcasm/Text/Features/scripts/labelled/features_advanced.csv")

# Check if 'Utterance' column exists
if 'emotion' in df.columns:
    df = df.drop(columns=['emotion'])

# Save the updated CSV
df.to_csv("embeddings_no_utterance.csv", index=False)