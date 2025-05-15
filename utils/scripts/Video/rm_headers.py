import pandas as pd

# Load the CSV file
df = pd.read_csv("/Users/anjelica/Project-MLPR/EmotionsInOurSarcasm/Model_1/data/embeddings/cnn/train/train4.csv")

# Check if 'Utterance' column exists
if 'Utterance' in df.columns:
    df = df.drop(columns=['Utterance'])

# Save the updated CSV
df.to_csv("/Users/anjelica/Project-MLPR/EmotionsInOurSarcasm/Model_1/data/embeddings/cnn/train/train4.csv", index=False)