import pandas as pd

df = pd.read_csv("Outputs/combined/embeddings_advanced.csv")

cols_to_drop = ['id', 'emotion', 'emotion_label', 'is_labeled']
existing_cols = [col for col in cols_to_drop if col in df.columns]


if 'emotion' in df.columns:
    df = df.drop(columns=existing_cols)

df.to_csv("Outputs/combined/final_textEmbed.csv", index=False)