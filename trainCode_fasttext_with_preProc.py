import json
import re
import nltk
import fasttext
import unicodedata
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from collections import defaultdict, Counter
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Ensure NLTK stopwords are available
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

# Load English stopwords
stop_words = set(stopwords.words('english'))

def load_data(json_file_path):
    """Load dataset from a JSON file."""
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"Loaded dataset with {len(data)} entries")
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def create_dataframe(data, limit=None):
    """Convert JSON data to Pandas DataFrame."""
    processed_examples = []
    for id, instance in list(data.items())[:limit]:
        text = instance["target_utterance"]
        context = " ".join(instance.get("context_utterances", []))
        emotion = instance["emotion"]
        full_text = context + " " + text
        processed_examples.append({
            "text": full_text,
            "emotion": emotion,
            "target_utterance": text,
            "context_utterances": context,
            "full_text": full_text
        })
    
    df = pd.DataFrame(processed_examples)
    print(f"Created DataFrame with {len(df)} entries and {df.shape[1]} columns")
    return df

def explore_dataset(df):
    """Perform basic dataset exploration."""
    print(f"\nDataset contains {len(df)} entries")
    print("\nColumn names:", df.columns.tolist())
    if 'emotion' in df.columns:
        emotion_counts = df['emotion'].value_counts()
        print("\nEmotion distribution:")
        print(emotion_counts)
        plt.figure(figsize=(10, 6))
        sns.barplot(x=emotion_counts.index, y=emotion_counts.values)
        plt.title('Emotion Distribution')
        plt.xlabel('Emotion')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('emotion_distribution.png')
        print("Saved emotion distribution plot.")

def clean_text(text):
    """Preprocess text by removing special characters and stopwords."""
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize('NFC', text)
    text = re.sub(r'[\W_]+', ' ', text).lower()
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

def create_fasttext_file(df, filename, text_column="text_cleaned", label_column="emotion"):
    """Generate FastText-compatible file."""
    with open(filename, 'w', encoding='utf-8') as f:
        for _, row in df.iterrows():
            f.write(f"__label__{row[label_column]} {row[text_column]}\n")
    print(f"Saved FastText input file: {filename}")

def train_fasttext_model(input_file, output_model_name="emotion_model.bin"):
    """Train FastText model."""
    model = fasttext.train_supervised(
        input=input_file, lr=0.5, epoch=25, wordNgrams=2, dim=100, loss='softmax'
    )
    model.save_model(output_model_name)
    print(f"Model saved as: {output_model_name}")
    return model

def extract_features(model, df, text_column="text_cleaned"):
    """Extract embeddings using FastText."""
    X = np.array([model.get_sentence_vector(text) for text in df[text_column]])
    y = df['emotion'].values
    print(f"Extracted features with shape: {X.shape}")
    return X, y

def main(json_file_path, limit=None):
    """Pipeline execution."""
    data = load_data(json_file_path)
    if not data:
        return
    df = create_dataframe(data, limit)
    explore_dataset(df)
    df['text_cleaned'] = df['text'].apply(clean_text)
    create_fasttext_file(df, "fasttext_input.txt")
    model = train_fasttext_model("fasttext_input.txt")
    X, y = extract_features(model, df)
    df.to_csv("processed_data.csv", index=False)
    print("Pipeline completed successfully.")
    return df, model, X, y

if __name__ == "__main__":
    main("train_data_final_plain.json", limit=60)
