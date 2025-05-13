import re
import json
import nltk
import warnings
import fasttext
import fasttext.util
import unicodedata
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
warnings.filterwarnings('ignore')

# Ensure stopwords are available
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

stop_words = set(stopwords.words('english'))

# --- Define Mappings (Customize as needed) ---
emotion_mapping = {
    'Anger': 0,
    'Surprise': 1,
    'Ridicule': 2,
    'Sad': 3
}
reverse_emotion_mapping = {v: k for k, v in emotion_mapping.items()}

# --- Text Preprocessing Functions ---
def normalize_hindi_spelling(text):
    """
    Normalize different spellings of Hindi words based on interchangeable letters.
    """
    replacements = {'q': 'k', 'z': 'j', 'o': 'u', 'w': 'v'}
    for old, new in replacements.items():
        text = text.replace(old, new)
    # Remove consecutive duplicate letters
    text = re.sub(r'(.)\1+', r'\1', text)
    return text

def clean_text_advanced(text, hindi_stopwords_file=None, remove_stopwords=True):
    """
    Advanced cleaning for code-mixed text (Hindi and English).
    Performs Unicode normalization, Hindi-specific normalization,
    lowercases English parts, removes special characters and stopwords,
    and applies stemming to English words.
    """
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize('NFC', text)
    text = normalize_hindi_spelling(text)
    
    # Lowercase all detected English words
    english_parts = re.findall(r'[a-zA-Z]+', text)
    for part in english_parts:
        text = text.replace(part, part.lower())
    
    # Retain Hindi (Devanagari), English letters and spaces:
    text = re.sub(r'[^\w\s\u0900-\u097F]', ' ', text)
    
    # Optionally remove stopwords
    if remove_stopwords:
        english_stopwords = set(stopwords.words('english'))
        if hindi_stopwords_file:
            # Load Hindi stopwords if a file is provided
            with open(hindi_stopwords_file, 'r', encoding='utf-8') as f:
                hindi_stopwords = set([line.strip() for line in f if line.strip()])
        else:
            hindi_stopwords = set()
        all_stopwords = english_stopwords.union(hindi_stopwords)
        words = text.split()
        words = [word for word in words if word not in all_stopwords]
        text = ' '.join(words)
    
    # Apply stemming to English words
    stemmer = PorterStemmer()
    words = text.split()
    stemmed_words = []
    for word in words:
        if re.match(r'^[a-zA-Z]+$', word):  # For English words
            stemmed_words.append(stemmer.stem(word))
        else:
            stemmed_words.append(word)
    text = ' '.join(stemmed_words)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def simple_clean_text(text):
    """
    Simple cleaning: remove special characters and lowercase text.
    """
    if not isinstance(text, str):
        return ""
    text = re.sub(r'[^\w\s]', '', text).lower()
    return text

def print_text_comparison(original_text, cleaned_text):
    """
    Print original and cleaned text along with words removed.
    """
    original_words = set(original_text.split())
    cleaned_words = set(cleaned_text.split())
    removed_words = original_words - cleaned_words
    print("Original Text:")
    print(original_text)
    print("\nCleaned Text:")
    print(cleaned_text)
    print("\nRemoved Words:")
    print(removed_words)

def create_dataframe(data, limit=None):
    """Create a dataframe from the loaded JSON data with error handling."""
    processed_examples = []
    count = 0
    for id, instance in data.items():
        if limit is not None and count >= limit:
            break
            
        # Check if required fields exist
        if "target_utterance" not in instance:
            print(f"Skipping instance {id}: missing 'target_utterance'")
            continue
            
        if "emotion" not in instance:
            print(f"Skipping instance {id}: missing 'emotion'")
            continue
            
        text = instance["target_utterance"]
        context = " ".join(instance["context_utterances"]) if "context_utterances" in instance else ""
        emotion = instance["emotion"]
        
        if emotion not in emotion_mapping:
            print(f"Skipping instance {id}: unknown emotion '{emotion}'")
            continue
            
        full_text = context + " " + text
        processed_examples.append({
            "text": full_text,
            "emotion": emotion_mapping[emotion],
            "target_utterance": text,
            "context_utterances": context,
            "full_text": full_text
        })
        count += 1
    
    if not processed_examples:
        raise ValueError("No valid instances found in the data!")
        
    df = pd.DataFrame(processed_examples)
    print(f"Created DataFrame with {len(df)} entries and {df.shape[1]} columns")
    print(df.head())
    
    return df

# --- Functions for FastText ---
def create_fasttext_file(dataframe, filename, text_column="text_cleaned", label_column="emotion"):
    """
    Create a file in the format FastText expects (e.g. "__label__<label> text")
    using the provided dataframe. This file is created from training data only.
    """
    with open(filename, 'w', encoding='utf-8') as f:
        for _, row in dataframe.iterrows():
            # FastText expects labels prefixed with __label__
            f.write(f"__label__{row[label_column]} {row[text_column]}\n")
    print(f"Created FastText input file: {filename}")

def train_fasttext_model(input_file, output_model_name="Outputs/emotion_model.bin", params=None):
    """
    Train a supervised FastText model with specified parameters.
    """
    if params is None:
        params = {
            'lr': 0.5,
            'epoch': 25,
            'wordNgrams': 2,
            'dim': 100,
            'loss': 'softmax'
        }
    print(f"Training FastText model with parameters: {params}")
    model = fasttext.train_supervised(
        input=input_file,
        lr=params['lr'],
        epoch=params['epoch'],
        wordNgrams=params['wordNgrams'],
        dim=params['dim'],
        loss=params['loss']
    )
    model.save_model(output_model_name)
    print(f"FastText model saved as: {output_model_name}")
    return model

def extract_features(model, dataframe, text_column="text_cleaned"):
    """
    Use a trained FastText model to extract embeddings (sentence vectors) for each row in dataframe.
    Returns the array of embeddings and corresponding labels.
    """
    X_features = []
    y_labels = []
    print("Extracting features...")
    for _, row in tqdm(dataframe.iterrows(), total=len(dataframe)):
        embedding = model.get_sentence_vector(row[text_column])
        X_features.append(embedding)
        y_labels.append(row['emotion'])
    X_features = np.array(X_features)
    y_labels = np.array(y_labels)
    print(f"Extracted features: shape = {X_features.shape}, labels shape = {y_labels.shape}")
    return X_features, y_labels

# --- Main Pipeline ---
def main(train_data_file, test_data_file, hindi_stopwords_file=None, use_advanced_cleaning=True):
    """
    Complete pipeline when data and labels are provided separately:
     1. Load training and test data (and labels) from CSV files.
     2. Merge them into a single DataFrame per split.
     3. Preprocess text.
     4. Create a FastText input file from the training data only.
     5. Train a supervised FastText model on training data.
     6. Extract embeddings on both training and test sets.
    """
    # --- 1. Load Data (assume CSV format) ---
    # Files should contain a column that holds the text (e.g. "text") and another holding labels
    train_data = pd.read_json(train_data_file)
    test_data = pd.read_json(test_data_file)
    
    train_df = create_dataframe(train_data)
    test_df = create_dataframe(test_data)

    print("Training data sample:")
    print(train_df.head())
    print("\nTest data sample:")
    print(test_df.head())
    
    # Convert emotion labels from text form to numeric mapping
    if train_df['emotion'].dtype == object:
        train_df['emotion'] = train_df['emotion'].map(emotion_mapping)
    if test_df['emotion'].dtype == object:
        test_df['emotion'] = test_df['emotion'].map(emotion_mapping)
    
    # --- 3. Preprocess Text ---
    cleaning_func = clean_text_advanced if use_advanced_cleaning else simple_clean_text
    train_df['text_cleaned'] = train_df['text'].apply(lambda x: cleaning_func(x, hindi_stopwords_file=hindi_stopwords_file))
    test_df['text_cleaned'] = test_df['text'].apply(lambda x: cleaning_func(x, hindi_stopwords_file=hindi_stopwords_file))
    
    print("\nSample text cleaning (Train Data):")
    print_text_comparison(train_df['text'].iloc[0], train_df['text_cleaned'].iloc[0])
    
    # --- 4. Create FastText Input File from Training Data ---
    fasttext_train_file = "fasttext_input_train.txt"
    create_fasttext_file(train_df, fasttext_train_file, text_column="text_cleaned", label_column="emotion")
    
    # --- 5. Train FastText Model on Training Data ---
    ft_model = train_fasttext_model(fasttext_train_file, output_model_name="Outputs/emotion_model.bin")
    
    # --- 6. Extract Embeddings from Both Training and Test Sets ---
    print("\nExtracting embeddings for training data...")
    X_train_features, y_train_labels = extract_features(ft_model, train_df, text_column="text_cleaned")
    
    print("\nExtracting embeddings for test data...")
    X_test_features, y_test_labels = extract_features(ft_model, test_df, text_column="text_cleaned")
    
    # Save the embeddings to CSV files.
    pd.DataFrame(X_train_features).to_csv("Outputs/Direct_fasttext_emb/train_features.csv", index=False)
    pd.DataFrame(X_test_features).to_csv("Outputs/Direct_fasttext_emb/test_features.csv", index=False)
    pd.DataFrame(y_train_labels, columns=['emotion']).to_csv("Outputs/Direct_fasttext_emb/y_train_labels.csv", index=False)
    pd.DataFrame(y_test_labels, columns=['emotion']).to_csv("Outputs/Direct_fasttext_emb/y_test_labels.csv", index=False)
    
    print("\nPipeline Completed Successfully!")
    return {
        "train_df": train_df,
        "test_df": test_df,
        "ft_model": ft_model,
        "X_train": X_train_features,
        "y_train": y_train_labels,
        "X_test": X_test_features,
        "y_test": y_test_labels
    }

def plot_embeddings(X_features, y_labels, method="tsne", perplexity=30, n_components=2, random_state=42):
    """
    Reduce dimensionality using TSNE (or PCA) and visualize clustering by emotion.
    """
    print(f"Reducing dimensionality using {method.upper()}...")
    if method.lower() == "tsne":
        reducer = TSNE(n_components=n_components, perplexity=perplexity, random_state=random_state)
    elif method.lower() == "pca":
        reducer = PCA(n_components=n_components, random_state=random_state)
    else:
        raise ValueError("Invalid method! Choose 'tsne' or 'pca'.")
    
    X_reduced = reducer.fit_transform(X_features)
    df_plot = pd.DataFrame(X_reduced, columns=['x', 'y'])
    df_plot['emotion'] = y_labels

    plt.figure(figsize=(10, 7))
    sns.scatterplot(x="x", y="y", hue="emotion", data=df_plot, palette="tab10", alpha=0.7, s=50, edgecolor="k")
    plt.title(f"Embedding Visualization using {method.upper()}")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.legend(title="Emotions", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig("Outputs/Images/embedding_cluster.png")
    plt.show()

# --- Example Usage ---
if __name__ == "__main__":
    # Set file paths for training and test data and labels.
    train_data_file = "Data/labeled_train_data.json"      # CSV with a column "text"
    test_data_file = "Data/test_data.json"          # CSV with a column "text"
    
    # Set additional parameters
    hindi_stopwords_file = None    # Provide path if you have Hindi stopwords file
    use_advanced_cleaning = True   # Change to False to use simple cleaning
    
    results = main(
        train_data_file=train_data_file,
        test_data_file=test_data_file,
        hindi_stopwords_file=hindi_stopwords_file,
        use_advanced_cleaning=use_advanced_cleaning
    )
    
    plot_embeddings(results["X_train"], results["y_train"], method="tsne")
    plot_embeddings(results["X_test"], results["y_test"], method="tsne")

# ____________________________________________________________________________________________