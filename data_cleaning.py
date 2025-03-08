import numpy as np
import pandas as pd
import re
import json
import fasttext
import fasttext.util
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, f1_score
import warnings
warnings.filterwarnings('ignore')
import unicodedata
import nltk
from nltk.stem import PorterStemmer
from collections import defaultdict, Counter
from nltk.corpus import stopwords
nltk.download('stopwords')

print(stopwords.words('english'))

# Access English stopwords
stop_words = set(stopwords.words('english'))

# Step 1: Data Loading and Initial Exploration

def load_data(json_file_path):
    """Load and format the dataset from a JSON file."""
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # Convert to DataFrame and transpose if necessary
        df = pd.DataFrame.from_dict(data, orient='index')
        print(f"Loaded dataset with {len(df)} entries and {df.shape[1]} columns")
        print(df)
        print(type(df))
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def explore_dataset(df):
    """Basic exploration of the dataset."""
    print("\nDataset Information:")
    print(f"Number of entries: {len(df)}")
    print("\nColumns in dataset:")
    for col in df.columns:
        print(f"- {col}")
    
    if 'emotion' in df.columns:
        print("\nEmotion distribution:")
        emotion_counts = df['emotion'].value_counts()
        print(emotion_counts)
        
        # Visualize emotion distribution
        plt.figure(figsize=(10, 6))
        sns.barplot(x=emotion_counts.index, y=emotion_counts.values)
        plt.title('Distribution of Emotions in Dataset')
        plt.xlabel('Emotion')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('emotion_distribution.png')
        print("Emotion distribution chart saved as 'emotion_distribution.png'")
    
    return emotion_counts if 'emotion' in df.columns else None

def clean_text(text):
    """Clean and normalize code-mixed text."""
    if not isinstance(text, str):
        return ""
    # Convert to lowercase
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters, numbers
    # Keep Hindi characters (Devanagari Unicode range) and English characters
    text = re.sub(r'[^\w\s\u0900-\u097F]', ' ', text)
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

json_file_path = "dataset.json"
df = load_data(json_file_path)

# Ensure df is loaded before calling explore_dataset
if df is not None:
    explore_dataset(df)

def load_hindi_stopwords(file_path):
    """Load Hindi stopwords from a text file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            stopwords = [line.strip() for line in file if line.strip()]
        return set(stopwords)
    except FileNotFoundError:
        print(f"Warning: Stopwords file not found at {file_path}. Using empty stopword list.")
        return set()

def clean_text_without_expansion_or_normalization(text, hindi_stopwords_file=None, remove_stopwords=True):
    """
    Clean and normalize code-mixed text (Hindi and English) without normalization and expansion.
    """
    if not isinstance(text, str):
        return ""
    
    # Unicode normalization (combines characters and their diacritics)
    text = unicodedata.normalize('NFC', text)
    text = normalize_hindi_spelling(text)
    
    # Convert English text to lowercase (Hindi is not affected)
    english_parts = re.findall(r'[a-zA-Z]+', text)
    for part in english_parts:
        text = text.replace(part, part.lower())
    
    # Keep Hindi characters (Devanagari Unicode range), English characters, and spaces
    text = re.sub(r'[^\w\s\u0900-\u097F]', ' ', text)
    
    # Remove stopwords
    if remove_stopwords:
        from nltk.corpus import stopwords 
        english_stopwords = set(stopwords.words('english'))

        # Load Hindi stopwords if file is provided
        hindi_stopwords = load_hindi_stopwords(hindi_stopwords_file) if hindi_stopwords_file else set()
        
        # Combine stopwords
        all_stopwords = english_stopwords.union(hindi_stopwords)
        
        # Remove stopwords
        words = text.split()
        words = [word for word in words if word not in all_stopwords]
        text = ' '.join(words)
    
    # Stem English words
    stemmer = PorterStemmer()
    words = text.split()
    
    # Use regex to identify English words (assuming Hindi words use Devanagari script)
    stemmed_words = []
    for word in words:
        if re.match(r'^[a-zA-Z]+$', word):  # English word
            stemmed_words.append(stemmer.stem(word))
        else:  # Hindi word or mixed
            stemmed_words.append(word)
    
    text = ' '.join(stemmed_words)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def normalize_hindi_spelling(text):
    """Normalize different spellings of Hindi words based on common interchangeable letters."""
    # Replace interchangeable letters
    replacements = {
        'q': 'k', 
        'z': 'j',
        'o': 'u',
        'w': 'v'
    }
    
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    # Handle repeated sequential letters (keep only one occurrence)
    text = re.sub(r'(.)\1+', r'\1', text)
    
    return text
def create_hindi_normalization_dict(texts):
    """
    Create a dictionary for normalizing Hindi words based on frequency.
    """
    # Group similar words
    word_groups = defaultdict(list)
    
    # Process each text
    for text in texts:
        words = text.split()
        for word in words:
            # Skip English words
            if re.match(r'^[a-zA-Z]+$', word):
                continue
            
            # Normalize the word to create a key
            key = normalize_hindi_spelling(word)
            word_groups[key].append(word)
    
    # For each group, find the most frequent form
    normalization_dict = {}
    for key, variants in word_groups.items():
        if len(variants) > 1:  # Only process words with variants
            # Count occurrences of each variant
            variant_counts = Counter(variants)
            # Select the most frequent variant
            most_common = variant_counts.most_common(1)[0][0]
            
            # Create mappings for all variants
            for variant in variants:
                if variant != most_common:
                    normalization_dict[variant] = most_common
    
    return normalization_dict

def print_text_comparison(original_text, cleaned_text):
    # Split the original and cleaned texts into words for comparison
    original_words = set(original_text.split())
    cleaned_words = set(cleaned_text.split())
    
    # Find removed words (present in original but not in cleaned)
    removed_words = original_words - cleaned_words
    
    print("Original Text:")
    print(original_text)
    print("\nCleaned Text:")
    print(cleaned_text)
    print("\nRemoved Words:")
    print(removed_words)


# Combine 'target_utterance' and 'context_utterance' for the first five rows
combined_texts = []
for i in range(5):
    combined_text = str(df['target_utterance'].iloc[i]) + " " + str(df['context_utterances'].iloc[i])
    combined_texts.append(combined_text)

# Clean the combined text entries
cleaned_combined_texts = [clean_text_without_expansion_or_normalization(text, hindi_stopwords_file="stop_hinglish.txt") for text in combined_texts]

# Print cleaned texts for the first 5 entries
for i, cleaned_text in enumerate(cleaned_combined_texts):
    print(f"Cleaned Text {i+1}: {cleaned_text}")
# Example: Apply to the first entry in the dataset (target_utterance)
original_text = df['target_utterance'].iloc[0]  # Original target utterance
cleaned_text = clean_text_without_expansion_or_normalization(original_text, hindi_stopwords_file="stop_hinglish.txt")

# Print comparison and removed words
print_text_comparison(original_text, cleaned_text)


