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
import torch
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings('ignore')
import re
import json
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download English stopwords if not available
nltk.download('stopwords')


# Step 1: Data Loading and Initial Exploration

def load_data(json_file_path):
    """Load and format the dataset from a JSON file."""
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # Convert to DataFrame and transpose if necessary
        df = pd.DataFrame.from_dict(data, orient='index')
        print(f"Loaded dataset with {len(df)} entries and {df.shape[1]} columns")
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

# Step 2: Text Cleaning and Normalization

def clean_text(text):
    """Clean and normalize code-mixed text."""
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Keep Hindi characters (Devanagari Unicode range) and English characters
    text = re.sub(r'[^\w\s\u0900-\u097F]', ' ', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def apply_text_cleaning(df):
    """Apply text cleaning to all relevant text fields in the dataset."""
    # Create a copy to avoid modifying the original
    cleaned_df = df.copy()
    
    # Clean target utterances
    if 'target_utterance' in cleaned_df.columns:
        cleaned_df['clean_target_utterance'] = cleaned_df['target_utterance'].apply(clean_text)
    
    # Clean context utterances
    if 'context_utterances' in cleaned_df.columns:
        cleaned_df['clean_context_utterances'] = cleaned_df['context_utterances'].apply(
            lambda utterances: [clean_text(utt) for utt in utterances] if isinstance(utterances, list) else []
        )
    
    print(f"Text cleaning applied to {len(cleaned_df)} entries")
    return cleaned_df

# =====================================================
# Step 3: Language Detection and Tokenization
# =====================================================

def is_hindi(word):
    """Check if a word contains Devanagari characters (simplified approach)."""
    devanagari_pattern = re.compile(r'[\u0900-\u097F]')
    return bool(devanagari_pattern.search(word))

def tokenize_mixed_text(text):
    """Tokenize code-mixed text while preserving language information."""
    if not isinstance(text, str):
        return []
    
    # Simple word tokenization
    words = text.split()
    
    # Add language tag to each word
    tagged_words = []
    for word in words:
        if is_hindi(word):
            tagged_words.append((word, 'hi'))
        else:
            tagged_words.append((word, 'en'))
    
    return tagged_words

def analyze_language_mixture(df, text_column):
    """Analyze the distribution of Hindi and English words in the dataset."""
    if text_column not in df.columns:
        print(f"Column {text_column} not found in dataframe.")
        return
    
    hindi_count = 0
    english_count = 0
    total_words = 0
    
    for text in df[text_column]:
        if not isinstance(text, str):
            continue
        
        tokens = tokenize_mixed_text(text)
        for _, lang in tokens:
            if lang == 'hi':
                hindi_count += 1
            else:
                english_count += 1
            total_words += 1
    
    if total_words > 0:
        hindi_percent = (hindi_count / total_words) * 100
        english_percent = (english_count / total_words) * 100
        
        print(f"\nLanguage mixture analysis for {text_column}:")
        print(f"Total words: {total_words}")
        print(f"Hindi words: {hindi_count} ({hindi_percent:.2f}%)")
        print(f"English words: {english_count} ({english_percent:.2f}%)")
        
        # Visualize language distribution
        plt.figure(figsize=(8, 6))
        plt.pie([hindi_percent, english_percent], labels=['Hindi', 'English'], 
                autopct='%1.1f%%', startangle=90, colors=['#ff9999','#66b3ff'])
        plt.title(f'Language Distribution in {text_column}')
        plt.axis('equal')
        plt.savefig(f'language_distribution_{text_column}.png')
        print(f"Language distribution chart saved as 'language_distribution_{text_column}.png'")
    else:
        print(f"No valid text found in {text_column}")

# =====================================================
# Step 4: Loading FastText Wiki Embeddings
# =====================================================

def load_fasttext_models():
    """Download and load fastText models for Hindi and English."""
    print("\nDownloading and loading fastText models...")
    
    # Download and load English model
    print("Downloading English fastText model...")
    fasttext.util.download_model('en', if_exists='ignore')
    print("Loading English fastText model...")
    en_model = fasttext.load_model('cc.en.300.bin')
    
    # Download and load Hindi model
    print("Downloading Hindi fastText model...")
    fasttext.util.download_model('hi', if_exists='ignore')
    print("Loading Hindi fastText model...")
    hi_model = fasttext.load_model('cc.hi.300.bin')
    
    print("FastText models loaded successfully")
    return en_model, hi_model

def reduce_model_dimensions(en_model, hi_model, dim=100):
    """Reduce the dimensionality of fastText models to save memory and computation."""
    print(f"\nReducing model dimensions to {dim}...")
    fasttext.util.reduce_model(en_model, dim)
    fasttext.util.reduce_model(hi_model, dim)
    print(f"Models successfully reduced to {dim} dimensions")
    return en_model, hi_model

# =====================================================
# Step 5: Feature Extraction with FastText
# =====================================================

def get_word_embedding(word, lang, en_model, hi_model):
    """Get fastText embedding for a word based on its language."""
    try:
        if lang == 'hi':
            return hi_model.get_word_vector(word)
        else:  # Default to English
            return en_model.get_word_vector(word)
    except:
        # If embedding retrieval fails, try the other language model
        try:
            if lang == 'hi':
                return en_model.get_word_vector(word)
            else:
                return hi_model.get_word_vector(word)
        except:
            # Return zero vector if word not found in either model
            return np.zeros(en_model.get_dimension())

def get_sentence_embedding(text, en_model, hi_model):
    """Create sentence embedding by averaging word vectors."""
    if not isinstance(text, str) or not text:
        return np.zeros(en_model.get_dimension())
    
    tokens = tokenize_mixed_text(text)
    if not tokens:
        return np.zeros(en_model.get_dimension())
    
    word_vectors = [get_word_embedding(word, lang, en_model, hi_model) 
                    for word, lang in tokens]
    
    return np.mean(word_vectors, axis=0)

def extract_fasttext_features(df, en_model, hi_model):
    """Extract fastText features for utterances in the dataset."""
    print("\nExtracting fastText features...")
    
    # Extract features for target utterances
    if 'clean_target_utterance' in df.columns:
        print("Processing target utterances...")
        df['target_embedding'] = df['clean_target_utterance'].apply(
            lambda x: get_sentence_embedding(x, en_model, hi_model)
        )
    
    # Extract features for context utterances
    if 'clean_context_utterances' in df.columns:
        print("Processing context utterances...")
        df['context_embeddings'] = df['clean_context_utterances'].apply(
            lambda utterances: [get_sentence_embedding(utt, en_model, hi_model) 
                               for utt in utterances] if isinstance(utterances, list) else []
        )
        
        # Average the context embeddings
        df['avg_context_embedding'] = df['context_embeddings'].apply(
            lambda embeddings: np.mean(embeddings, axis=0) if len(embeddings) > 0 
                              else np.zeros(en_model.get_dimension())
        )
    
    print("Feature extraction completed")
    return df

# =====================================================
# Step 6: Feature Combination and Preparation
# =====================================================

def combine_features(df, embedding_dim):
    """Combine target and context embeddings into final feature vectors."""
    print("\nCombining features...")
    
    features = []
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        feature_vector = []
        
        # Add target embedding
        if 'target_embedding' in df.columns:
            feature_vector.extend(row['target_embedding'])
        
        # Add average context embedding
        if 'avg_context_embedding' in df.columns:
            feature_vector.extend(row['avg_context_embedding'])
        
        # If no features were added, create zero vector
        if not feature_vector:
            feature_vector = np.zeros(embedding_dim * 2)
        
        features.append(feature_vector)
    
    # Convert to numpy array
    feature_array = np.array(features)
    print(f"Combined features shape: {feature_array.shape}")
    return feature_array

def prepare_labels(df, label_col='emotion'):
    """Prepare labels for classification."""
    if label_col not in df.columns:
        print(f"Label column '{label_col}' not found in dataframe.")
        return None, None
    
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(df[label_col])
    
    print(f"\nLabel encoding:")
    for i, label in enumerate(label_encoder.classes_):
        print(f"  {label} -> {i}")
    
    return encoded_labels, label_encoder

# =====================================================
# Step 7: Creating PyTorch Dataset for Model Training
# =====================================================

class EmotionDataset(Dataset):
    """PyTorch Dataset for emotion classification."""
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def create_data_loaders(features, labels, test_size=0.2, val_size=0.1, batch_size=32, random_state=42):
    """Split data and create PyTorch DataLoaders."""
    # First split into train+val and test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        features, labels, test_size=test_size, random_state=random_state, stratify=labels
    )
    
    # Then split train+val into train and val
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, 
        test_size=val_size/(1-test_size),  # Adjust val_size to be relative to train_val size
        random_state=random_state,
        stratify=y_train_val
    )
    
    # Create datasets
    train_dataset = EmotionDataset(X_train, y_train)
    val_dataset = EmotionDataset(X_val, y_val)
    test_dataset = EmotionDataset(X_test, y_test)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    print("\nData split summary:")
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}")
    print(f"  Test samples: {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader

# =====================================================
# Step 8: Main Execution Function
# =====================================================

def preprocess_hinglish_dataset(json_file_path, embedding_dim=100):
    """Main function to preprocess Hinglish dataset with fastText embeddings."""
    print("Starting preprocessing pipeline...")
    
    # Step 1: Load and explore data
    df = load_data(json_file_path)
    if df is None:
        return None
    
    emotion_counts = explore_dataset(df)
    
    # Step 2: Clean text
    cleaned_df = apply_text_cleaning(df)
    
    # Step 3: Analyze language mixture
    if 'clean_target_utterance' in cleaned_df.columns:
        analyze_language_mixture(cleaned_df, 'clean_target_utterance')
    
    # Step 4: Load fastText models
    en_model, hi_model = load_fasttext_models()
    
    # Reduce model dimensions to save memory
    en_model, hi_model = reduce_model_dimensions(en_model, hi_model, dim=embedding_dim)
    
    # Step 5: Extract features
    feature_df = extract_fasttext_features(cleaned_df, en_model, hi_model)
    
    # Step 6: Combine features
    X = combine_features(feature_df, embedding_dim)
    
    # Prepare labels
    y, label_encoder = prepare_labels(feature_df)
    if y is None:
        return None
    
    # Step 7: Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(X, y)
    
    print("\nPreprocessing completed successfully!")
    
    return {
        'processed_df': feature_df,
        'features': X,
        'labels': y,
        'label_encoder': label_encoder,
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'en_model': en_model,
        'hi_model': hi_model
    }

# =====================================================
# Example Usage
# =====================================================

if __name__ == "__main__":
    # Replace with your dataset path
    json_file_path = "your_hinglish_dataset.json"
    
    # Run the preprocessing pipeline
    results = preprocess_hinglish_dataset(json_file_path)
    
    if results:
        print("\nProcessed data ready for model training!")
        # Now you can use the processed data for training models