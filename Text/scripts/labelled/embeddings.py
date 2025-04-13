# import numpy as np
# import pandas as pd
# import re
# import json
# import fasttext
# import fasttext.util
# from tqdm import tqdm
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder
# from sklearn.metrics import classification_report, accuracy_score, f1_score
# import warnings
# warnings.filterwarnings('ignore')
# import unicodedata
# import nltk
# from nltk.stem import PorterStemmer
# from collections import defaultdict, Counter
# from nltk.corpus import stopwords

# # Download NLTK resources if not already downloaded
# try:
#     nltk.data.find('corpora/stopwords')
# except LookupError:
#     nltk.download('stopwords', quiet=True)

# # Access English stopwords
# stop_words = set(stopwords.words('english'))


# emotion_mapping = {
#     'Anger': 0,
#     'Surprise': 1,
#     'Ridicule': 2,
#     'Sad': 3
# }

# reverse_emotion_mapping = {v: k for k, v in emotion_mapping.items()}

# # Step 1: Data Loading and Initial Exploration
# def load_data(json_file_path):
#     """Load and format the dataset from a JSON file."""
#     try:
#         with open(json_file_path, 'r', encoding='utf-8') as f:
#             data = json.load(f)
#         print(f"Loaded dataset with {len(data)} entries")
#         return data
#     except Exception as e:
#         print(f"Error loading data: {e}")
#         return None

# def create_dataframe(data, limit=None):
#     """Create a dataframe from the loaded JSON data."""
#     processed_examples = []
#     count = 0
#     for id, instance in data.items():
#         if limit is not None and count >= limit:
#             break
#         text = instance["target_utterance"]
#         context = " ".join(instance["context_utterances"]) if "context_utterances" in instance else ""
#         emotion = instance["emotion"]
#         full_text = context + " " + text
#         if emotion not in emotion_mapping:
#             print(f"Skipping instance {id} due to unknown emotion: {emotion}")
#             continue
#         processed_examples.append({
#             "text": full_text,
#             "emotion": emotion_mapping[emotion],
#             "target_utterance": text,
#             "context_utterances": context,
#             "full_text": full_text
#         })
#         count += 1
    
#     train_df = pd.DataFrame(processed_examples)
#     print(f"Created DataFrame with {len(train_df)} entries and {train_df.shape[1]} columns")
#     print(train_df.head())
    
#     return train_df

# def explore_dataset(train_df):
#     """Basic exploration of the dataset."""
#     print("\nDataset Information:")
#     print(f"Number of entries: {len(train_df)}")
#     print("\nColumns in dataset:")
#     for col in train_df.columns:
#         print(f"- {col}")
    
#     if 'emotion' in train_df.columns:
#         print("\nEmotion distribution:")
#         emotion_counts = train_df['emotion'].value_counts()
#         print(emotion_counts)
        
#         # Visualize emotion distribution
#         plt.figure(figsize=(10, 6))
#         sns.barplot(x=emotion_counts.index, y=emotion_counts.values)
#         plt.title('Distribution of Emotions in Dataset')
#         plt.xlabel('Emotion')
#         plt.ylabel('Count')
#         plt.xticks(rotation=45)
#         plt.tight_layout()
#         plt.savefig('Outputs/labeled/emotion_distribution.png')
#         print("Emotion distribution chart saved as 'emotion_distribution.png'")
    
#     return emotion_counts if 'emotion' in train_df.columns else None

# # Step 2: Text Preprocessing Functions
# def load_hindi_stopwords(file_path):
#     """Load Hindi stopwords from a text file."""
#     try:
#         with open(file_path, 'r', encoding='utf-8') as file:
#             stopwords = [line.strip() for line in file if line.strip()]
#         return set(stopwords)
#     except FileNotFoundError:
#         print(f"Warning: Stopwords file not found at {file_path}. Using empty stopword list.")
#         return set()

# def normalize_hindi_spelling(text):
#     """Normalize different spellings of Hindi words based on common interchangeable letters."""
#     # Replace interchangeable letters
#     replacements = {
#         'q': 'k', 
#         'z': 'j',
#         'o': 'u',
#         'w': 'v'
#     }
    
#     for old, new in replacements.items():
#         text = text.replace(old, new)
    
#     # Handle repeated sequential letters (keep only one occurrence)
#     text = re.sub(r'(.)\1+', r'\1', text)
    
#     return text

# def create_hindi_normalization_dict(texts):
#     """Create a dictionary for normalizing Hindi words based on frequency."""
#     # Group similar words
#     word_groups = defaultdict(list)
    
#     # Process each text
#     for text in texts:
#         words = text.split()
#         for word in words:
#             # Skip English words
#             if re.match(r'^[a-zA-Z]+$', word):
#                 continue
            
#             # Normalize the word to create a key
#             key = normalize_hindi_spelling(word)
#             word_groups[key].append(word)
    
#     # For each group, find the most frequent form
#     normalization_dict = {}
#     for key, variants in word_groups.items():
#         if len(variants) > 1:  # Only process words with variants
#             # Count occurrences of each variant
#             variant_counts = Counter(variants)
#             # Select the most frequent variant
#             most_common = variant_counts.most_common(1)[0][0]
            
#             # Create mappings for all variants
#             for variant in variants:
#                 if variant != most_common:
#                     normalization_dict[variant] = most_common
    
#     return normalization_dict

# def clean_text_advanced(text, hindi_stopwords_file=None, remove_stopwords=True):

#     """
#     Advanced cleaning for code-mixed text (Hindi and English).
#     Includes Unicode normalization, stopword removal, and stemming for English words.
#     """
#     if not isinstance(text, str):
#         return ""
    
#     # Unicode normalization (combines characters and their diacritics)
#     text = unicodedata.normalize('NFC', text)
#     text = normalize_hindi_spelling(text)
    
#     # Convert English text to lowercase (Hindi is not affected)
#     english_parts = re.findall(r'[a-zA-Z]+', text)
#     for part in english_parts:
#         text = text.replace(part, part.lower())
    
#     # Remove special characters, keeping Hindi characters (Devanagari Unicode range), English characters, and spaces
#     text = re.sub(r'[^\w\s\u0900-\u097F]', ' ', text)
    
#     # Remove stopwords
#     if remove_stopwords:
#         english_stopwords = set(stopwords.words('english'))

#         # Load Hindi stopwords if file is provided
#         hindi_stopwords = load_hindi_stopwords(hindi_stopwords_file) if hindi_stopwords_file else set()
        
#         # Combine stopwords
#         all_stopwords = english_stopwords.union(hindi_stopwords)
        
#         # Remove stopwords
#         words = text.split()
#         words = [word for word in words if word not in all_stopwords]
#         text = ' '.join(words)
    
#     # Stem English words
#     stemmer = PorterStemmer()
#     words = text.split()
    
#     # Use regex to identify English words (assuming Hindi words use Devanagari script)
#     stemmed_words = []
#     for word in words:
#         if re.match(r'^[a-zA-Z]+$', word):  # English word
#             stemmed_words.append(stemmer.stem(word))
#         else:  # Hindi word or mixed
#             stemmed_words.append(word)
    
#     text = ' '.join(stemmed_words)
    
#     # Normalize whitespace
#     text = re.sub(r'\s+', ' ', text).strip()
    
#     return text

# def simple_clean_text(text):
#     """Simple cleaning function (similar to the one in the second snippet)."""
#     if not isinstance(text, str):
#         return ""
#     # Remove special characters and convert to lowercase
#     text = re.sub(r'[^\w\s]', '', text).lower()
#     return text

# def print_text_comparison(original_text, cleaned_text):
#     """Print comparison between original and cleaned text."""
#     # Split the original and cleaned texts into words for comparison
#     original_words = set(original_text.split())
#     cleaned_words = set(cleaned_text.split())
    
#     # Find removed words (present inˀ original but not in cleaned)
#     removed_words = original_words - cleaned_words
    
#     print("Original Text:")
#     print(original_text)
#     print("\nCleaned Text:")
#     print(cleaned_text)
#     print("\nRemoved Words:")
#     print(removed_words)

# # Step 3: FastText Model Training and Feature Extraction
# def create_fasttext_file(dataframe, filename, text_column="text_cleaned", label_column="emotion"):
#     """Create a text file in the format required by FastText."""
#     with open(filename, 'w', encoding='utf-8') as f:
#         for _, row in dataframe.iterrows():
#             f.write(f"__label__{row[label_column]} {row[text_column]}\n")
#     print(f"Created FastText input file: {filename}")

# def train_fasttext_model(input_file, output_model_name="Outputs/labeled/emotion_model.bin", params=None):
#     """Train a FastText model with the given parameters."""
#     if params is None:
#         params = {
#             'lr': 0.5,
#             'epoch': 25,
#             'wordNgrams': 2,
#             'dim': 100,
#             'loss': 'softmax'
#         }
    
#     print(f"Training FastText model with parameters: {params}")
#     model = fasttext.train_supervised(
#         input=input_file,
#         lr=params['lr'],
#         epoch=params['epoch'],
#         wordNgrams=params['wordNgrams'],
#         dim=params['dim'],
#         loss=params['loss']
#     )
    
#     # Save the model
#     model.save_model(output_model_name)
#     print(f"FastText model saved as: {output_model_name}")
#     return model

# def extract_features(model, dataframe, text_column="text_cleaned"):
#     """Extract features using the trained FastText model."""
#     X_features = []
#     y_labels = []
    
#     print("Extracting features...")
#     for _, row in tqdm(dataframe.iterrows(), total=len(dataframe)):
#         # Get text embedding (sentence vector)
#         embedding = model.get_sentence_vector(row[text_column])
#         X_features.append(embedding)
#         y_labels.append(row['emotion'])
    
#     X_features = np.array(X_features)
#     y_labels = np.array(y_labels)
    
#     print(f"Features extracted: X shape = {X_features.shape}, y shape = {y_labels.shape}")
#     return X_features, y_labels

# # Step 4: Main Execution Pipeline
# def main(json_file_path, hindi_stopwords_file=None, limit=None, use_advanced_cleaning=True):
#     """Main execution pipeline."""
#     # 1. Load data
#     print("\n=== Loading Data ===")
#     data = load_data(json_file_path)
#     if data is None:
#         return
    
#     # 2. Create DataFrame
#     print("\n=== Creating DataFrame ===")
#     train_df = create_dataframe(data, limit=limit)
    
    
#     # 3. Explore dataset
#     print("\n=== Exploring Dataset ===")
#     explore_dataset(train_df)
    
#     # 4. Preprocess text
#     print("\n=== Preprocessing Text ===")
#     if use_advanced_cleaning:
#         train_df['text_cleaned'] = train_df['text'].apply(
#             lambda x: clean_text_advanced(x, hindi_stopwords_file=hindi_stopwords_file)
#         )
#         cleaning_method = "advanced"
#     else:
#         train_df['text_cleaned'] = train_df['text'].apply(simple_clean_text)
#         cleaning_method = "simple"
    
#     print(f"Applied {cleaning_method} cleaning to text")
    
#     # 5. Print sample comparison
#     if len(train_df) > 0:
#         print("\n=== Sample Text Cleaning Comparison ===")
#         print_text_comparison(train_df['text'].iloc[0], train_df['text_cleaned'].iloc[0])
    
#     # 6. Create FastText input file
#     print("\n=== Creating FastText Input File ===")
#     fasttext_file = f"Outputs/labeled/fasttext_input_{cleaning_method}.txt"
#     create_fasttext_file(train_df, fasttext_file)
    
#     # 7. Train FastText model
#     print("\n=== Training FastText Model ===")
#     model = train_fasttext_model(fasttext_file, output_model_name=f"emotion_model_{cleaning_method}.bin")
    
#     # 8. Extract features
#     print("\n=== Extracting Features ===")
#     X_features, y_labels = extract_features(model, train_df)
#     # Save y_labels
#     pd.DataFrame(y_labels, columns=['emotion']).to_csv("Outputs/labeled/y_labels.csv", index=False)
#     print("y_labels saved to 'Outputs/labeled/y_labels.csv'")
#     # 9. Print some feature statistics
#     print("\n=== Feature Statistics ===")
#     print(f"Feature dimensionality: {X_features.shape[1]}")
#     print(f"Feature mean: {np.mean(X_features)}")
#     print(f"Feature std: {np.std(X_features)}")
    
#     # 10. Save processed data
#     print("\n=== Saving Processed Data ===")
#     train_df.to_csv(f"Outputs/labeled/processed_data_{cleaning_method}.csv", index=False)
    
#     # 11. Save feature data
#     print("\n=== Saving Feature Data ===")
#     feature_df = pd.DataFrame(X_features)
#     feature_df['emotion'] = y_labels
#     feature_df.to_csv(f"Outputs/labeled/features_{cleaning_method}.csv", index=False)
    
#     print("\n=== Pipeline Completed Successfully ===")
#     return train_df, model, X_features, y_labels

# # Example usage
# if __name__ == "__main__":
#     # Set parameters
#     json_file_path = "Data/labeled_train_data.json"
#     hindi_stopwords_file = "stop_hinglish.txt"  # Set to None if not available
#     limit = None # Set to None to process all data
#     use_advanced_cleaning = True  # Set to False to use simple cleaning
    
#     # Run the pipeline
#     train_df, model, X_features, y_labels = main(
#         json_file_path=json_file_path, 
#         hindi_stopwords_file=hindi_stopwords_file,
#         limit=limit,
#         use_advanced_cleaning=use_advanced_cleaning
#     )

# # If i use Unsupervised fasttext model for sentence vector embedding extraction and not training a classification model.
# # def train_unsupervised_fasttext(input_file, output_model_name="unsupervised_model.bin"):
# #     """Train an unsupervised FastText model to get sentence embeddings."""
# #     model = fasttext.train_unsupervised(input=input_file, model="skipgram", dim=100)
    
# #     # Save the model
# #     model.save_model(output_model_name)
# #     print(f"FastText unsupervised model saved as: {output_model_name}")
# #     return model

# # def extract_sentence_embeddings(model, dataframe, text_column="text_cleaned"):
# #     """Extract sentence embeddings using an unsupervised FastText model."""
# #     X_features = []
    
# #     print("Extracting sentence embeddings...")
# #     for _, row in dataframe.iterrows():
# #         embedding = model.get_sentence_vector(row[text_column])
# #         X_features.append(embedding)
    
# #     X_features = np.array(X_features)
# #     print(f"Extracted embeddings shape: {X_features.shape}")
# #     return X_features

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.manifold import TSNE
# from sklearn.decomposition import PCA
# from collections import Counter

# def plot_embeddings(X_features, y_labels, method="tsne", perplexity=30, n_components=2, random_state=42):
#     """
#     Reduce the dimensionality of FastText sentence embeddings and visualize the clustering by emotion.

#     Parameters:
#         X_features (numpy.ndarray): Sentence embeddings.
#         y_labels (numpy.ndarray): Corresponding emotion labels.
#         method (str): Dimensionality reduction method ('tsne' or 'pca').
#         perplexity (int): Perplexity parameter for t-SNE (ignored if using PCA).
#         n_components (int): Number of dimensions to reduce to (default: 2).
#         random_state (int): Random seed for reproducibility.
#     """
#     print(f"Reducing dimensionality using {method.upper()}...")
    
#     if method.lower() == "tsne":
#         reducer = TSNE(n_components=n_components, perplexity=perplexity, random_state=random_state)
#     elif method.lower() == "pca":
#         reducer = PCA(n_components=n_components, random_state=random_state)
#     else:
#         raise ValueError("Invalid method! Choose 'tsne' or 'pca'.")
    
#     X_reduced = reducer.fit_transform(X_features)

#     # Convert to DataFrame for visualization
#     df_plot = pd.DataFrame(X_reduced, columns=['x', 'y'])
#     df_plot['emotion'] = y_labels

#     plt.figure(figsize=(10, 7))
#     sns.scatterplot(
#         x="x", y="y", hue="emotion", data=df_plot, palette="tab10", alpha=0.7, s=50, edgecolor="k"
#     )
#     plt.savefig("Outputs/labeled/emotions_frequency.png")
    
#     # Add emotion cluster labels if possible
#     most_common_emotions = [e[0] for e in Counter(y_labels).most_common()]
#     for emotion in most_common_emotions:
#         subset = df_plot[df_plot['emotion'] == emotion]
#         centroid_x = subset["x"].mean()
#         centroid_y = subset["y"].mean()
#         plt.text(centroid_x, centroid_y, emotion, fontsize=12, fontweight='bold', ha='center', va='center')

#     plt.title(f"Sentence Embeddings Visualization using {method.upper()}")
#     plt.xlabel("Component 1")
#     plt.ylabel("Component 2")
#     plt.legend(title="Emotions", bbox_to_anchor=(1.05, 1), loc="upper left")
#     plt.tight_layout()
#     plt.savefig("Outputs/labeled/_graph_cluster_of_emotions.png")
#     plt.show()

# # Example usage
# plot_embeddings(X_features, y_labels, method="tsne")  # Use "pca" for PCA visualization



# TRIED WITH UNSUPERVISED MODEL, DOESN'T SEEM TO WORK
# import numpy as np
# import pandas as pd
# import re
# import json
# import fasttext
# import fasttext.util
# from tqdm import tqdm
# import matplotlib.pyplot as plt
# import seaborn as sns
# import os
# import warnings
# warnings.filterwarnings('ignore')
# import unicodedata
# import nltk
# from nltk.stem import PorterStemmer
# from collections import defaultdict, Counter
# from nltk.corpus import stopwords

# # Download NLTK resources if not already downloaded
# try:
#     nltk.data.find('corpora/stopwords')
# except LookupError:
#     nltk.download('stopwords', quiet=True)

# # Access English stopwords
# stop_words = set(stopwords.words('english'))

# # Create output directories if they don't exist
# os.makedirs("Outputs/combined", exist_ok=True)

# # Set up emotion mapping like in original code
# emotion_mapping = {
#     'Anger': 0,
#     'Surprise': 1,
#     'Ridicule': 2,
#     'Sad': 3
# }

# reverse_emotion_mapping = {v: k for k, v in emotion_mapping.items()}

# # Text preprocessing functions from original code
# def load_hindi_stopwords(file_path):
#     """Load Hindi stopwords from a text file."""
#     try:
#         with open(file_path, 'r', encoding='utf-8') as file:
#             stopwords = [line.strip() for line in file if line.strip()]
#         return set(stopwords)
#     except FileNotFoundError:
#         print(f"Warning: Stopwords file not found at {file_path}. Using empty stopword list.")
#         return set()

# def normalize_hindi_spelling(text):
#     """Normalize different spellings of Hindi words based on common interchangeable letters."""
#     # Replace interchangeable letters
#     replacements = {
#         'q': 'k', 
#         'z': 'j',
#         'o': 'u',
#         'w': 'v'
#     }
    
#     for old, new in replacements.items():
#         text = text.replace(old, new)
    
#     # Handle repeated sequential letters (keep only one occurrence)
#     text = re.sub(r'(.)\1+', r'\1', text)
    
#     return text

# def clean_text_advanced(text, hindi_stopwords_file=None, remove_stopwords=True):
#     """
#     Advanced cleaning for code-mixed text (Hindi and English).
#     Includes Unicode normalization, stopword removal, and stemming for English words.
#     """
#     if not isinstance(text, str):
#         return ""
    
#     # Unicode normalization (combines characters and their diacritics)
#     text = unicodedata.normalize('NFC', text)
#     text = normalize_hindi_spelling(text)
    
#     # Convert English text to lowercase (Hindi is not affected)
#     english_parts = re.findall(r'[a-zA-Z]+', text)
#     for part in english_parts:
#         text = text.replace(part, part.lower())
    
#     # Remove special characters, keeping Hindi characters (Devanagari Unicode range), English characters, and spaces
#     text = re.sub(r'[^\w\s\u0900-\u097F]', ' ', text)
    
#     # Remove stopwords
#     if remove_stopwords:
#         english_stopwords = set(stopwords.words('english'))

#         # Load Hindi stopwords if file is provided
#         hindi_stopwords = load_hindi_stopwords(hindi_stopwords_file) if hindi_stopwords_file else set()
        
#         # Combine stopwords
#         all_stopwords = english_stopwords.union(hindi_stopwords)
        
#         # Remove stopwords
#         words = text.split()
#         words = [word for word in words if word not in all_stopwords]
#         text = ' '.join(words)
    
#     # Stem English words
#     stemmer = PorterStemmer()
#     words = text.split()
    
#     # Use regex to identify English words (assuming Hindi words use Devanagari script)
#     stemmed_words = []
#     for word in words:
#         if re.match(r'^[a-zA-Z]+$', word):  # English word
#             stemmed_words.append(stemmer.stem(word))
#         else:  # Hindi word or mixed
#             stemmed_words.append(word)
    
#     text = ' '.join(stemmed_words)
    
#     # Normalize whitespace
#     text = re.sub(r'\s+', ' ', text).strip()
    
#     return text

# def simple_clean_text(text):
#     """Simple cleaning function."""
#     if not isinstance(text, str):
#         return ""
#     # Remove special characters and convert to lowercase
#     text = re.sub(r'[^\w\s]', '', text).lower()
#     return text

# # Load and process data functions
# def load_labeled_data(json_file_path):
#     """Load and format the labeled dataset from a JSON file."""
#     try:
#         with open(json_file_path, 'r', encoding='utf-8') as f:
#             data = json.load(f)
#         print(f"Loaded labeled dataset with {len(data)} entries")
#         return data
#     except Exception as e:
#         print(f"Error loading labeled data: {e}")
#         return None

# def load_unlabeled_data(json_file_path):
#     """Load and format the unlabeled dataset from a JSON file."""
#     try:
#         with open(json_file_path, 'r', encoding='utf-8') as f:
#             data = json.load(f)
#         print(f"Loaded unlabeled dataset with {len(data)} entries")
#         return data
#     except Exception as e:
#         print(f"Error loading unlabeled data: {e}")
#         return None

# def create_labeled_dataframe(data, limit=None):
#     """Create a dataframe from the loaded JSON labeled data."""
#     processed_examples = []
#     count = 0
#     for id, instance in data.items():
#         if limit is not None and count >= limit:
#             break
#         text = instance["target_utterance"]
#         context = " ".join(instance["context_utterances"]) if "context_utterances" in instance else ""
#         emotion = instance["emotion"]
#         full_text = context + " " + text
#         if emotion not in emotion_mapping:
#             print(f"Skipping instance {id} due to unknown emotion: {emotion}")
#             continue
#         processed_examples.append({
#             "id": id,
#             "text": full_text,
#             "emotion": emotion_mapping[emotion],
#             "target_utterance": text,
#             "context_utterances": context,
#             "full_text": full_text,
#             "is_labeled": True
#         })
#         count += 1
    
#     df = pd.DataFrame(processed_examples)
#     print(f"Created labeled DataFrame with {len(df)} entries")
    
#     return df

# def create_unlabeled_dataframe(data, limit=None):
#     """Create a dataframe from the loaded JSON unlabeled data."""
#     processed_examples = []
#     count = 0
#     for id, instance in data.items():
#         if limit is not None and count >= limit:
#             break
#         text = instance["target_utterance"]
#         context = " ".join(instance["context_utterances"]) if "context_utterances" in instance else ""
#         full_text = context + " " + text
#         processed_examples.append({
#             "id": id,
#             "text": full_text,
#             "emotion": -1,  # Use -1 to indicate unlabeled
#             "target_utterance": text,
#             "context_utterances": context,
#             "full_text": full_text,
#             "is_labeled": False
#         })
#         count += 1
    
#     df = pd.DataFrame(processed_examples)
#     print(f"Created unlabeled DataFrame with {len(df)} entries")
    
#     return df

# # FastText unsupervised model functions
# def create_fasttext_corpus(dataframe, filename, text_column="text_cleaned"):
#     """Create a text file to train unsupervised FastText model."""
#     with open(filename, 'w', encoding='utf-8') as f:
#         for _, row in dataframe.iterrows():
#             f.write(f"{row[text_column]}\n")
#     print(f"Created FastText corpus file: {filename}")

# def train_unsupervised_fasttext(input_file, output_model_name="unsupervised_model.bin", params=None):
#     """Train an unsupervised FastText model to get sentence embeddings."""
#     if params is None:
#         params = {
#             'model': 'skipgram',  # 'skipgram' or 'cbow'
#             'dim': 100,           # embedding dimension
#             'epoch': 25,          # number of training epochs
#             'lr': 0.05,           # learning rate
#             'wordNgrams': 2       # max length of word ngrams
#         }
    
#     print(f"Training unsupervised FastText model with parameters: {params}")
#     model = fasttext.train_unsupervised(
#         input=input_file,
#         model=params['model'],
#         dim=params['dim'],
#         epoch=params['epoch'],
#         lr=params['lr'],
#         wordNgrams=params['wordNgrams']
#     )
    
#     # Save the model
#     model.save_model(output_model_name)
#     print(f"FastText unsupervised model saved as: {output_model_name}")
#     return model

# def extract_sentence_embeddings(model, dataframe, text_column="text_cleaned"):
#     """Extract sentence embeddings using an unsupervised FastText model."""
#     X_features = []
    
#     print("Extracting sentence embeddings...")
#     for _, row in tqdm(dataframe.iterrows(), total=len(dataframe)):
#         embedding = model.get_sentence_vector(row[text_column])
#         X_features.append(embedding)
    
#     X_features = np.array(X_features)
#     print(f"Extracted embeddings shape: {X_features.shape}")
#     return X_features

# def plot_embeddings(X_features, y_labels, is_labeled, method="tsne", perplexity=30, n_components=2, random_state=42):
#     """
#     Reduce the dimensionality of FastText sentence embeddings and visualize the clustering.
#     Distinguishes between labeled and unlabeled data points.
#     """
#     print(f"Reducing dimensionality using {method.upper()}...")
    
#     from sklearn.manifold import TSNE
#     from sklearn.decomposition import PCA
    
#     if method.lower() == "tsne":
#         reducer = TSNE(n_components=n_components, perplexity=perplexity, random_state=random_state)
#     elif method.lower() == "pca":
#         reducer = PCA(n_components=n_components, random_state=random_state)
#     else:
#         raise ValueError("Invalid method! Choose 'tsne' or 'pca'.")
    
#     X_reduced = reducer.fit_transform(X_features)

#     # Convert to DataFrame for visualization
#     df_plot = pd.DataFrame(X_reduced, columns=['x', 'y'])
#     df_plot['emotion'] = y_labels
#     df_plot['is_labeled'] = is_labeled
    
#     # Create two plots
    
#     # Plot 1: Color by labeled vs unlabeled
#     plt.figure(figsize=(10, 7))
#     sns.scatterplot(
#         x="x", y="y", hue="is_labeled", data=df_plot, palette=["gray", "blue"], 
#         alpha=0.7, s=50, edgecolor="k"
#     )
#     plt.title(f"Sentence Embeddings: Labeled vs Unlabeled ({method.upper()})")
#     plt.xlabel("Component 1")
#     plt.ylabel("Component 2")
#     plt.legend(title="Data Type", labels=["Unlabeled", "Labeled"])
#     plt.tight_layout()
#     plt.savefig(f"Outputs/combined/embeddings_labeled_vs_unlabeled_{method}.png")
    
#     # Plot 2: For labeled data only, color by emotion
#     plt.figure(figsize=(10, 7))
#     labeled_df = df_plot[df_plot['is_labeled'] == True].copy()
#     labeled_df['emotion_name'] = labeled_df['emotion'].map(reverse_emotion_mapping)
    
#     sns.scatterplot(
#         x="x", y="y", hue="emotion_name", data=labeled_df, palette="tab10", 
#         alpha=0.7, s=50, edgecolor="k"
#     )
    
#     # Add emotion cluster labels
#     for emotion in labeled_df['emotion_name'].unique():
#         subset = labeled_df[labeled_df['emotion_name'] == emotion]
#         centroid_x = subset["x"].mean()
#         centroid_y = subset["y"].mean()
#         plt.text(centroid_x, centroid_y, emotion, fontsize=12, fontweight='bold', ha='center', va='center')

#     plt.title(f"Sentence Embeddings: Emotions ({method.upper()})")
#     plt.xlabel("Component 1")
#     plt.ylabel("Component 2")
#     plt.legend(title="Emotions", bbox_to_anchor=(1.05, 1), loc="upper left")
#     plt.tight_layout()
#     plt.savefig(f"Outputs/combined/embeddings_emotions_{method}.png")
    
#     print(f"Embedding visualization plots saved to 'Outputs/combined/'")

# # Main pipeline
# def main(labeled_json_path, unlabeled_json_path, hindi_stopwords_file=None, 
#          limit_labeled=None, limit_unlabeled=None, use_advanced_cleaning=True):
#     """Main execution pipeline for processing both labeled and unlabeled data."""
#     # 1. Load data
#     print("\n=== Loading Data ===")
#     labeled_data = load_labeled_data(labeled_json_path)
#     unlabeled_data = load_unlabeled_data(unlabeled_json_path)
    
#     if labeled_data is None or unlabeled_data is None:
#         return
    
#     # 2. Create DataFrames
#     print("\n=== Creating DataFrames ===")
#     labeled_df = create_labeled_dataframe(labeled_data, limit=limit_labeled)
#     unlabeled_df = create_unlabeled_dataframe(unlabeled_data, limit=limit_unlabeled)
    
#     # 3. Combine DataFrames
#     combined_df = pd.concat([labeled_df, unlabeled_df], ignore_index=True)
#     print(f"Combined DataFrame has {len(combined_df)} entries")
#     print(f"Labeled: {sum(combined_df['is_labeled'])} | Unlabeled: {sum(~combined_df['is_labeled'])}")
    
#     # 4. Preprocess text
#     print("\n=== Preprocessing Text ===")
#     if use_advanced_cleaning:
#         combined_df['text_cleaned'] = combined_df['text'].apply(
#             lambda x: clean_text_advanced(x, hindi_stopwords_file=hindi_stopwords_file)
#         )
#         cleaning_method = "advanced"
#     else:
#         combined_df['text_cleaned'] = combined_df['text'].apply(simple_clean_text)
#         cleaning_method = "simple"
    
#     print(f"Applied {cleaning_method} cleaning to text")
    
#     # 5. Create FastText corpus
#     print("\n=== Creating FastText Corpus ===")
#     fasttext_corpus = f"Outputs/combined/fasttext_corpus_{cleaning_method}.txt"
#     create_fasttext_corpus(combined_df, fasttext_corpus)
    
#     # 6. Train FastText unsupervised model
#     print("\n=== Training FastText Unsupervised Model ===")
#     unsupervised_model = train_unsupervised_fasttext(
#         fasttext_corpus, 
#         output_model_name=f"Outputs/combined/unsupervised_model_{cleaning_method}.bin"
#     )
    
#     # 7. Extract sentence embeddings
#     print("\n=== Extracting Sentence Embeddings ===")
#     X_features = extract_sentence_embeddings(unsupervised_model, combined_df)
#     y_labels = combined_df['emotion'].values
#     is_labeled = combined_df['is_labeled'].values
    
#     # 8. Save processed data
#     print("\n=== Saving Processed Data ===")
#     combined_df.to_csv(f"Outputs/combined/processed_data_{cleaning_method}.csv", index=False)
    
#     # 9. Save embeddings with metadata
#     print("\n=== Saving Embeddings ===")
#     embedding_df = pd.DataFrame(X_features)
#     embedding_df['id'] = combined_df['id']
#     embedding_df['emotion'] = y_labels
#     embedding_df['is_labeled'] = is_labeled
#     embedding_df.to_csv(f"Outputs/combined/embeddings_{cleaning_method}.csv", index=False)
    
#     # 10. Visualize embeddings
#     print("\n=== Visualizing Embeddings ===")
#     plot_embeddings(X_features, y_labels, is_labeled, method="tsne")
#     plot_embeddings(X_features, y_labels, is_labeled, method="pca")
    
#     print("\n=== Pipeline Completed Successfully ===")
#     return combined_df, unsupervised_model, X_features, y_labels, is_labeled

# # Example usage
# if __name__ == "__main__":
#     # Set parameters
#     labeled_json_path = "Data/labeled_train_data.json"
#     unlabeled_json_path = "Data/unlabeled_train_data.json"
#     hindi_stopwords_file = None  # Set to None if not available
#     limit_labeled = None  # Set to None to process all labeled data
#     limit_unlabeled = None  # Set to None to process all unlabeled data
#     use_advanced_cleaning = True  # Set to False to use simple cleaning
    
#     # Run the pipeline
#     combined_df, model, X_features, y_labels, is_labeled = main(
#         labeled_json_path=labeled_json_path,
#         unlabeled_json_path=unlabeled_json_path,
#         hindi_stopwords_file=hindi_stopwords_file,
#         limit_labeled=limit_labeled,
#         limit_unlabeled=limit_unlabeled,
#         use_advanced_cleaning=use_advanced_cleaning
#     )


import numpy as np
import pandas as pd
import re
import json
import fasttext
import fasttext.util
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')
import unicodedata
import nltk
from nltk.stem import PorterStemmer
from collections import defaultdict, Counter
from nltk.corpus import stopwords
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# Download NLTK resources if not already downloaded
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

# Access English stopwords
stop_words = set(stopwords.words('english'))

# Create output directories if they don't exist
os.makedirs("Outputs/combined", exist_ok=True)

# Set up emotion mapping like in original code
emotion_mapping = {
    'Anger': 0,
    'Surprise': 1,
    'Ridicule': 2,
    'Sad': 3
}

reverse_emotion_mapping = {v: k for k, v in emotion_mapping.items()}

# Text preprocessing functions
def load_hindi_stopwords(file_path):
    """Load Hindi stopwords from a text file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            stopwords = [line.strip() for line in file if line.strip()]
        return set(stopwords)
    except FileNotFoundError:
        print(f"Warning: Stopwords file not found at {file_path}. Using empty stopword list.")
        return set()

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

def clean_text_advanced(text, hindi_stopwords_file=None, remove_stopwords=True):
    """
    Advanced cleaning for code-mixed text (Hindi and English).
    Includes Unicode normalization, stopword removal, and stemming for English words.
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
    
    # Remove special characters, keeping Hindi characters (Devanagari Unicode range), English characters, and spaces
    text = re.sub(r'[^\w\s\u0900-\u097F]', ' ', text)
    
    # Remove stopwords
    if remove_stopwords:
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

def simple_clean_text(text):
    """Simple cleaning function."""
    if not isinstance(text, str):
        return ""
    # Remove special characters and convert to lowercase
    text = re.sub(r'[^\w\s]', '', text).lower()
    return text

# Load and process data functions
def load_data(json_file_path, data_type="data"):
    """Load and format the dataset from a JSON file."""
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"Loaded {data_type} dataset with {len(data)} entries")
        return data
    except Exception as e:
        print(f"Error loading {data_type} data: {e}")
        return None

def create_combined_dataframe(labeled_data, unlabeled_data, limit_labeled=None, limit_unlabeled=None):
    """Create a combined dataframe from labeled and unlabeled data."""
    all_examples = []
    
    # Process labeled data
    count = 0
    for id, instance in labeled_data.items():
        if limit_labeled is not None and count >= limit_labeled:
            break
        text = instance["target_utterance"]
        context = " ".join(instance["context_utterances"]) if "context_utterances" in instance else ""
        emotion = instance["emotion"]
        full_text = context + " " + text
        
        if emotion not in emotion_mapping:
            print(f"Skipping labeled instance {id} due to unknown emotion: {emotion}")
            continue
            
        all_examples.append({
            "id": id,
            "text": full_text,
            "emotion": emotion_mapping[emotion],
            "emotion_label": emotion,
            "target_utterance": text,
            "context_utterances": context,
            "full_text": full_text,
            "is_labeled": True
        })
        count += 1
    
    print(f"Processed {count} labeled examples")
    
    # Process unlabeled data
    count = 0
    for id, instance in unlabeled_data.items():
        if limit_unlabeled is not None and count >= limit_unlabeled:
            break
        text = instance["target_utterance"]
        context = " ".join(instance["context_utterances"]) if "context_utterances" in instance else ""
        full_text = context + " " + text
        
        all_examples.append({
            "id": id,
            "text": full_text,
            "emotion": -1,  # Placeholder for unlabeled data
            "emotion_label": "Unknown",
            "target_utterance": text,
            "context_utterances": context,
            "full_text": full_text,
            "is_labeled": False
        })
        count += 1
    
    print(f"Processed {count} unlabeled examples")
    
    # Create DataFrame
    combined_df = pd.DataFrame(all_examples)
    print(f"Created combined DataFrame with {len(combined_df)} entries")
    print(f"  - Labeled entries: {sum(combined_df['is_labeled'])}")
    print(f"  - Unlabeled entries: {sum(~combined_df['is_labeled'])}")
    
    # Analyze emotion distribution in labeled data
    if sum(combined_df['is_labeled']) > 0:
        labeled_emotions = combined_df[combined_df['is_labeled']]['emotion_label'].value_counts()
        print("\nEmotion distribution in labeled data:")
        print(labeled_emotions)
    
    return combined_df

# FastText supervised model functions
def create_fasttext_supervised_file(dataframe, filename, text_column="text_cleaned"):
    """Create a text file in the format required by FastText for supervised learning."""
    with open(filename, 'w', encoding='utf-8') as f:
        # Write labeled data with proper labels
        labeled_df = dataframe[dataframe['is_labeled']]
        for _, row in labeled_df.iterrows():
            f.write(f"__label__{row['emotion']} {row[text_column]}\n")
        
        # For unlabeled data, we'll use a placeholder label (will only be used for training)
        # This allows us to get embeddings later for all data
        unlabeled_df = dataframe[~dataframe['is_labeled']]
        if len(unlabeled_df) > 0:
            # We'll create a placeholder label that's not used elsewhere
            # The model will learn this as a separate class, but we won't use it for predictions
            for _, row in unlabeled_df.iterrows():
                f.write(f"__label__unknown {row[text_column]}\n")
    
    print(f"Created FastText supervised input file: {filename}")
    print(f"  - Added {len(labeled_df)} labeled examples")
    print(f"  - Added {len(unlabeled_df)} unlabeled examples with placeholder label")

def train_fasttext_supervised_model(input_file, output_model_name="supervised_model.bin", params=None):
    """Train a supervised FastText model with the given parameters."""
    if params is None:
        params = {
            'lr': 0.5,
            'epoch': 25,
            'wordNgrams': 2,
            'dim': 100,
            'loss': 'softmax'
        }
    
    print(f"Training FastText supervised model with parameters: {params}")
    model = fasttext.train_supervised(
        input=input_file,
        lr=params['lr'],
        epoch=params['epoch'],
        wordNgrams=params['wordNgrams'],
        dim=params['dim'],
        loss=params['loss']
    )
    
    # Save the model
    model.save_model(output_model_name)
    print(f"FastText supervised model saved as: {output_model_name}")
    return model

def extract_embeddings(model, dataframe, text_column="text_cleaned"):
    """Extract embeddings using the trained FastText model."""
    X_features = []
    
    print("Extracting embeddings...")
    for _, row in tqdm(dataframe.iterrows(), total=len(dataframe)):
        # Get text embedding (sentence vector)
        embedding = model.get_sentence_vector(row[text_column])
        X_features.append(embedding)
    
    X_features = np.array(X_features)
    print(f"Extracted embeddings shape: {X_features.shape}")
    return X_features

def plot_embeddings(X_features, dataframe, method="tsne", perplexity=30, n_components=2, random_state=42):
    """
    Reduce dimensionality and visualize clusters, with different plots for labeled vs unlabeled
    and emotion distribution.
    """
    print(f"Reducing dimensionality using {method.upper()}...")
    
    if method.lower() == "tsne":
        reducer = TSNE(n_components=n_components, perplexity=perplexity, random_state=random_state)
    elif method.lower() == "pca":
        reducer = PCA(n_components=n_components, random_state=random_state)
    else:
        raise ValueError("Invalid method! Choose 'tsne' or 'pca'.")
    
    X_reduced = reducer.fit_transform(X_features)

    # Convert to DataFrame for visualization
    df_plot = pd.DataFrame(X_reduced, columns=['x', 'y'])
    df_plot['emotion'] = dataframe['emotion'].values
    df_plot['emotion_label'] = dataframe['emotion_label'].values
    df_plot['is_labeled'] = dataframe['is_labeled'].values
    
    # PLOT 1: Labeled vs Unlabeled
    plt.figure(figsize=(10, 7))
    sns.scatterplot(
        x="x", y="y", hue="is_labeled", data=df_plot, palette=["gray", "blue"], 
        alpha=0.7, s=50, edgecolor="k"
    )
    plt.title(f"Text Embeddings: Labeled vs Unlabeled ({method.upper()})")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.legend(title="Data Type", labels=["Unlabeled", "Labeled"])
    plt.tight_layout()
    plt.savefig(f"Outputs/combined/embeddings_labeled_vs_unlabeled_{method}.png")
    
    # PLOT 2: Emotions (only for labeled data)
    plt.figure(figsize=(10, 7))
    labeled_df = df_plot[df_plot['is_labeled'] == True].copy()
    
    if len(labeled_df) > 0:
        sns.scatterplot(
            x="x", y="y", hue="emotion_label", data=labeled_df, palette="tab10", 
            alpha=0.7, s=50, edgecolor="k"
        )
        
        # Add emotion cluster labels
        for emotion in labeled_df['emotion_label'].unique():
            subset = labeled_df[labeled_df['emotion_label'] == emotion]
            centroid_x = subset["x"].mean()
            centroid_y = subset["y"].mean()
            plt.text(centroid_x, centroid_y, emotion, fontsize=12, fontweight='bold', 
                     ha='center', va='center')

        plt.title(f"Text Embeddings: Emotions ({method.upper()})")
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        plt.legend(title="Emotions", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        plt.savefig(f"Outputs/combined/embeddings_emotions_{method}.png")
    
    print(f"Embedding visualization plots saved to 'Outputs/combined/'")

# Main pipeline
def main(labeled_json_path, unlabeled_json_path, hindi_stopwords_file=None, 
         limit_labeled=None, limit_unlabeled=None, use_advanced_cleaning=True):
    """Main execution pipeline for processing both labeled and unlabeled data."""
    # 1. Load data
    print("\n=== Loading Data ===")
    labeled_data = load_data(labeled_json_path, "labeled")
    unlabeled_data = load_data(unlabeled_json_path, "unlabeled")
    
    if labeled_data is None or unlabeled_data is None:
        return
    
    # 2. Create combined DataFrame
    print("\n=== Creating Combined DataFrame ===")
    combined_df = create_combined_dataframe(
        labeled_data, unlabeled_data, 
        limit_labeled=limit_labeled, 
        limit_unlabeled=limit_unlabeled
    )
    
    # 3. Preprocess text
    print("\n=== Preprocessing Text ===")
    if use_advanced_cleaning:
        combined_df['text_cleaned'] = combined_df['text'].apply(
            lambda x: clean_text_advanced(x, hindi_stopwords_file=hindi_stopwords_file)
        )
        cleaning_method = "advanced"
    else:
        combined_df['text_cleaned'] = combined_df['text'].apply(simple_clean_text)
        cleaning_method = "simple"
    
    print(f"Applied {cleaning_method} cleaning to text")
    
    # 4. Create FastText supervised input file
    print("\n=== Creating FastText Supervised Input File ===")
    fasttext_file = f"Outputs/combined/fasttext_supervised_{cleaning_method}.txt"
    create_fasttext_supervised_file(combined_df, fasttext_file)
    
    # 5. Train FastText supervised model
    print("\n=== Training FastText Supervised Model ===")
    model = train_fasttext_supervised_model(
        fasttext_file, 
        output_model_name=f"Outputs/combined/supervised_model_{cleaning_method}.bin"
    )
    
    # 6. Extract embeddings for all data
    print("\n=== Extracting Embeddings ===")
    X_features = extract_embeddings(model, combined_df)
    
    # 7. Save processed data
    print("\n=== Saving Processed Data ===")
    combined_df.to_csv(f"Outputs/combined/processed_data_{cleaning_method}.csv", index=False)
    
    # 8. Save embeddings with metadata
    print("\n=== Saving Embeddings ===")
    embedding_df = pd.DataFrame(X_features)
    embedding_df['id'] = combined_df['id']
    embedding_df['emotion'] = combined_df['emotion']
    embedding_df['emotion_label'] = combined_df['emotion_label']
    embedding_df['is_labeled'] = combined_df['is_labeled']
    embedding_df.to_csv(f"Outputs/combined/embeddings_{cleaning_method}.csv", index=False)
    
    # 9. Visualize embeddings
    print("\n=== Visualizing Embeddings ===")
    plot_embeddings(X_features, combined_df, method="tsne")
    plot_embeddings(X_features, combined_df, method="pca")
    
    print("\n=== Pipeline Completed Successfully ===")
    return combined_df, model, X_features

# Example usage
if __name__ == "__main__":
    # Set parameters
    labeled_json_path = "Data/labeled_train_data.json"
    unlabeled_json_path = "Data/unlabeled_train_data.json"
    hindi_stopwords_file = None  # Set to None if not available
    limit_labeled = None  # Set to None to process all labeled data
    limit_unlabeled = None  # Set to None to process all unlabeled data
    use_advanced_cleaning = True  # Set to False to use simple cleaning
    
    # Run the pipeline
    combined_df, model, X_features = main(
        labeled_json_path=labeled_json_path,
        unlabeled_json_path=unlabeled_json_path,
        hindi_stopwords_file=hindi_stopwords_file,
        limit_labeled=limit_labeled,
        limit_unlabeled=limit_unlabeled,
        use_advanced_cleaning=use_advanced_cleaning
    )