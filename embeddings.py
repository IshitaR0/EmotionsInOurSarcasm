import numpy as np
import pandas as pd
import fasttext
import fasttext.util
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tqdm import tqdm
import os
import pickle

# Import functions from data_cleaning.py
from data_cleaning import (
    load_data, 
    clean_text_without_expansion_or_normalization,
    load_hindi_stopwords
)

def load_fasttext_model(model_path=None, language='hi', vector_size=300):
    """
    Load the FastText model for Hindi (or download if not available).
    
    Args:
        model_path: Path to a pre-trained FastText model, if None, will download
        language: Language code ('hi' for Hindi)
        vector_size: Dimension of word vectors
        
    Returns:
        Loaded FastText model
    """
    try:
        if model_path and os.path.exists(model_path):
            print(f"Loading FastText model from {model_path}")
            model = fasttext.load_model(model_path)
        else:
            print(f"Downloading FastText model for language: {language}")
            # Download and load Hindi FastText model
            fasttext.util.download_model(language, if_exists='ignore')
            model_path = f'cc.{language}.300.bin'
            model = fasttext.load_model(model_path)
            
            # Reduce model size if needed
            if vector_size < 300:
                print(f"Reducing model dimension from 300 to {vector_size}")
                fasttext.util.reduce_model(model, vector_size)
                
        print("Model loaded successfully!")
        return model
    except Exception as e:
        print(f"Error loading FastText model: {e}")
        return None

def get_text_embedding(model, text, method='mean', vector_size=300):
    """
    Get embedding for a text by combining word embeddings.
    
    Args:
        model: FastText model
        text: Input text
        method: Method to combine word embeddings ('mean', 'max', 'sum')
        vector_size: Size of word vectors
        
    Returns:
        Numpy array of embeddings
    """
    if not text or not isinstance(text, str):
        return np.zeros(vector_size)
    
    words = text.split()
    if not words:
        return np.zeros(vector_size)
    
    word_vectors = [model.get_word_vector(word) for word in words]
    
    if method == 'mean':
        return np.mean(word_vectors, axis=0)
    elif method == 'max':
        return np.max(word_vectors, axis=0)
    elif method == 'sum':
        return np.sum(word_vectors, axis=0)
    else:
        # Default to mean
        return np.mean(word_vectors, axis=0)

def get_corpus_embeddings(model, texts, method='mean'):
    """
    Generate embeddings for a corpus of texts.
    
    Args:
        model: FastText model
        texts: List of texts
        method: Method to combine word embeddings
        
    Returns:
        Numpy array of document embeddings
    """
    embeddings = []
    for text in tqdm(texts, desc="Generating embeddings"):
        embeddings.append(get_text_embedding(model, text, method))
    return np.array(embeddings)

def visualize_embeddings(embeddings, labels, method='tsne', filename='embeddings_viz'):
    """
    Visualize embeddings using dimensionality reduction.
    
    Args:
        embeddings: Document embeddings
        labels: Document labels (e.g., emotions)
        method: 'tsne' or 'pca'
        filename: Base name for the output file
    """
    # Reduce dimensions to 2D
    if method == 'tsne':
        reducer = TSNE(n_components=2, perplexity=30, random_state=42)
    else:  # pca
        reducer = PCA(n_components=2)
    
    reduced = reducer.fit_transform(embeddings)
    
    # Create DataFrame for plotting
    df_plot = pd.DataFrame({
        'x': reduced[:, 0],
        'y': reduced[:, 1],
        'label': labels
    })
    
    # Plot
    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=df_plot, x='x', y='y', hue='label', palette='viridis')
    plt.title(f'Document Embeddings visualized using {method.upper()}')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f'{filename}_{method}.png')
    plt.close()
    print(f"Visualization saved as '{filename}_{method}.png'")

def save_embeddings(embeddings, file_path):
    """Save embeddings to a file."""
    with open(file_path, 'wb') as f:
        pickle.dump(embeddings, f)
    print(f"Embeddings saved to {file_path}")

def load_embeddings(file_path):
    """Load embeddings from a file."""
    with open(file_path, 'rb') as f:
        embeddings = pickle.load(f)
    print(f"Embeddings loaded from {file_path}")
    return embeddings

def main():
    # Load the dataset
    json_file_path = "dataset.json"
    df = load_data(json_file_path)
    
    if df is None:
        print("Failed to load dataset. Exiting.")
        return
    
    # Load the FastText model for Hindi
    model = load_fasttext_model(vector_size=300)
    if model is None:
        print("Failed to load FastText model. Exiting.")
        return
    
    # Clean the text data
    print("Cleaning text data...")
    hindi_stopwords_file = "stop_hinglish.txt"
    
    # Process target utterances
    df['cleaned_target'] = df['target_utterance'].apply(
        lambda x: clean_text_without_expansion_or_normalization(x, hindi_stopwords_file=hindi_stopwords_file)
    )
    
    # Process context utterances
    df['cleaned_context'] = df['context_utterances'].apply(
        lambda x: clean_text_without_expansion_or_normalization(x, hindi_stopwords_file=hindi_stopwords_file)
    )
    
    # Combine for full context
    df['full_text'] = df['cleaned_target'] + " " + df['cleaned_context']
    
    # Generate embeddings for target utterances
    print("Generating embeddings for target utterances...")
    target_embeddings = get_corpus_embeddings(model, df['cleaned_target'].tolist())
    
    # Generate embeddings for context utterances
    print("Generating embeddings for context utterances...")
    context_embeddings = get_corpus_embeddings(model, df['cleaned_context'].tolist())
    
    # Generate embeddings for combined text
    print("Generating embeddings for full text...")
    full_embeddings = get_corpus_embeddings(model, df['full_text'].tolist())
    
    # Save embeddings
    save_embeddings(target_embeddings, 'target_embeddings.pkl')
    save_embeddings(context_embeddings, 'context_embeddings.pkl')
    save_embeddings(full_embeddings, 'full_embeddings.pkl')
    
    # Visualize embeddings if emotion labels are available
    if 'emotion' in df.columns:
        print("Visualizing embeddings...")
        visualize_embeddings(full_embeddings, df['emotion'].tolist(), method='tsne')
        visualize_embeddings(full_embeddings, df['emotion'].tolist(), method='pca')
    
    print("Processing complete!")

if __name__ == "__main__":
    main()