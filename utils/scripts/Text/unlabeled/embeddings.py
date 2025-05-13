import json
import fasttext
import fasttext.util
import pandas as pd
import numpy as np
from tqdm import tqdm
import re
import unicodedata
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from collections import Counter


nltk.download('stopwords')

# Optional: set seed for reproducibility
np.random.seed(42)

# Load English stopwords
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize('NFC', text)
    english_parts = re.findall(r'[a-zA-Z]+', text)
    for part in english_parts:
        text = text.replace(part, part.lower())
    text = re.sub(r'[^\w\s\u0900-\u097F]', ' ', text)
    words = text.split()
    words = [stemmer.stem(word) if re.match(r'^[a-zA-Z]+$', word) else word for word in words if word not in stop_words]
    cleaned_text = ' '.join(words)
    return cleaned_text

# 1. Load your unlabelled data (from ID 651 onwards)
def load_unlabelled_data(json_path, start_index=651, end_index=893):
    print(f"Loading unlabelled data from {json_path}...")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} entries from the JSON file.")

    rows = []
    for id_str, entry in data.items():
        try:
            id_num = int(id_str)
        except ValueError:
            continue  # Skip if the key is not a number
        if id_num < start_index or id_num > end_index:
            continue
        text = entry.get("target_utterance", "")
        context = " ".join(entry.get("context_utterances", []))
        full_text = f"{context} {text}".strip()
        rows.append({"full_text": full_text})

    print(f"Filtered data: {len(rows)} entries from ID {start_index} to {end_index}.")
    return pd.DataFrame(rows)

# 2. Save the text data to a file for training FastText
def save_data_for_training(dataframe, output_file="Outputs/unlabeled/training_data.txt", text_column="full_text"):
    print(f"Saving cleaned text data for training FastText model...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for text in dataframe[text_column]:
            cleaned = clean_text(text)
            f.write(cleaned + '\n')
    print(f"Text data saved to {output_file}.")

# 3. Train FastText Model
def train_fasttext_model(input_file, output_model_name="fasttext_model.bin"):
    print(f"Training FastText model on {input_file}...")
    model = fasttext.train_unsupervised(input=input_file, model="skipgram", dim=100)
    model.save_model(output_model_name)
    print(f"FastText model saved as: {output_model_name}")
    return model

# 4. Extract sentence embeddings
def extract_sentence_embeddings(model, dataframe, text_column="full_text"):
    print("Extracting sentence embeddings...")
    embeddings = []
    for _, row in tqdm(dataframe.iterrows(), total=dataframe.shape[0]):
        cleaned_text = clean_text(row[text_column])
        embedding = model.get_sentence_vector(cleaned_text)
        embeddings.append(embedding)
    embeddings = np.array(embeddings)
    print(f"Extracted embeddings shape: {embeddings.shape}")
    return embeddings

# TO visualize the clusters

def plot_embeddings(X_features, y_labels, method="tsne", perplexity=30, n_components=2, random_state=42):
    """
    Reduce the dimensionality of FastText sentence embeddings and visualize the clustering by emotion.

    Parameters:
        X_features (numpy.ndarray): Sentence embeddings.
        y_labels (numpy.ndarray): Corresponding emotion labels.
        method (str): Dimensionality reduction method ('tsne' or 'pca').
        perplexity (int): Perplexity parameter for t-SNE (ignored if using PCA).
        n_components (int): Number of dimensions to reduce to (default: 2).
        random_state (int): Random seed for reproducibility.
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
    df_plot['emotion'] = y_labels

    plt.figure(figsize=(10, 7))
    sns.scatterplot(
        x="x", y="y", hue="emotion", data=df_plot, palette="tab10", alpha=0.7, s=50, edgecolor="k"
    )
    plt.savefig("Outputs/unlabeled/emotions_frequency.png")
    
    # Add emotion cluster labels if possible
    most_common_emotions = [e[0] for e in Counter(y_labels).most_common()]
    for emotion in most_common_emotions:
        subset = df_plot[df_plot['emotion'] == emotion]
        centroid_x = subset["x"].mean()
        centroid_y = subset["y"].mean()
        plt.text(centroid_x, centroid_y, emotion, fontsize=12, fontweight='bold', ha='center', va='center')

    plt.title(f"Sentence Embeddings Visualization using {method.upper()}")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.legend(title="Emotions", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig("Outputs/unlabeled/_graph_cluster_of_emotions.png")
    plt.show()


# 5. MAIN
def main(json_path, output_file_for_training="Outputs/unlabeled/training_data.txt", use_existing_model=False, pre_trained_model_path=None):
    print(f"Starting the main function with data from: {json_path}")
    
    # Load filtered unlabelled data
    df = load_unlabelled_data(json_path, start_index=651, end_index=893)
    print(f"Loaded data with {len(df)} entries.")

    if df.empty:
        print("No data found in the given index range. Exiting.")
        return None, None

    if not use_existing_model:
        save_data_for_training(df, output_file_for_training)
        model = train_fasttext_model(output_file_for_training)
    else:
        if pre_trained_model_path is None:
            print("Error: Provide path to the pre-trained model.")
            return None, None
        model = fasttext.load_model(pre_trained_model_path)
        print("Pre-trained model loaded.")

    embeddings = extract_sentence_embeddings(model, df)

    # Save embeddings and texts
    np.save("unlabelled_embeddings.npy", embeddings)
    print("Embeddings saved to unlabelled_embeddings.npy")

    embeddings_df = pd.DataFrame(embeddings, columns=[f"dim_{i}" for i in range(embeddings.shape[1])])

    # Save to CSV
    embeddings_df.to_csv("Outputs/unlabeled/unlabelled_embeddings.csv", index=False)
    print("Embeddings saved to Outputs/unlabeled/unlabelled_embeddings.csv")

    df.to_csv("Outputs/unlabeled/unlabelled_texts.csv", index=False)
    print("Text data saved to Outputs/unlabeled/unlabelled_texts.csv")
    # Plot and save embedding clusters
    y_labels = np.full(embeddings.shape[0], -1)
    plot_embeddings(embeddings, y_labels)

    return embeddings, df

# Run the script
if __name__ == "__main__":
    json_path = "Data/unlabeled_train_data.json" 
    embeddings, unlabelled_df = main(json_path)