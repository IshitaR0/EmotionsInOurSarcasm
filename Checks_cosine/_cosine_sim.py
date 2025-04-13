import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def assign_pseudo_labels(X_labeled, X_unlabeled, labeled_df, threshold=0.85):
    """
    Assign pseudo-labels to unlabeled data based on cosine similarity with labeled data.
    
    Parameters:
        X_labeled (np.ndarray): Embeddings of labeled samples (N_labeled x D)
        X_unlabeled (np.ndarray): Embeddings of unlabeled samples (N_unlabeled x D)
        labeled_df (pd.DataFrame): DataFrame with columns ['text', 'emotion', 'emotion_label', ...]
        threshold (float): Minimum cosine similarity to assign a pseudo-label
        
    Returns:
        pd.DataFrame: Unlabeled data with assigned pseudo-labels and similarity scores
    """
    labels = labeled_df["emotion"].tolist()

    # Compute cosine similarity
    similarity_matrix = cosine_similarity(X_unlabeled, X_labeled)
    most_similar_indices = similarity_matrix.argmax(axis=1)
    top_scores = similarity_matrix[np.arange(len(X_unlabeled)), most_similar_indices]

    pseudo_labels = []
    for idx, score in zip(most_similar_indices, top_scores):
        label = labels[idx] if score >= threshold else "uncertain"
        pseudo_labels.append(label)

    # Build the output DataFrame
    result_df = pd.DataFrame({
        "pseudo_label": pseudo_labels,
        "similarity_score": top_scores
    })

    return result_df

# Example usage (you’ll replace this with your actual data):
if __name__ == "__main__":
    import pandas as pd
    import numpy as np

    # Load CSV embeddings and convert to NumPy arrays
    X_labeled = pd.read_csv("Embeddings/Text/labeled_embed.csv").values
    X_unlabeled = pd.read_csv("Embeddings/Text/unlabeled_embed.csv").values

    # Load your labeled DataFrame
    labeled_df = pd.read_json("Data/labeled_train_data.json")
  # Should include 'emotion_label'

    # Assign pseudo-labels
    result = assign_pseudo_labels(X_labeled, X_unlabeled, labeled_df, threshold=0.85)

    # Save to CSV
    result.to_csv("Checks_cosine/outputs/pseudo_labeled_output.csv", index=False)
    print("✅ Pseudo-labels saved to pseudo_labeled_output.csv")
