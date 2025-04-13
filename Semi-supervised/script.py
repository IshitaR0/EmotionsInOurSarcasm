# import  pandas as pd
# import numpy as np
# import pandas as pd
# from sklearn.neighbors import kneighbors_graph
# import scipy.sparse as sp
# from sklearn.decomposition import PCA
# from sklearn.preprocessing import StandardScaler

# text_embeddings = pd.read_csv("/Users/anjelica/EmotionsInOurSarcasm/Text/scripts/final_embeddings.csv")
# video_embeddings = pd.read_csv("/Users/anjelica/EmotionsInOurSarcasm/Video/Final_video_embeddings.csv")
# labels = pd.read_csv("/Users/anjelica/EmotionsInOurSarcasm/Text/scripts/labelled/y_labels.csv")
# text_scaler = StandardScaler()
# video_scaler = StandardScaler()

# text_embeddings_scaled = text_scaler.fit_transform(text_embeddings)
# video_embeddings_scaled = video_scaler.fit_transform(video_embeddings)
# print(f"Text embeddings shape: {text_embeddings.shape}")
# print(f"Video embeddings shape: {video_embeddings.shape}")
# print(f"Labels shape: {labels.shape}")

# # Option 1: Reduce video dimensions with PCA to balance with text
# # pca = PCA(n_components=100)
# # video_embeddings_reduced = pca.fit_transform(video_embeddings_scaled)

# # Option 2: Concatenate with weighting to prevent the larger modality from dominating
# alpha = 0.5  # Balancing factor
# combined_embeddings = np.concatenate([
#     alpha * text_embeddings_scaled,
#     (1-alpha) * video_embeddings_scaled / 20  # Scaling down the larger dimension
# ], axis=1)


# # Create a k-nearest neighbors graph
# k = 10  # Number of neighbors
# affinity_matrix = kneighbors_graph(
#     combined_embeddings, k, 
#     mode='distance', 
#     include_self=False,
#     n_jobs=-1
# )

# # Convert distances to similarities using Gaussian kernel
# sigma = 1.0  # Bandwidth parameter
# affinity_matrix.data = np.exp(-affinity_matrix.data ** 2 / (2 * sigma ** 2))

# # Make the affinity matrix symmetric
# affinity_matrix = 0.5 * (affinity_matrix + affinity_matrix.T)


# from sklearn.semi_supervised import LabelPropagation, LabelSpreading

# # Prepare labels for semi-supervised learning
# # -1 indicates unlabeled data
# y_train = np.full(combined_embeddings.shape[0], -1)
# # Extract values from DataFrame and flatten to 1D array
# labels_array = labels.values.flatten()  
# # Ensure we only use the first 649 labels
# y_train[:649] = labels_array[:649]  # Your 649 labeled instances

# # Option 1: Use scikit-learn's LabelSpreading
# model = LabelSpreading(kernel='knn', alpha=0.2, max_iter=30)
# model.fit(combined_embeddings, y_train)
# predicted_labels = model.predict(combined_embeddings)

# # Option 2: Custom implementation for more control
# def custom_label_propagation(affinity, labels, alpha=0.2, max_iter=30):
#     """
#     Implement label propagation algorithm:
#     - affinity: sparse matrix of shape (n_samples, n_samples)
#     - labels: array-like of shape (n_samples,), with -1 for unlabeled
#     - alpha: clamping factor
#     - max_iter: maximum number of iterations
#     """
#     n_samples = affinity.shape[0]
#     n_classes = len(np.unique(labels[labels != -1]))
    
#     # Initialize label distributions (one-hot encoding)
#     Y = np.zeros((n_samples, n_classes))
#     for i in range(n_samples):
#         if labels[i] != -1:
#             Y[i, labels[i]] = 1
    
#     # Normalize the affinity matrix
#     D = np.array(affinity.sum(axis=1)).flatten()
#     D_inv = sp.diags(1.0 / np.sqrt(D))
#     affinity_norm = D_inv @ affinity @ D_inv
    
#     # Iterate until convergence
#     Y_prev = Y.copy()
#     for i in range(max_iter):
#         # Propagate labels
#         Y = affinity_norm @ Y
        
#         # Clamp labeled data
#         for j in range(n_samples):
#             if labels[j] != -1:
#                 Y[j] = np.zeros(n_classes)
#                 Y[j, labels[j]] = 1
        
#         # Check for convergence
#         if np.abs(Y - Y_prev).sum() < 1e-5:
#             break
#         Y_prev = Y.copy()
    
#     # Return predicted labels
#     return np.argmax(Y, axis=1)

# # Use custom implementation
# predicted_labels_option2 = custom_label_propagation(affinity_matrix, y_train)

# from sklearn.metrics import accuracy_score, classification_report

# # Evaluate on labeled data
# labeled_accuracy = accuracy_score(labels_array[:649], predicted_labels[:649])
# print(f"Accuracy on labeled data: {labeled_accuracy:.4f}")

# # Get confidence scores (probability distributions)
# label_distributions = model.label_distributions_

# # Only accept high-confidence predictions
# confidence_threshold = 0.5
# high_confidence_mask = np.max(label_distributions, axis=1) > confidence_threshold

# # Final predictions
# final_labels = np.full(combined_embeddings.shape[0], -1)
# final_labels[:649] = labels_array[:649]  # Keep original labels
# final_labels[649:] = np.where(
#     high_confidence_mask[649:],
#     predicted_labels[649:],
#     -1  # Keep as unlabeled if low confidence
# )

# # Count how many new labels we assigned
# newly_labeled = np.sum(final_labels[649:] != -1)
# print(f"Newly labeled instances: {newly_labeled} out of {len(final_labels) - 649}")


# # Optional: Implement an iterative process for active learning
# # This would involve:
# # 1. Identify most uncertain samples (lowest confidence)
# # 2. Present them to human annotator for labeling
# # 3. Add newly labeled samples to the training set
# # 4. Retrain the model
# # Repeat until budget is exhausted or desired performance is achieved


# ##IDK
# # def get_samples_for_annotation(label_distributions, n_samples=10):
# #     """Get the most uncertain samples for human annotation"""
# #     # Entropy as uncertainty measure
# #     entropies = -np.sum(label_distributions * np.log(label_distributions + 1e-10), axis=1)
    
# #     # Only consider currently unlabeled data (index 649 onwards)
# #     unlabeled_entropies = entropies[649:]
    
# #     # Get indices of most uncertain samples
# #     most_uncertain = np.argsort(unlabeled_entropies)[-n_samples:]
    
# #     # Return original indices (adding 649)
# #     return most_uncertain + 649

# # Train a supervised classifier on all labeled data (original + newly labeled)
# from sklearn.ensemble import RandomForestClassifier

# # Get mask of all labeled data
# labeled_mask = final_labels != -1

# # Train final model
# final_model = RandomForestClassifier(n_estimators=100)
# final_model.fit(
#     combined_embeddings[labeled_mask],
#     final_labels[labeled_mask]
# )

# # Now you can use this model for prediction on new data

# import matplotlib.pyplot as plt
# from sklearn.manifold import TSNE

# # Project to 2D for visualization
# tsne = TSNE(n_components=2, random_state=42)
# embeddings_2d = tsne.fit_transform(combined_embeddings)

# # Create a plot
# plt.figure(figsize=(10, 8))
# emotion_names = ['anger', 'surprise', 'ridicule', 'sad']
# colors = ['red', 'green', 'blue', 'purple']

# # Plot original labeled data
# for i in range(4):  # Four emotion classes
#     mask = labels_array[:649] == i
#     plt.scatter(
#         embeddings_2d[:649][mask][:, 0],
#         embeddings_2d[:649][mask][:, 1],
#         color=colors[i],
#         label=f'Original {emotion_names[i]}',
#         alpha=0.7
#     )

# # Plot newly labeled data
# for i in range(4):  # Four emotion classes
#     mask = final_labels[649:] == i
#     plt.scatter(
#         embeddings_2d[649:][mask][:, 0],
#         embeddings_2d[649:][mask][:, 1],
#         color=colors[i],
#         marker='x',
#         label=f'New {emotion_names[i]}',
#         alpha=0.5
#     )

# # Plot unlabeled data
# unlabeled_mask = final_labels[649:] == -1
# plt.scatter(
#     embeddings_2d[649:][unlabeled_mask][:, 0],
#     embeddings_2d[649:][unlabeled_mask][:, 1],
#     color='gray',
#     marker='.',
#     label='Unlabeled',
#     alpha=0.2
# )

# plt.legend()
# plt.title('Visualization of Original, Newly Labeled, and Unlabeled Data')
# plt.savefig('emotion_classification_visualization.png')
# plt.show()












# _________________________________

import pandas as pd
import numpy as np
from sklearn.neighbors import kneighbors_graph
import scipy.sparse as sp
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.semi_supervised import LabelSpreading
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Load data
text_embeddings = pd.read_csv("Outputs/combined/final_textEmbed.csv")
video_embeddings = pd.read_csv("/Users/anjelica/EmotionsInOurSarcasm/Video/Final_video_embeddings.csv")
labels = pd.read_csv("/Users/anjelica/EmotionsInOurSarcasm/Text/scripts/labelled/y_labels.csv")

# Print shapes to verify
print(f"Text embeddings shape: {text_embeddings.shape}")
print(f"Video embeddings shape: {video_embeddings.shape}")
print(f"Labels shape: {labels.shape}")

# Convert to numpy arrays if they aren't already
text_embeddings = np.array(text_embeddings)
video_embeddings = np.array(video_embeddings)
labels_array = np.array(labels).flatten().astype(int)

# Check label distribution
print(f"Label distribution: {np.bincount(labels_array[labels_array != -1])}")

# Standardize embeddings
text_scaler = StandardScaler()
video_scaler = StandardScaler()
text_embeddings_scaled = text_scaler.fit_transform(text_embeddings)
video_embeddings_scaled = video_scaler.fit_transform(video_embeddings)

# Use PCA for dimension reduction
pca = PCA(n_components=100, random_state=42)
video_embeddings_reduced = pca.fit_transform(video_embeddings_scaled)
print(f"Explained variance ratio sum: {np.sum(pca.explained_variance_ratio_):.4f}")

# Create balanced combined embeddings
combined_embeddings = np.concatenate([
    text_embeddings_scaled,
    video_embeddings_reduced
], axis=1)

# Build better graph - adjust k based on dataset size
k = min(20, int(combined_embeddings.shape[0] * 0.05))
print(f"Using k={k} neighbors")
affinity_matrix = kneighbors_graph(
    combined_embeddings, k,
    mode='distance',
    include_self=False,
    n_jobs=-1
)

# Calculate average distance for better sigma value
avg_distance = np.mean(affinity_matrix.data)
sigma = avg_distance
print(f"Average distance: {avg_distance:.4f}, using sigma={sigma:.4f}")

# Convert distances to similarities
affinity_matrix.data = np.exp(-affinity_matrix.data ** 2 / (2 * sigma ** 2))
affinity_matrix = 0.5 * (affinity_matrix + affinity_matrix.T)

# Prepare labeled and unlabeled data
y_train = np.full(combined_embeddings.shape[0], -1)
y_train[:649] = labels_array[:649]

# Try different model parameters
model = LabelSpreading(kernel='knn', alpha=0.3, max_iter=100, tol=1e-6)
model.fit(combined_embeddings, y_train)
predicted_labels = model.predict(combined_embeddings)

# Evaluate on labeled data
labeled_accuracy = accuracy_score(labels_array[:649], predicted_labels[:649])
print(f"Accuracy on labeled data: {labeled_accuracy:.4f}")
print(classification_report(labels_array[:649], predicted_labels[:649]))

# Try a lower confidence threshold
label_distributions = model.label_distributions_
confidence_threshold = 0.5
high_confidence_mask = np.max(label_distributions, axis=1) > confidence_threshold

# Final predictions
final_labels = np.full(combined_embeddings.shape[0], -1)
final_labels[:649] = labels_array[:649]
final_labels[649:] = np.where(high_confidence_mask[649:], predicted_labels[649:], -1)

# Count newly labeled instances
newly_labeled = np.sum(final_labels[649:] != -1)
print(f"Newly labeled instances: {newly_labeled} out of {len(final_labels) - 649}")

# Better visualization
plt.figure(figsize=(12, 10))
emotion_names = ['anger', 'surprise', 'ridicule', 'sad']
colors = ['red', 'green', 'blue', 'purple']

# Use better TSNE parameters
tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=2000)
embeddings_2d = tsne.fit_transform(combined_embeddings)

# Plot with more information
for i in range(4):
    mask = labels_array[:649] == i
    count = np.sum(mask)
    plt.scatter(
        embeddings_2d[:649][mask][:, 0],
        embeddings_2d[:649][mask][:, 1],
        color=colors[i],
        label=f'Original {emotion_names[i]} ({count})',
        alpha=0.7
    )

for i in range(4):
    mask = final_labels[649:] == i
    count = np.sum(mask)
    plt.scatter(
        embeddings_2d[649:][mask][:, 0],
        embeddings_2d[649:][mask][:, 1],
        color=colors[i],
        marker='x',
        label=f'New {emotion_names[i]} ({count})',
        alpha=0.5
    )

unlabeled_mask = final_labels[649:] == -1
unlabeled_count = np.sum(unlabeled_mask)
plt.scatter(
    embeddings_2d[649:][unlabeled_mask][:, 0],
    embeddings_2d[649:][unlabeled_mask][:, 1],
    color='gray',
    marker='.',
    label=f'Unlabeled ({unlabeled_count})',
    alpha=0.2
)

plt.grid(alpha=0.3)
plt.legend(fontsize=10)
plt.title('Emotion Classification Visualization', fontsize=14)
plt.xlabel('t-SNE Component 1', fontsize=12)
plt.ylabel('t-SNE Component 2', fontsize=12)
plt.savefig('improved_emotion_classification.png', dpi=300, bbox_inches='tight')
plt.show()