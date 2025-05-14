import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# ----- Load Training Data -----
print("Loading training data...")
df = pd.read_csv("Embeddings/Text/train_text.csv")
text_embeddings = df.values    

df1 = pd.read_csv("Embeddings/Videos/train/train_embeddings.csv")
video_embeddings = df1.values 

df2 = pd.read_csv("Embeddings/Labels/train_labels.csv")
labels = df2.values
labels = labels.ravel()

unique_classes = np.unique(labels)
class_names = [f"Class {i}" for i in unique_classes] 

X_train = np.concatenate((text_embeddings, video_embeddings), axis=1)  

print("Training data shapes:")
print("X_train shape:", X_train.shape)
print("labels shape:", labels.shape)

# ----- Load Validation Data -----
print("\nLoading validation data...")
val_text_df = pd.read_csv("Embeddings/Text/val_text.csv")
val_text_embeddings = val_text_df.values

val_video_df = pd.read_csv("Embeddings/Videos/val/val_embeddings.csv")
val_video_embeddings = val_video_df.values

val_labels_df = pd.read_csv("Embeddings/Labels/val_labels.csv")
val_labels = val_labels_df.values.ravel()

# Handle potential size mismatches in validation data
if val_text_embeddings.shape[0] != val_video_embeddings.shape[0]:
    min_samples = min(val_text_embeddings.shape[0], val_video_embeddings.shape[0])
    val_text_embeddings = val_text_embeddings[:min_samples]
    val_video_embeddings = val_video_embeddings[:min_samples]
    print(f"Using only the first {min_samples} samples from both validation feature datasets")

X_val = np.concatenate((val_text_embeddings, val_video_embeddings), axis=1)

# Handle potential mismatch between features and labels
if X_val.shape[0] != val_labels.shape[0]:
    min_samples = min(X_val.shape[0], val_labels.shape[0])
    X_val = X_val[:min_samples]
    val_labels = val_labels[:min_samples]
    print(f"Adjusted validation data to use only {min_samples} samples that have both features and labels")

print("Validation data shapes:")
print("X_val shape:", X_val.shape)
print("val_labels shape:", val_labels.shape)

# ----- Load Test Data -----
print("\nLoading test data...")
df3 = pd.read_csv("Embeddings/Text/test_text.csv") 
X_test_text = df3.values

df4_vid = pd.read_csv("Embeddings/Videos/test/test_embeddings.csv")
X_test_vids = df4_vid.values

df5 = pd.read_csv("Embeddings/Labels/test_labels.csv")
y_test = df5.values.ravel()

# Handle potential size mismatches in test data
if X_test_text.shape[0] != X_test_vids.shape[0]:
    min_samples = min(X_test_text.shape[0], X_test_vids.shape[0])
    X_test_text = X_test_text[:min_samples]
    X_test_vids = X_test_vids[:min_samples]
    print(f"Using only the first {min_samples} samples from both test feature datasets")

X_test = np.concatenate((X_test_text, X_test_vids), axis=1)

#----------------------------------------------------------------------------------------
# Use the file path utils/scripts/Video/cnn_embeddings.py for custom CNN video embeddings
#----------------------------------------------------------------------------------------

# Handle potential mismatch between features and labels
if X_test.shape[0] != y_test.shape[0]:
    min_samples = min(X_test.shape[0], y_test.shape[0])
    X_test = X_test[:min_samples]
    y_test = y_test[:min_samples]
    print(f"Adjusted test data to use only {min_samples} samples that have both features and labels")

print("Test data shapes:")
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)

# ----- Train the Model -----
print("\nTraining Random Forest model...")
clf = RandomForestClassifier(n_estimators=1000, random_state=42)
clf.fit(X_train, labels)

# ----- Evaluate on Validation Data -----
print("\nEvaluating on validation data...")
val_pred = clf.predict(X_val)
val_accuracy = accuracy_score(val_labels, val_pred)
print(f"Validation Accuracy: {val_accuracy:.4f}")
print("\nValidation Classification Report:")
print(classification_report(val_labels, val_pred))

# Create validation confusion matrix
val_cm = confusion_matrix(val_labels, val_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(val_cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Validation Confusion Matrix")
plt.savefig("Models/Model_0/Output/Validation_confusion_matrix.png")
plt.close()

# ----- Evaluate on Test Data -----
print("\nEvaluating on test data...")
test_pred = clf.predict(X_test)
test_accuracy = accuracy_score(y_test, test_pred)
print(f"Test Accuracy: {test_accuracy:.4f}")
print("\nTest Classification Report:")
print(classification_report(y_test, test_pred))

# Create test confusion matrix
test_cm = confusion_matrix(y_test, test_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(test_cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Test Confusion Matrix")
plt.savefig("Models/Model_0/Output/Test_confusion_matrix.png")
plt.close()

# ----- Feature Importance Analysis -----
print("\nAnalyzing feature importance...")
feature_importances = clf.feature_importances_
# Get indices of the top 20 most important features
top_indices = np.argsort(feature_importances)[-20:]
top_importances = feature_importances[top_indices]

# Plot feature importance
plt.figure(figsize=(10, 8))
plt.barh(range(len(top_indices)), top_importances, align='center')
plt.yticks(range(len(top_indices)), [f"Feature {i}" for i in top_indices])
plt.xlabel('Feature Importance')
plt.title('Top 20 Most Important Features')
plt.tight_layout()
plt.savefig("Models/Model_0/Output/Feature_importance.png")
plt.close()

# ----- Save Model Results -----
results = {
    'train_size': X_train.shape[0],
    'val_size': X_val.shape[0],
    'test_size': X_test.shape[0],
    'validation_accuracy': val_accuracy,
    'test_accuracy': test_accuracy
}

results_df = pd.DataFrame([results])
results_df.to_csv("Models/Model_0/Output/model_results.csv", index=False)

print("\nAnalysis complete. Results saved to 'Models/Model_0/Output/' directory.")

# _________________________________________________________________________________________
