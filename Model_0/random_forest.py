import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# Load training data
df = pd.read_csv("Embeddings/Text/train_emb.csv")
text_embeddings = df.values    

df1 = pd.read_csv("Embeddings/Vid_new/train_final_emb.csv")
video_embeddings = df1.values 

df4 = pd.read_csv("Embeddings/train_y_labels.csv")
labels = df4.values
labels = labels.ravel()

# Get the unique classes to use as class names
unique_classes = np.unique(labels)
class_names = [f"Class {i}" for i in unique_classes]  # Generate generic class names

X = np.concatenate((text_embeddings, video_embeddings), axis=1)  

print("X shape:", X.shape)
print("labels shape:", labels.shape)

# Train a Random Forest
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X, labels)

# Load test data
df2 = pd.read_csv("Embeddings/Text/test_final_emb.csv") 
X_test = df2.values
print("X_test shape:", X_test.shape)

# Load test video embeddings
df2_vid = pd.read_csv("Embeddings/Vid_new/test_final.csv")  # Adjust path as needed
X_test_vids = df2_vid.values
print("X_test_vids shape:", X_test_vids.shape)

# Check if the number of samples match
if X_test.shape[0] != X_test_vids.shape[0]:
    # Use only the matching samples
    min_samples = min(X_test.shape[0], X_test_vids.shape[0])
    X_test = X_test[:min_samples]
    X_test_vids = X_test_vids[:min_samples]
    print(f"Using only the first {min_samples} samples from both datasets")

# Now combine the features
X_test_combined = np.concatenate((X_test, X_test_vids), axis=1)
print("X_test_combined shape:", X_test_combined.shape)

# Load test labels
df3 = pd.read_csv("Embeddings/test_y_labels.csv")
y_test = df3.values.ravel()
print("y_test shape:", y_test.shape)

# Make sure test labels match the number of test samples
if y_test.shape[0] != X_test_combined.shape[0]:
    min_samples = min(y_test.shape[0], X_test_combined.shape[0])
    X_test_combined = X_test_combined[:min_samples]
    y_test = y_test[:min_samples]
    print(f"Adjusted to use only {min_samples} samples that have both features and labels")

# Predict and evaluate
y_pred = clf.predict(X_test_combined)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# If you know the actual class names, you can specify them
# Example if classes are 0, 1, 2 representing "Happy", "Sad", "Angry":
# class_labels = ["Happy", "Sad", "Angry"]
# print("\nClassification Report with Class Names:\n", 
#       classification_report(y_test, y_pred, target_names=class_labels))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()

# If you know the class names and want to add them to the confusion matrix:
# class_labels = ["Class 0", "Class 1", "Class 2"]  # Replace with actual class names
# plt.figure(figsize=(8,6))
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
#             xticklabels=class_labels, yticklabels=class_labels)
# plt.xlabel("Predicted")
# plt.ylabel("True")
# plt.title("Confusion Matrix")
# plt.show()


# ______________________________________________
# import numpy as np
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# from sklearn.ensemble import RandomForestClassifier
# import matplotlib.pyplot as plt
# import seaborn as sns

# df = pd.read_csv("Embeddings/Text/features_advanced.csv")
# text_embeddings = df.values    
# # print(text_embeddings.shape)

# df1 = pd.read_csv("Embeddings/Vid_new/final_vid_emb_comb.csv")
# video_embeddings = df1.values 

# df2 = pd.read_csv("Embeddings/y_labels.csv")
# labels = df2.values
# labels = labels.ravel()

# unique_classes = np.unique(labels)
# class_names = [f"Class {i}" for i in unique_classes]  # Generate generic class names

# X = np.concatenate((text_embeddings, video_embeddings), axis=1)  

# print("X shape:", X.shape)
# print("labels shape:", labels.shape)


# # Train a Random Forest
# clf = RandomForestClassifier(n_estimators=100, random_state=42)

# clf.fit(X, labels)


# df3 = pd.read_csv("Embeddings/Text/test_emb.csv") 
# X_test = df3.values

# df4 = pd.read_csv("Embeddings/Vid_new/test_final.csv")
# y_test_vids = df4.values


# df5 = pd.read_csv("Embeddings/test_y_labels.csv")
# y_test = df5.values

# X_test_concatd = np.concatenate((X_test, y_test_vids), axis=1)  

# # Predict and evaluate
# y_pred = clf.predict(X_test_concatd)

# print("Accuracy:", accuracy_score(y_test, y_pred))
# print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))

# # Confusion Matrix
# cm = confusion_matrix(y_test, y_pred)
# plt.figure(figsize=(8,6))
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
# plt.xlabel("Predicted")
# plt.ylabel("True")
# plt.title("Confusion Matrix")
# plt.show()

