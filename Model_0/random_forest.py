import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# Load training data
df = pd.read_csv("Embeddings/Text/train_emb.csv")
text_embeddings = df.values    

df1 = pd.read_csv("Embeddings/Videos/train_final_emb.csv")
video_embeddings = df1.values 

df2 = pd.read_csv("Embeddings/Labels/train_y_labels.csv")
labels = df2.values
labels = labels.ravel()

unique_classes = np.unique(labels)
class_names = [f"Class {i}" for i in unique_classes] 

X = np.concatenate((text_embeddings, video_embeddings), axis=1)  

print("X shape:", X.shape)
print("labels shape:", labels.shape)

# Train a Random Forest
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X, labels)

df3 = pd.read_csv("Embeddings/Text/test_emb.csv") 
X_test = df3.values
print("X_test shape:", X_test.shape)

df4_vid = pd.read_csv("Embeddings/Videos/test_final_emb.csv")  # Adjust path as needed
X_test_vids = df4_vid.values
print("X_test_vids shape:", X_test_vids.shape)

if X_test.shape[0] != X_test_vids.shape[0]:
    min_samples = min(X_test.shape[0], X_test_vids.shape[0])
    X_test = X_test[:min_samples]
    X_test_vids = X_test_vids[:min_samples]
    print(f"Using only the first {min_samples} samples from both datasets")

X_test_combined = np.concatenate((X_test, X_test_vids), axis=1)
print("X_test_combined shape:", X_test_combined.shape)

df5 = pd.read_csv("Embeddings/Labels/test_y_labels.csv")
y_test = df5.values.ravel()
print("y_test shape:", y_test.shape)

if y_test.shape[0] != X_test_combined.shape[0]:
    min_samples = min(y_test.shape[0], X_test_combined.shape[0])
    X_test_combined = X_test_combined[:min_samples]
    y_test = y_test[:min_samples]
    print(f"Adjusted to use only {min_samples} samples that have both features and labels")

y_pred = clf.predict(X_test_combined)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.savefig("Model_0/Output/Confusion_matrix.png")
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

