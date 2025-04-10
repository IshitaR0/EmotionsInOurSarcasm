# import json
# import pandas as pd
# import re
# import fasttext
# import numpy as np


# ## Phase 1

# with open('train_data_final_plain.json', 'r', encoding='utf-8') as f:
#     data = json.load(f)

# processed_examples = []

# count = 0

# for id, instance in data.items():

#     if count >= 60:
#         break

#     text = instance["target_utterance"]
    
#     context = " ".join(instance["context_utterances"])
    
#     emotion = instance["emotion"]
    
#     full_text = context + " " + text
    
#     text_cleaned = re.sub(r'[^\w\s]', '', full_text).lower()
    
#     processed_examples.append({
#         "text": text_cleaned,
#         "emotion": emotion
#     })
#     count+=1

# train_df = pd.DataFrame(processed_examples)

# # print(train_df.head())

# ## Phase 2

# def create_fasttext_file(dataframe, filename):
#     with open(filename, 'w', encoding='utf-8') as f:
#         for _, row in dataframe.iterrows():
#             f.write(f"__label__{row['emotion']} {row['text']}\n")

# create_fasttext_file(train_df, 'train.txt')

# ## Phase 3

# # Train the fastText model
# model = fasttext.train_supervised(
#     input='train.txt',
#     lr=0.5,
#     epoch=25,
#     wordNgrams=2,
#     dim=100,  # Embedding dimension
#     loss='softmax'
# )

# # Save the model
# model.save_model('emotion_model.bin')


# ## Phase 4

# def get_text_embedding(text):
#     # Get sentence vector (average of word vectors)
#     return model.get_sentence_vector(text)

# # Extract features for all examples
# X_features = []
# y_labels = []

# for _, row in train_df.iterrows():
#     # Get text embedding
#     embedding = get_text_embedding(row['text'])
#     X_features.append(embedding)
#     y_labels.append(row['emotion'])

# X_features = np.array(X_features)
# y_labels = np.array(y_labels)

# print("X-Features: ", X_features)
# print("y-labels: ", y_labels)
