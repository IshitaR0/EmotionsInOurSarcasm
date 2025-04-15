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
def save_data_for_training(dataframe, output_file="training_data.txt", text_column="full_text"):
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

# 5. MAIN
def main(json_path, output_file_for_training="training_data.txt", use_existing_model=False, pre_trained_model_path=None):
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
    embeddings_df.to_csv("unlabelled_embeddings.csv", index=False)
    print("Embeddings saved to unlabelled_embeddings.csv")

    df.to_csv("unlabelled_texts.csv", index=False)
    print("Text data saved to unlabelled_texts.csv")

    return embeddings, df

# Run the script
if __name__ == "__main__":
    json_path = "/Users/anjelica/plaksha/mlpr/Project/train_data_final_plain.json" 
    embeddings, unlabelled_df = main(json_path)
