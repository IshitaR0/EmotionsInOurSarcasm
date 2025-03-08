from transformers import RobertaTokenizer
import torch
from datasets import load_dataset
import json
import pandas as pd

with open("/Users/anjelica/plaksha/mlpr/Project/train_data_final_plain.json", "r") as f:
    data = json.load(f)

rows = []
for key, value in data.items():
    rows.append({
        "episode_name": value["episode_name"],
        "target_speaker": value["target_speaker"],
        "target_utterance": value["target_utterance"],
        "context_speakers": value["context_speakers"],
        "context_utterances": " ".join(value["context_utterances"]),  # Merge context utterances
        "english_explanation": value.get("english_explanation", ""),
        "code_mixed_explanation": value.get("code_mixed_explanation", ""),
        "sarcasm_target": value.get("sarcasm_target", ""),
        "sarcasm_type": value.get("sarcasm_type", ""),
        "start_time": value["start_time"],
        "end_time": value["end_time"],
        # "emotion": value["emotion"]
    })
df = pd.DataFrame(rows)
df["full_text"] = df["target_utterance"] + " " + df["context_utterances"]
df.head()