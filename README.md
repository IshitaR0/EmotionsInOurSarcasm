# Emotion Recognition in Code-Mixed (Hinglish) Sarcastic conversations


## Overview

This project tackles the task of recognizing underlying emotions in sarcastic, code-mixed (Hinglish) conversations using multimodal data. 

## Key Features

- Using both text and video modality.
- Utilized the WITS dataset ("Why is this Sarcastic") containing multi-modal code-mixed(Hinglish) data.
- We categorized sarcastic utterances into four emotion classes:
  - Anger (0)
  - Surprise (1)
  - Ridicule (2)
  - Sad (3)

## Dataset
<div align="center">
  <img src="https://github.com/user-attachments/assets/7bfa5f28-be2b-411a-bf18-dd783bacbb74" alt="Image 1" width="22%" />
  <img src="https://github.com/user-attachments/assets/e28743d7-f984-41ed-8460-1fa62b5d75a5" alt="Image 2"  width="40%" />
  <br />
  <img src="https://github.com/user-attachments/assets/588a969d-7391-40e1-8928-cfa94453230b" alt="Image 3" width="31%" />
  <img src="https://github.com/user-attachments/assets/83632785-54aa-4413-be8f-6ca49f82d3da" alt="Image 4" width="31%" />
</div>
 

## Directory Structure

```
EmotionsInOurSarcasm/
├── Distributions/
│   ├── emotion_analysis/
|             ├── script.py
|             ├── train/
|             ├── test/
│             └── val/
├── Models/
│   ├── Model_0/
|              ├── Output/
│              └── model.py
│   ├── Model_1/
|              ├── Outputs/
│              └── model.py
├── utils/scripts
│   ├── Text/
│           └── embeddings.py
|   ├── Video/
|           ├── cnn_embeddings.py
│           └── embeddings.py
├── .gitignore
├── README.md
└── requirements.txt
```


### Dataset
- `Distributions/emotion_analysis`: Contains distribution plots for the 4 emotion classes across train/val/test. 

### Feature Extraction
- `utils/scripts/Text`: This folder contains the script to generate FastText embeddings for text modality.
- `utils/scripts/Video/embeddings.py`: This script generates vector embeddings for video modality using ResNet50 after extracting keyframes using OpenCV and averaging them.
- `utils/scripts/Video/embeddings.py`: This acript generates vector embeddings for video modality using our custom CNN.

### Models
- `models/Model_0`: Implementation of Random Forest Classifier model and it's output. 
- `models/Model_1`: Implementation of Shallow Neural Network with Attention Fusion Mechanism (SNN-AFM) and it's output.


## Performance Metrics

| Model | Accuracy | Weighted F1-Score |
|-------|----------|------------------|
| SNN-AFM | 51.12% | 0.51 |
| Random Forest | 52.91% | 0.53 |

## Output Plots
![image](https://github.com/user-attachments/assets/fa7b3347-ceef-44a7-83c7-750dc4c09dbf)




## Installation and Usage

```bash
# Clone the repository
git clone https://github.com/IshitaR0/EmotionsInOurSarcasm.git
cd EmotionsInOurSarcasm

# Install dependencies
pip install -r requirements.txt
```

## Future Improvements

- Include audio modality for enhanced performance
- Address underrepresented emotion classes


## Contributors

- Anjelica
- Ishita Rathore
- Suhaani Sachdeva


