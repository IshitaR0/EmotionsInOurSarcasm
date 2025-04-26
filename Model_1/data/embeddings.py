import pandas as pd

# Load the CSV file
file_path = "raw/final_averaged_features3.csv"
df = pd.read_csv(file_path)

# If 'Utterance' column exists
if 'Utterance' in df.columns:
    # Extract numeric part for sorting
    df['Utterance_Number'] = df['Utterance'].str.extract(r'(\d+)').astype(int)
    
    # Sort by the numeric part
    df = df.sort_values(by='Utterance_Number').drop(columns=['Utterance_Number'])
    df = df.drop(columns=['Utterance'])

new_file = "new_train.csv"  
# Save the updated CSV
df.to_csv(new_file, index=False)
