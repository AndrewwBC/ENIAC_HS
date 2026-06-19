import pandas as pd
from sklearn.model_selection import train_test_split
from base import BASE_DIR
import re
augmented_csv_path = BASE_DIR / "src/minabr/dataset/minabr.csv"

df = pd.read_csv(augmented_csv_path)

print(df['odio'].value_counts())
# 0    1751
# 1    1747

def clean_text(text: str):
    text = text.lower()
    text = re.sub(r'^rt\s+', '', text, flags=re.IGNORECASE)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'#(\w+)', r'\1', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

df['comment'] = df['comment'].apply(clean_text)

final_df = df[["comment", "odio"]]
final_df = final_df.rename(columns={"comment": "text", "odio": "label"})
train, test = train_test_split(final_df, test_size=0.2, shuffle=True, stratify=final_df['label'])