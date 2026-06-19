import re
import json
import pandas as pd
df = pd.read_csv("./minabr.csv")

def clean_text(text: str):
    text = text.lower()
    text = re.sub(r'^rt\s+', '', text, flags=re.IGNORECASE)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'#(\w+)', r'\1', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

df['comment'] = df['comment'].apply(clean_text)

print(df['odio'].value_counts())


only_hate_comments = df.query("odio == 1")[["comment"]]
list_of_comments = only_hate_comments['comment'].to_list()

with open("./list_of_comments.json", "w", encoding="utf-8") as f:
    json.dump(list_of_comments, f, ensure_ascii=False, indent=4)