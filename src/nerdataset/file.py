from torch.utils.data import Dataset
from transformers import BertTokenizer
import torch

class NERDataset(Dataset):
    def __init__(self, df, max_length = 128):
        self.df = df
        self.max_length = max_length
        self.tokenizer: BertTokenizer = BertTokenizer.from_pretrained('neuralmind/bert-base-portuguese-cased')
    
    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        text = row['text']
        label = row['label']

        encodings = self.tokenizer(
            text, 
            max_length=self.max_length, 
            padding='max_length', # Preenche automático com zeros até o limite
            truncation=True,      # Corta automático se a frase passar do limite
            return_tensors="pt"
        )

        return {
            'input_ids': encodings['input_ids'].squeeze(0),
            'attention_mask': encodings['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long)
        }