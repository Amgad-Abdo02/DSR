from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer
import torch
import pandas as pd


Tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-cased')


class Tokenization(Dataset):
    def __init__(self, dataframe:pd.dataframe,Symptoms_col_name:str,tokenizer =Tokenizer, max_len:int =512)-> None:
        self.len = len(dataframe)
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.Symptoms = Symptoms_col_name

    def __getitem__(self, index):
        Review = eval(f"str(self.data.{self.Symptoms}[index])")
        Review = " ".join(Review.split())
        inputs = self.tokenizer.encode_plus(
            Review,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'targets': torch.tensor(self.data.label[index], dtype=torch.long)
        }

    def __len__(self):
        return self.len

def loader(training_set:Tokenization,Batch_size:int=8,shuffle:bool=True,num_workers:int = 0)-> DataLoader:
    params = {'batch_size': Batch_size,
                'shuffle': shuffle,
                'num_workers': num_workers}
    
    return DataLoader(training_set,**params)