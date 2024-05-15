import pandas as pd
import numpy as np
import json
import torch
from tqdm import tqdm
import transformers
import ast
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler


tqdm.pandas()

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#define tokenizer/model
tokenizer = transformers.RobertaTokenizer.from_pretrained("roberta-base")

model = transformers.AutoModel.from_pretrained("roberta-base")

model.to(DEV)

def embed_sequence(text):
    '''
    Function that generates a RoBERTa embedding from a string.
    '''
        
    encoding = tokenizer.encode_plus(text, return_tensors="pt", max_length=512)

    encoding = encoding.to(DEV)
    
    with torch.no_grad():
        o = model(**encoding)
    
    sentence_embedding = o.last_hidden_state[:,0]
    
    return sentence_embedding

train_list = []
val_list = []
test_list = []

with open('../data/counter_context/data/gold/train.jsonl', 'r') as f:
    for line in f:
        train_list.append(json.loads(line))
        
with open('../data/counter_context/data/gold/val.jsonl', 'r') as f:
    for line in f:
        val_list.append(json.loads(line))
        
with open('../data/counter_context/data/gold/test.jsonl', 'r') as f:
    for line in f:
        test_list.append(json.loads(line))
        

yu_train = pd.DataFrame(train_list)
yu_val = pd.DataFrame(val_list)
yu_test = pd.DataFrame(test_list)

input_df = pd.concat([yu_train, yu_val, yu_test])

input_df['input'] = input_df['context'] + ' </s> ' + input_df['target']

input_df['counterspeech'] = input_df['label'].apply(lambda x: x == '2').astype('int') #A label of '2' means counterspeech


text_x = torch.empty(size=(len(input_df), 768))

i = 0
for text in tqdm(input_df['input'].to_list()):
    text_x[i] = embed_sequence(text)

    i += 1

labels = torch.tensor(input_df['counterspeech'].to_list())

eye_matrix = torch.eye(2)

y = eye_matrix[labels]

training_set = TensorDataset(text_x, y)


torch.save(training_set, '../data/yu_et_al_input.pth')

