import pandas as pd
import numpy as np
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

def create_row_vector(row):
    return [row[c] for c in row.index]

text_data = pd.read_csv('../data/counterspeech_dataset.csv', lineterminator='\n')

text_data['newcomer_input'] = text_data['context'] + ' </s> ' + text_data['newcomer']
text_data['reply_input'] = text_data['newcomer'] + ' </s> ' + text_data['reply']

newcomers = text_data[['subreddit', 'newcomer_input', 'newcomer_counterspeech',
                       'newcomer_score']].rename(columns={'newcomer_input':'input',
                       'newcomer_counterspeech':'counterspeech', 'newcomer_score':'score'})

replies = text_data[['subreddit', 'reply_input', 'reply_counterspeech',
                     'reply_score']].rename(columns={'reply_input':'input',
                     'reply_counterspeech':'counterspeech', 'reply_score':'score'})

#make each row a model input
input_df = pd.concat([newcomers, replies])    

text_x = torch.empty(size=(len(input_df), 768))

i = 0
for text in tqdm(input_df['input'].to_list()):
    text_x[i] = embed_sequence(text)

    i += 1

labels = torch.tensor(input_df['counterspeech'].to_list())
scores = torch.tensor(input_df['score'].fillna(0).to_list()) #missing scores are 0

categories = input_df['subreddit'].unique() #get all subreddits
category_to_label = {category: label for label, category in enumerate(categories)}
num_categories = len(categories)

input_df['subreddit_label'] = input_df['subreddit'].map(category_to_label) #assign each subreddit a number
one_hot_encoded = np.eye(num_categories)[input_df['subreddit_label']] #turn subreddit number into one-hot encoding
one_hot_tensor = torch.tensor(one_hot_encoded, dtype=torch.float32) #tensor representing subreddits

eye_matrix = torch.eye(2)

y = eye_matrix[labels]

training_set = TensorDataset(text_x, scores, one_hot_tensor, y)


torch.save(training_set, '../data/counterspeech_model_input.pth')
