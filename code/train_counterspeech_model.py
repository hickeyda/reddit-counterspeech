import numpy as np

#Code adapted from Matheus Schmitz AVAXTAR Model: https://github.com/Matheus-Schmitz/avaxtar
# Neural Network
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from torch.utils.data import TensorDataset
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm
import pandas as pd

import logging

ADDITIONAL_FEATURES=True

dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class CounterspeechNN(nn.Module):

    def __init__(self, num_features=768, learning_rate=1e-4, optimizer=torch.optim.AdamW, loss_fn=nn.BCELoss(), device='cpu'):
        super(CounterspeechNN, self).__init__()
        self.fc1 = nn.Linear(num_features, num_features//4)
        self.fc2 = nn.Linear(num_features, num_features//2)
        self.fc3 = nn.Linear(num_features//4 + 26, 2)
        self.fc4 = nn.Linear(num_features//4, 2)
        self.dropout1 = nn.Dropout(0.60)
        self.dropout2 = nn.Dropout(0.60)

        self.device = device
        self.optimizer = optimizer(self.parameters(), lr=learning_rate)
        self.loss_fn = loss_fn.to(self.device)

	# x represents our data
    def forward(self, x, additional_features=False, x2=None, x3=None):
        x = self.fc1(x)
        x = torch.tanh(x) 
        x = self.dropout1(x)

        
        if additional_features:
            x = torch.cat((x, x2.unsqueeze(1).float(), x3), dim=1)
            x = self.fc3(x)

        else:
            x = self.fc4(x)

        output = F.softmax(x, dim=1)
        return output

    def fit(self, train_loader, valid_loader, writing_file, seed, additional_features=False, epochs=1000, additional_features_valid_loading=True):
        for epoch in range(epochs):
            self.train() 

            LOSS_train = 0.0
            PRECISION_0_train, RECALL_0_train, F1_0_train, ACCURACY_0_train = 0.0, 0.0, 0.0, 0.0
            PRECISION_1_train, RECALL_1_train, F1_1_train, ACCURACY_1_train = 0.0, 0.0, 0.0, 0.0
            full_preds = []
            full_labels = []

            for j, data in enumerate(train_loader, 0):

                # Get the inputs; data is a list of [inputs, labels]

                if additional_features:
                    inputs, input2, input3, labels = data
                    inputs = inputs.to(self.device)
                    input2 = input2.to(self.device)
                    input3 = input3.to(self.device)

                else:
                    inputs, labels = data
                    inputs = inputs.to(self.device)


                labels = labels.to(self.device)
                labels = labels.float()
                full_labels += [int(l[1]) for l in labels]

                # Zero the parameter gradients
                self.optimizer.zero_grad()

                # Forward (aka predict)
                if additional_features:
                    outputs = self.forward(inputs, additional_features, input2, input3)

                else:
                    outputs = self.forward(inputs, additional_features)

                predictions = np.array([[1.0, 0.0] if np.argmax(p.cpu().detach().numpy())==0 else [0.0, 1.0] for p in outputs])
                full_preds += [float(p[1]) for p in outputs]

                # Loss + Backprop
                loss = self.loss_fn(outputs, labels)
                LOSS_train += loss.item()
                loss.backward()
                self.optimizer.step()

                # Metrics
                (precision, recall, f1, accuracy) = self.get_metrics(predictions, labels, for_class=0)
                PRECISION_0_train += precision
                RECALL_0_train += recall
                F1_0_train += f1
                ACCURACY_0_train += accuracy

                (precision, recall, f1, accuracy) = self.get_metrics(predictions, labels, for_class=1)
                PRECISION_1_train += precision
                RECALL_1_train += recall
                F1_1_train += f1
                ACCURACY_1_train += accuracy    

			# Average the metrics based on the number of batches
            LOSS_train /= len(train_loader)
            PRECISION_0_train /= len(train_loader)
            RECALL_0_train /= len(train_loader)
            F1_0_train /= len(train_loader)
            ACCURACY_0_train /= len(train_loader)
            PRECISION_1_train /= len(train_loader)
            RECALL_1_train /= len(train_loader)
            F1_1_train /= len(train_loader)
            ACCURACY_1_train /= len(train_loader)
            roc_train = roc_auc_score(full_labels, full_preds)

            # Validation set
            self.eval()
            with torch.no_grad():

                LOSS_valid = 0.0
                PRECISION_0_valid, RECALL_0_valid, F1_0_valid, ACCURACY_0_valid = 0.0, 0.0, 0.0, 0.0
                PRECISION_1_valid, RECALL_1_valid, F1_1_valid, ACCURACY_1_valid = 0.0, 0.0, 0.0, 0.0
                val_preds = []
                val_labels = []

                for k, data in enumerate(valid_loader, 0):

                        # Get the inputs; data is a list of [inputs, labels]
                    
                    if additional_features_valid_loading:
                        inputs, input2, input3, labels = data
                        inputs = inputs.to(self.device)
                        input2 = input2.to(self.device)
                        input3 = input3.to(self.device)

                    else:
                        inputs, labels = data

                    
                    labels = labels.to(self.device)
                    val_labels += [int(l[1]) for l in labels]

                    # Forward (aka predict)
                    if additional_features:
                        outputs = self.forward(inputs, additional_features, input2, input3)

                    else:
                        outputs = self.forward(inputs, additional_features)

                    predictions = np.array([[1.0, 0.0] if np.argmax(p.cpu().detach().numpy())==0 else [0.0, 1.0] for p in outputs])
                    val_preds += [float(p[1]) for p in outputs]

                    # Loss
                    loss = self.loss_fn(outputs, labels)
                    LOSS_valid += loss.item()

                    # Metrics
                    (precision, recall, f1, accuracy) = self.get_metrics(predictions, labels, for_class=0)
                    PRECISION_0_valid += precision
                    RECALL_0_valid += recall
                    F1_0_valid += f1
                    ACCURACY_0_valid += accuracy

                    (precision, recall, f1, accuracy) = self.get_metrics(predictions, labels, for_class=1)
                    PRECISION_1_valid += precision
                    RECALL_1_valid += recall
                    F1_1_valid += f1
                    ACCURACY_1_valid += accuracy

                # Average the metrics based on the number of batches
                LOSS_valid /= len(valid_loader)
                PRECISION_0_valid /= len(valid_loader)
                RECALL_0_valid /= len(valid_loader)
                F1_0_valid /= len(valid_loader)
                ACCURACY_0_valid /= len(valid_loader)
                PRECISION_1_valid /= len(valid_loader)
                RECALL_1_valid /= len(valid_loader)
                F1_1_valid /= len(valid_loader)
                ACCURACY_1_valid /= len(valid_loader)
                roc_valid = roc_auc_score(val_labels, val_preds)


			##################
			### STATISTICS ###
			##################
            '''
            if (epoch+1)%10 == 0:
                print(f'Epoch {epoch+1}/{epochs}'+'\n')
                print(f'Train Loss: {LOSS_train:.5f}  |  Valid Loss: {LOSS_valid:.5f}')
                print(f'Train Accu: {ACCURACY_0_train:.5f}  |  Valid Accu: {ACCURACY_0_valid:.5f}')
                print()
                print("Class 0:")
                print(f'Train Prec: {PRECISION_0_train:.5f}  |  Valid Prec: {PRECISION_0_valid:.5f}')
                print(f'Train Rcll: {RECALL_0_train:.5f}  |  Valid Rcll: {RECALL_0_valid:.5f}')
                print(f'Train  F1 : {F1_0_train:.5f}  |  Valid  F1 : {F1_0_valid:.5f}')
                print()
                print("Class 1:")
                print(f'Train Prec: {PRECISION_1_train:.5f}  |  Valid Prec: {PRECISION_1_valid:.5f}')
                print(f'Train Rcll: {RECALL_1_train:.5f}  |  Valid Rcll: {RECALL_1_valid:.5f}')
                print(f'Train  F1 : {F1_1_train:.5f}  |  Valid  F1 : {F1_1_valid:.5f}')
                print()
                print(f"Train ROC-AUC: {roc_train:.5f} | Valid ROC-AUC: {roc_valid:.5f}")
                print('-'*43)
                print()
            '''
		
        #print('Finished Training')
        if writing_file:
            with open(writing_file + '.csv', 'a+') as f:
                f.write(str(seed) + ',' + str(roc_valid) + '\n')

            
            for threshold in np.arange(0, 1, 0.05):
                first_one = ','
                if threshold == 0:
                    first_one = f'{seed},'
                TP, FP, TN, FN = self.get_confusion_matrix(val_labels, val_preds, threshold)

                with open(writing_file + '_true_positives.csv', 'a+') as f:
                    f.write(first_one + str(TP))
                
                with open(writing_file + '_false_positives.csv', 'a+') as f:
                    f.write(first_one + str(FP))

                with open(writing_file + '_true_negatives.csv', 'a+') as f:
                    f.write(first_one + str(TN))
                
                with open(writing_file + '_false_negatives.csv', 'a+') as f:
                    f.write(first_one + str(FN))

            for evaluation in ['_true_positives.csv', '_false_positives.csv', '_true_negatives.csv', '_false_negatives.csv']:
                with open(writing_file + evaluation, 'a+') as f:
                    f.write('\n')
            
    def get_metrics(self, preds, labels, for_class=1):
        TP, FP, TN, FN = 0, 0, 0, 0

        # Iterate over all predictions
        for idx in range(len(preds)):
			# If we predicted the sample to be class {for_class}
            if preds[idx][for_class] == 1:
				# Then check whether the prediction was right or wrong
                if labels[idx][for_class] == 1:
                    TP += 1
                else:
                    FP += 1
			# Else we predicted another class 
            else:
				# Check whether the "not class {for_class}" prediction was right or wrong
                if labels[idx][for_class] != 1:
                    TN += 1
                else:
                    FN += 1

        precision = TP/(TP+FP) if TP+FP > 0 else 0 # Of all "class X" calls I made, how many were right?
        recall = TP/(TP+FN) if TP+FN > 0 else 0 # Of all "class X" calls I should have made, how many did I actually make?
        f1 = (2*precision*recall)/(precision+recall) if precision+recall > 0 else 0
        accuracy = (TP+TN)/(TP+FP+TN+FN)

        return (precision, recall, f1, accuracy)

    def get_confusion_matrix(self, labels, preds, threshold, for_class=1):
        TP, FP, TN, FN = 0, 0, 0, 0

        # Iterate over all predictions
        for idx in range(len(preds)):
			# If we predicted the sample to be class {for_class}
            if preds[idx] > threshold:
				# Then check whether the prediction was right or wrong
                if labels[idx] == 1:
                    TP += 1
                else:
                    FP += 1
			# Else we predicted another class 
            else:
				# Check whether the "not class {for_class}" prediction was right or wrong
                if labels[idx] != 1:
                    TN += 1
                else:
                    FN += 1

        return (TP, FP, TN, FN)


    def predict(self, X_input):
        self.eval()
        result = []

        # Predict from PyTorch dataloader
        if type(X_input) == torch.utils.data.dataloader.DataLoader:
            
            with torch.no_grad():
                for k, data in enumerate(X_input, 0):
					# Get the inputs; data is a list of [inputs, labels]
                    inputs, labels = data
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    # Forward (aka predict)
                    outputs = self.forward(inputs)
                    predictions = np.array([[1.0, 0.0] if np.argmax(p.cpu().detach().numpy())==0 else [0.0, 1.0] for p in outputs])

                    if len(result)==0:
                        result = predictions
                    else:
                        result = np.concatenate((result, predictions))

                return np.array([np.argmax(x) for x in result])
        
        # Predict from Numpy array or list
        else:
            
            if type(X_input) == list:
                X_input = np.array(X_input)
        
            if isinstance(X_input, np.ndarray): 
                X_test = torch.from_numpy(X_input).float()

                with torch.no_grad():
                    for k, data in enumerate(X_test):
                        # Get the inputs; data is a list of [inputs] 
                        inputs = data.reshape(1,-1).to(self.device)
                        # Forward (aka predict)
                        outputs = self.forward(inputs)
                        predictions = np.array([[1.0, 0.0] if np.argmax(p.cpu().detach().numpy())==0 else [0.0, 1.0] for p in outputs])

                        if len(result)==0:
                            result = predictions
                        else:
                            result = np.concatenate((result, predictions))

                    return np.array([np.argmax(x) for x in result])
                            
            else:
                raise("Input must be a dataloader, numpy array or list")


    def predict_proba(self, additional_features, X_input, additional_features_loading=True):
        self.eval()
        result = []
        
        # Predict from PyTorch dataloader
        if type(X_input) == torch.utils.data.dataloader.DataLoader:

            with torch.no_grad():
                for k, data in enumerate(X_input, 0):
                    # Get the inputs; data is a list of [inputs, labels]
                    if additional_features_loading:
                        #inputs, input2, input3, labels = data
                        inputs, input2, input3 = data
                        #inputs, labels = data
                        inputs = inputs.to(self.device)
                        input2 = input2.to(self.device)
                        input3 = input3.to(self.device)

                    else:
                        inputs, labels = data
                        inputs = inputs.to(self.device)

                    #labels = labels.to(self.device)
                    # Forward (aka predict)
                    if additional_features:
                        outputs = self.forward(inputs, additional_features, input2, input3)

                    else:
                        outputs = self.forward(inputs, additional_features)

                    class_proba = outputs.cpu().detach().numpy()
                    
                    if len(result)==0:
                        result = class_proba
                    else:
                        result = np.concatenate((result, class_proba))

                return result
        
        # Predict from Numpy array or list
        else:
            
            if type(X_input) == list:
                X_input = np.array(X_input)
        
            if isinstance(X_input, np.ndarray):  
                X_input = torch.from_numpy(X_input).float()

                with torch.no_grad():
                    for k, data in enumerate(X_input):
                        # Get the inputs; data is a list of [inputs] 
                        inputs, input2, input3 = data.reshape(1,-1).to(self.device)
                        # Forward (aka predict)
                        outputs = self.forward(inputs, input2, input3)
                        class_proba = outputs.cpu().detach().numpy()

                        if len(result)==0:
                            result = class_proba
                        else:
                            result = np.concatenate((result, class_proba))

                    return result

            else:
                raise("Input must be a dataloader, numpy array or list")
            

if __name__ == '__main__':
    
    bootstrap_path = f'../data/counterspeech_results_additional_features_{ADDITIONAL_FEATURES}'

    with open(bootstrap_path + '.csv', 'w') as f:
        f.write('seed,roc_auc\n')

    #create files to write confusion matrices to for each threshold    
    for evaluation in ['_true_positives.csv', '_false_positives.csv', '_true_negatives.csv', '_false_negatives.csv']:
        with open(bootstrap_path + evaluation, 'w') as f:
            f.write('seed')
            for threshold in np.arange(0, 1, 0.05):
                f.write(f',threshold_{threshold * 100}')

            f.write('\n')
    
    full_dataset = torch.load('./counterspeech_model_input.pth')
    
    for i in tqdm(range(50)): #for each random seed
        model = CounterspeechNN(device=dev) #initialize model 

        #split dataset
        train_x, val_x, train_x2, val_x2, train_x3, val_x3, train_y, val_y = train_test_split(full_dataset.tensors[0], full_dataset.tensors[1], full_dataset.tensors[2], full_dataset.tensors[3], test_size=0.15, random_state=i)
        train_x, test_x, train_x2, test_x2, train_x3, test_x3, train_y, test_y = train_test_split(train_x, train_x2, train_x3, train_y, test_size=0.176, random_state=i)
        
        train_dataset = TensorDataset(train_x, train_x2, train_x3, train_y)

        valid_dataset = TensorDataset(test_x, test_x2, test_x3, test_y) #use test_x/val_x depending on testing/validation

        train_loader = DataLoader(
                    train_dataset,  # The training samples.
                    sampler = RandomSampler(train_dataset), # Select batches randomly
                    batch_size = 32 # Trains with this batch size.
                )

        # For validation the order doesn't matter, so we'll just read them sequentially.
        val_loader = DataLoader(
                        valid_dataset, # The validation samples.
                        sampler = SequentialSampler(valid_dataset), # Pull out batches sequentially.
                        batch_size = 32 # Evaluate with this batch size.
                    )

        model.fit(train_loader, val_loader, writing_file=bootstrap_path, seed=i, additional_features=ADDITIONAL_FEATURES, epochs=100)
            