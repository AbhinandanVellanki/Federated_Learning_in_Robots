import torch
from torchtext import data
from torchtext import datasets
from transformers import BertTokenizer
import random
import numpy as np
from transformers import BertTokenizer, BertModel
import BERTGRUSentiment
import torch.optim as optim
import torch.nn as nn
import time

class Sentiment_Analyse():

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    #Set arbitrary tokens and indexes
    init_token = tokenizer.cls_token
    eos_token = tokenizer.sep_token
    pad_token = tokenizer.pad_token
    unk_token = tokenizer.unk_token
    #print(init_token, eos_token, pad_token, unk_token)
    init_token_idx = tokenizer.cls_token_id
    eos_token_idx = tokenizer.sep_token_id
    pad_token_idx = tokenizer.pad_token_id
    unk_token_idx = tokenizer.unk_token_id
    #print(init_token_idx, eos_token_idx, pad_token_idx, unk_token_idx)

    max_input_length = tokenizer.max_model_input_sizes['bert-base-uncased']
    #print("The max input length of Sentiment Analyser: ",max_input_length)


    #create instance of model
    #set model hyperparameters
    BATCH_SIZE = 128
    HIDDEN_DIM = 256
    OUTPUT_DIM = 1
    N_LAYERS = 2
    BIDIRECTIONAL = True
    DROPOUT = 0.25
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    bert = BertModel.from_pretrained('bert-base-uncased')
    model = BERTGRUSentiment.BERTGRU(bert, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS, BIDIRECTIONAL, DROPOUT)

    print("Model Created!")

    def tokenize_and_cut(self, sentence):
            tokens = self.tokenizer.tokenize(sentence) 
            tokens = tokens[:self.max_input_length-2]
            return tokens

    def count_parameters(self):
            return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def binary_accuracy(self, preds, y):
            #Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8

            #round predictions to the closest integer
            rounded_preds = torch.round(torch.sigmoid(preds))
            correct = (rounded_preds == y).float() #convert into float for division 
            acc = correct.sum() / len(correct)
            return acc
        
    def train(self, model, iterator, optimizer, criterion):
        epoch_loss = 0
        epoch_acc = 0
        self.model.train()
        for batch in iterator:
            optimizer.zero_grad()
            predictions = model(batch.text).squeeze(1)
            loss = criterion(predictions, batch.label)
            acc = self.binary_accuracy(predictions, batch.label)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_acc += acc.item()
        return epoch_loss / len(iterator), epoch_acc / len(iterator)

    def evaluate(self, model, iterator, criterion):
        epoch_loss = 0
        epoch_acc = 0
        self.model.eval()
        with torch.no_grad():
            for batch in iterator:
                predictions = model(batch.text).squeeze(1)
                loss = criterion(predictions, batch.label)
                acc = self.binary_accuracy(predictions, batch.label)
                epoch_loss += loss.item()
                epoch_acc += acc.item()
        return epoch_loss / len(iterator), epoch_acc / len(iterator)

    def epoch_time(self, start_time, end_time):
        elapsed_time = end_time - start_time
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
        return elapsed_mins, elapsed_secs

    def initialise_train(self):

        #create random seeds
        SEED = 1234
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.backends.cudnn.deterministic = True
        
        #Use torchtext.data to create dataset using random seeds
        TEXT = data.Field(batch_first = True,
                        use_vocab = False,
                        tokenize = self.tokenize_and_cut,
                        preprocessing = self.tokenizer.convert_tokens_to_ids,
                        init_token = self.init_token_idx,
                        eos_token = self.eos_token_idx,
                        pad_token = self.pad_token_idx,
                        unk_token = self.unk_token_idx)

        LABEL = data.LabelField(dtype = torch.float)

        train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)
        train_data, valid_data = train_data.split(random_state = random.seed(SEED))

        print(f"Number of training examples: {len(train_data)}")
        print(f"Number of validation examples: {len(valid_data)}")
        print(f"Number of testing examples: {len(test_data)}")
        print(vars(train_data.examples[6]))
        tokens = self.tokenizer.convert_ids_to_tokens(vars(train_data.examples[6])['text'])
        print(tokens)
        LABEL.build_vocab(train_data)
        print(LABEL.vocab.stoi)

        #Freeze some model parameters to increase training speed
        for name, param in self.model.named_parameters():                
            if name.startswith('bert'):
                param.requires_grad = False

        #Setup for training
        train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
            (train_data, valid_data, test_data), 
            batch_size = self.BATCH_SIZE, 
            device = self.device)
        optimizer = optim.Adam(self.model.parameters())
        criterion = nn.BCEWithLogitsLoss()
        self.model = self.model.to(self.device)
        criterion = criterion.to(self.device)
        N_EPOCHS = 5
        best_valid_loss = float('inf')

        #start training loop
        for epoch in range(N_EPOCHS):
            
            start_time = time.time()
            
            train_loss, train_acc = self.train(self.model, train_iterator, optimizer, criterion)
            valid_loss, valid_acc = self.evaluate(self.model, valid_iterator, criterion)
                
            end_time = time.time()
                
            epoch_mins, epoch_secs = self.epoch_time(start_time, end_time)
                
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(self.model.state_dict(), 'tut6-model.pt')
            
            print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
            print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')

    def predict_sentiment(self, sentence):
        self.model.load_state_dict(torch.load('./tut6-model.pt', map_location=torch.device('cpu')))
        self.model.eval()
        tokens = self.tokenizer.tokenize(sentence)
        tokens = tokens[:self.max_input_length-2]
        indexed = [self.init_token_idx] + self.tokenizer.convert_tokens_to_ids(tokens) + [self.eos_token_idx]
        tensor = torch.LongTensor(indexed).to(self.device)
        tensor = tensor.unsqueeze(0)
        prediction = torch.sigmoid(self.model(tensor))

        return prediction.item()
    
if __name__=='__main__':
    pred_sent = Sentiment_Analyse()
    #pred_sent.initialise_train()
    print(pred_sent.predict_sentiment("testing"))
        
