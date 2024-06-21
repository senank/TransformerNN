import sys
from typing import List
import torch
import torch.nn as nn
from torch.nn import functional as F
from tiktoken import get_encoding as encode_tok

from model import BigramModel, BLOCK_SIZE, BATCH_SIZE, n_emb, N_ATTENTION_LAYERS
from pdb import set_trace as DB

# !wget dataset

# Constants
FILENAME = 'input.txt'


TRAINING_ITERATIONS = 3000
EVAL_ITERATIONS = 100
EVAL_INTERVAL = 200
LEARNING_RATE = 0.5e-3
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# Encoding Character datastructures
STOI = {}
ITOS = {}

# Data processing
def extract_data(filename: str):
    with open(filename, 'r', encoding='utf-8') as f:
        text = f.read()
    return text

def load_data(filename=FILENAME):
    # load data set
    file_data = extract_data(filename)
    
    # get unique characters in input and fill encoding table
    chars = sorted(list(set(file_data)))
    vocab_size = len(chars)
    for i, ch in enumerate(chars):
        STOI[ch] = i
        ITOS[i] = ch
    
    return file_data, vocab_size

def get_data_sets(data):
    dataTrain = data[:int((len(data)*0.8))]
    dataVal = data[int((len(data)*0.8)):int((len(data)*0.9))]
    dataTest = data[int((len(data)*0.9)):]
    return dataTrain, dataVal, dataTest

def get_minibatch(data):
    ix = torch.randint(len(data)-BLOCK_SIZE, (BATCH_SIZE,))
    x=[]
    y=[]
    for i in ix:
        x.append(data[i:i+BLOCK_SIZE])
        y.append(data[i+1:i+BLOCK_SIZE+1])
    x, y = torch.stack(x), torch.stack(y)
    x, y = x.to(DEVICE), y.to(DEVICE)
    return x, y


# Encoding tokens
def encode(input: str):
    output = []
    for ch in input:
        output.append(STOI[ch])
    return output

def decode(input: List[int]):
    output = ''
    for i in input:
        output += ITOS[i]
    return output


# Training
def train_model(model, data, eval=False, dataVal=None):
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    for iteration in range(TRAINING_ITERATIONS):
        x, y = get_minibatch(data)
        logits, loss = model(x, y)
        if eval and (iteration % EVAL_INTERVAL == 0):
            losses = estimate_loss(model, dataTrain, dataVal)
            print("Iteration {} : Training set Loss = {}, Validation loss = {}".format(iteration, losses['train'], losses['val']))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

@torch.no_grad()
def estimate_loss(model, dataTrain, dataVal):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        if split == 'train':
            losses = torch.zeros(EVAL_ITERATIONS)
            for k in range(EVAL_ITERATIONS):
                x, y = get_minibatch(dataTrain)
                logits, loss = model(x, y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        else:
            losses = torch.zeros(EVAL_ITERATIONS)
            for k in range(EVAL_ITERATIONS):
                x, y = get_minibatch(dataVal)
                logits, loss = model(x, y)
                losses[k] = loss.item()
            out[split] = losses.mean()
    model.train()
    return out

# Testing
def test_loss(model, test_data):
    model.eval()
    x, y = get_minibatch(test_data)
    logits, loss = model(x, y)
    losses = torch.zeros(1)
    losses = loss.item()
    model.train()
    return losses

def test_generation(model, input_test, output_length):
    print(decode(model.generate(input_test, output_length)[0].tolist())) # must index [0] to pluck out from (1, T)
    print("\n")


# Main 
if __name__ == '__main__':
    train_or_gen = input("\nType 'T' for Train, or 'G' for Generate from pre-trained model\n")
    if train_or_gen not in ['T', 'G']:
        print("Please input a valid input: 'T' or 'G'\n")
        exit()
    
    
    if train_or_gen == 'T':
        seperate_data = input("\nWould you like to use your own data? (Y/N)\n")
        if seperate_data == "N":
            file_data, vocab_size = load_data()
        else:
            file_ = input("\nPlease type the filename below (e.g. input.txt, input.csv):\n")
            try:
                file_data, vocab_size = load_data(file_)
            except:
                print("\nError loading dataset... Exiting\n")
                exit()
        
        # Using alternative encoders as easy as:
        enc = encode_tok('gpt2')
        # print(enc.decode(enc.encode('hello world')))

        # Divide data
        data = torch.tensor(encode(file_data), dtype=torch.long)
        dataTrain, dataVal, dataTest = get_data_sets(data)
        
        
        
        # Defining Model 
        model = BigramModel(vocab_size, N_ATTENTION_LAYERS).to(DEVICE)
        
        # Pre-training generation
        print('\n########################\nPRE-training generation:\n########################')
        input_test = torch.zeros((1, 1), dtype=torch.long, device=DEVICE) # Define a (1, 1) tensor with value 0 for starting char
        test_generation(model, input_test, 500)

        # Training Model
        train_model(model, dataTrain, True, dataVal)
        print('\n########################\n   Training Phase\n########################')

        # Post-training generation
        print('\n########################\nPOST-training generation:\n########################')
        test_generation(model, input_test, 1000)

        test_val = test_loss(model, dataTest)
        print("TEST_SET LOSS = {}".format(test_val))
        save_weights = input("\n Would you like to save your trained weights? (Y/N)\n")
        if save_weights == 'Y':
            weight_name = input("Please name your file (this will have .pth as it's file extension): \n")
            torch.save(model.state_dict(), '{}.pth'.format(weight_name))
    elif train_or_gen == 'G':
        # Initialize the model
        file_data, vocab_size = load_data()
        model = BigramModel(vocab_size, N_ATTENTION_LAYERS).to(DEVICE)

        # Load the weights from the file
        model.load_state_dict(torch.load('model_weights_3000.pth'))
        input_test = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)\

        # Set the model to evaluation mode
        model.eval()
        length = input("\nPlease input the number of characters you would like to generate:\n")
        try:
            length = int(length)
        except:
            print("\nPlease input a valid integer\n")
            exit()
        print('\n########################\n      Generating \n########################')
        test_generation(model, input_test, length)
        model.train()
    
    



    

    



