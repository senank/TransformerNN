import sys
import torch
import torch.nn as nn
from torch.nn import functional as F
from tiktoken import get_encoding as encode_tok

from model import BigramModel

# !wget dataset

# Constants
FILENAME = 'input.txt'
BLOCK_SIZE = 8
BATCH_SIZE = 4

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
    
    return file_data

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
    return torch.stack(x), torch.stack(y)


# Encoding tokens
def encode(input: str):
    output = []
    for ch in input:
        output.append(STOI[ch])
    return output

def decode(input: str):
    output = ''
    for i in input:
        output += ITOS[i]
    return output



if __name__ == '__main__':
    if len(sys.argv) == 1:
        file_data= load_data()
    else:
        file_data = load_data(sys.argv[1])
    
    # Using alternative encoders as easy as:
    enc = encode_tok('gpt2')
    # print(enc.decode(enc.encode('hello world')))

    # Divide data
    data = torch.tensor(encode(file_data), dtype=torch.long)
    dataTrain, dataVal, dataTest = get_data_sets(data)
    
    #training 
    xtrain, ytrain = get_minibatch(dataTrain)
    



    

    



