import torch
from tiktoken import get_encoding as encode_tok

# !wget dataset

# Constants
FILENAME = 'input.txt'

# Encoding Character datastructures
STOI = {}
ITOS = {}

# Input processing
def extract_data(filename: str):
    with open(filename, 'r', encoding='utf-8') as f:
        text = f.read()
    return text

def get_data_sets(data):
    dataTrain = data[:int((len(data)*0.8))]
    dataVal = data[int((len(data)*0.8)):int((len(data)*0.9))]
    dataTest = data[int((len(data)*0.9)):]
    return dataTrain, dataVal, dataTest

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
    # load data set
    file_data = extract_data(FILENAME)
    
    # get unique characters in input and fill encoding table
    chars = sorted(list(set(file_data)))
    vocab_size = len(chars)
    for i, ch in enumerate(chars):
        STOI[ch] = i
        ITOS[i] = ch

    # Using alternative encoders as easy as:
    enc = encode_tok('gpt2')
    # print(enc.decode(enc.encode('hello world')))

    # Divide data
    data = torch.tensor(encode(file_data), dtype=torch.long)
    dataTrain, dataVal, dataTest = get_data_sets(data)
    print(dataTrain.shape)

    



