import torch
import torch.nn as nn
from torch.nn import functional as F
from tiktoken import get_encoding as encode_tok

from pdb import set_trace as DB

n_emb = 32
BLOCK_SIZE = 8
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
HEAD_COUNT = 4
HEAD_SIZE = n_emb//HEAD_COUNT

# BigramModel
class BigramModel(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.token_emb_table = nn.Embedding(vocab_size, n_emb)
        self.position_emb_table = nn.Embedding(BLOCK_SIZE, n_emb)

        self.lm_head = nn.Linear(n_emb, vocab_size)
        self.sa_heads = MultiheadAttention(HEAD_COUNT, HEAD_SIZE)
        
    
    def forward(self, idx, targets=None):
        # Simple Biagram model doesn't use any @
        token_emb = self.token_emb_table(idx)
        pos_emb = self.position_emb_table(torch.arange(len(idx[0])))
        x = pos_emb + token_emb

        # Pass through multi-headed attention block
        x = self.sa_heads(x)
        
        # Pass through language model
        logits = self.lm_head(x)

        if targets == None: # If generating
            loss = None
        
        else: # if training
            batch, time, channels = logits.shape
            targets = targets.view(batch*time)
            loss = F.cross_entropy(logits.view(batch*time, channels), targets)
        
        return logits, loss
    
    def generate(self, idx, max_tokens: int):
        output = idx
        for i in range(max_tokens):
            logits, loss = self(idx) # forward pass to get prediction

            logits = logits[:, -1, :] # get last column of matrix that contains all the different possible next tokens
            probs = F.softmax(logits, dim=-1) # get probabilities using softmax
            idx_next = torch.multinomial(probs, num_samples=1) #get 1 sample from the probabilities of next tokens

            # construct new token sequence for output/next iteration
            output = torch.cat((output, idx_next), dim=1)
            # Pass up to the context window back for further output
            idx = output[0][-BLOCK_SIZE:].view(1, -1)
        return output

class AttentionHead(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_emb, head_size, bias=False)
        self.query = nn.Linear(n_emb, head_size, bias=False)
        self.value = nn.Linear(n_emb, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))
    
    def forward(self, idx):
        B, T, C = idx.shape
        k = self.key(idx)
        q = self.query(idx)

        wei = (k @ q.transpose(1, 2)) * C**(-0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=2)

        v = self.value(idx)
        out = wei @ v
        return out

class MultiheadAttention(nn.Module):
    def __init__(self, head_count, head_size):
        super().__init__()
        self.heads = []
        for i in range(head_count):
            self.heads.append(AttentionHead(head_size))
    
    def forward(self, idx):
        return torch.cat([head(idx) for head in self.heads], dim=-1)