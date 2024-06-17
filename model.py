import torch
import torch.nn as nn
from torch.nn import functional as F
from tiktoken import get_encoding as encode_tok

from pdb import set_trace as DB

n_emb = 32
BLOCK_SIZE = 8
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# BigramModel
class BigramModel(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.token_emb_table = nn.Embedding(vocab_size, n_emb)
        self.lm_head = nn.Linear(n_emb, vocab_size)

        self.position_emb_table = nn.Embedding(BLOCK_SIZE, n_emb)
        
    
    def forward(self, idx, targets=None):
        # Simple Biagram model doesn't use any @
        token_emb = self.token_emb_table(idx)
        pos_emb = self.position_emb_table(torch.arange(len(idx[0])))
        
        x = pos_emb + token_emb
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
