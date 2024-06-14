import torch
import torch.nn as nn
from torch.nn import functional as F
from tiktoken import get_encoding as encode_tok

# BigramModel
class BigramModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
    
    def forward(self, idx, targets=None):
        # Simple Biagram model doesn't use any @
        logits = self.token_embedding_table(idx)

        if targets == None: # If generating
            loss = None
        
        else: # if training
            batch, time, channels = logits.shape
            targets = targets.view(batch*time)
            loss = F.cross_entropy(logits.view(batch*time, channels), targets)
        
        return logits, loss
    
    def generate(self, idx, max_tokens):
        for i in range(max_tokens):
            logits, loss = self(idx) # forward pass to get prediction

            logits = logits[:, -1, :] # get last column of matrix that contains all the different possible next tokens
            probs = F.softmax(logits, dim=-1) # get probabilities using softmax
            idx_next = torch.multinomial(probs, num_samples=1) #get 1 sample from the probabilities of next tokens

            # construct new token sequence for output/next iteration
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
