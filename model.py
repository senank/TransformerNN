import torch
import torch.nn as nn
from torch.nn import functional as F
from tiktoken import get_encoding as encode_tok
from norm import LayerNorm, BatchNorm

from pdb import set_trace as DB

### CONSTANTS ###
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Model Constants
BATCH_SIZE = 64
n_emb = 512
BLOCK_SIZE = 256
DROPOUT = 0.15

#Attention
N_ATTENTION_LAYERS = 6
HEAD_COUNT = 8 # n_emb / head_count = 64 dimensions for each head






### MODELS ###
# Main models
class BigramModel(nn.Module):
    def __init__(self, vocab_size: int, n_sa_layers: int):
        super().__init__()
        self.token_emb_table = nn.Embedding(vocab_size, n_emb)
        self.position_emb_table = nn.Embedding(BLOCK_SIZE, n_emb)

        # # Single head of attention
        # self.sa_heads = AttentionHead(n_emb)
        # self.feedforward = FeedForward(n_emb)
        
        # # Multiheaded attention
        # head_size = n_emb // HEAD_COUNT
        # self.sa_heads = MultiheadAttention(HEAD_COUNT, head_size)
        # self.feedforward = FeedForward(n_emb)
        
        # Attention block
        

        self.sa_blocks = nn.Sequential(
            *[Block(n_emb, HEAD_COUNT) for sa_layer in range(n_sa_layers)],
            nn.LayerNorm(n_emb)
        )
        
        self.lm_head = nn.Linear(n_emb, vocab_size)
        
        
    
    def forward(self, idx, targets=None):
        # Simple Biagram model doesn't use any @
        token_emb = self.token_emb_table(idx)
        pos_emb = self.position_emb_table(torch.arange(len(idx[0]), device=DEVICE))
        x = pos_emb + token_emb

        # # Pass through multi-headed attention block and feedforward layer
        # x = self.sa_heads(x)
        # x = self.feedforward(x)

        # Pass through multiple multi-headed attention blocks
        x = self.sa_blocks(x)

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

# Attention block
class AttentionHead(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_emb, head_size, bias=False)
        self.query = nn.Linear(n_emb, head_size, bias=False)
        self.value = nn.Linear(n_emb, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))
        self.dropout = nn.Dropout(DROPOUT)
    
    def forward(self, idx):
        B, T, C = idx.shape
        k = self.key(idx)
        q = self.query(idx)

        wei = (k @ q.transpose(1, 2)) * C**(-0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=2)
        wei = self.dropout(wei)

        v = self.value(idx)
        out = wei @ v
        return out

class MultiheadAttention(nn.Module):
    def __init__(self, head_count, head_size, n_emb):
        super().__init__()
        heads = []
        for i in range(head_count):
            heads.append(AttentionHead(head_size))
        self.heads = nn.ModuleList(heads)
        
        self.proj = nn.Linear(n_emb, n_emb) # Adding residual connections
        self.dropout = nn.Dropout(DROPOUT) # Adding dropout

    def forward(self, idx):
        ff_output = torch.cat([head(idx) for head in self.heads], dim=-1)
        residual_projections = self.proj(ff_output)
        return self.dropout(residual_projections)

class FeedForward(nn.Module):
    def __init__(self, n_emb):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(n_emb, n_emb * 4), # Note: * 4 for the residual connection
            nn.ReLU(),

            # Residual connection
            nn.Linear(n_emb * 4, n_emb) # Note: Can be added outside of Seq as its out layer similar to MultiheadAttention implementation
        )

    def forward(self, x):
        return self.layers(x)

class Block(nn.Module):
    def __init__(self, n_emb, n_heads):
        # n_emb: Embedding dimension, n_heads: number of heads in attention block
        super().__init__()
        head_size = n_emb // n_heads # Head size dynamically calculated based on n_emd:n_heads ratio
        self.sa = MultiheadAttention(n_heads, head_size, n_emb)
        self.ln1 = nn.LayerNorm(n_emb)
        
        self.ff = FeedForward(n_emb)
        self.ln2 = nn.LayerNorm(n_emb)

    def forward(self, x):
        # Adding to make these residual connections
        # x = x + self.sa(x) 
        # x = x + self.ff(x)
        x = x + self.sa(self.ln1(x)) 
        x = x + self.ff(self.ln2(x))
        return x
