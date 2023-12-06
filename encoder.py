import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import math

@torch.no_grad()
def estimate_loss(model, eval_iters):
    # Put the model in eval mode here
    model.eval()

    losses = torch.zeros(eval_iters) # Initilize an array of tensor of zeros of size eval_iters
    iter = 0
    for xb, yb in test_dl:
        logits, loss = model(xb,yb)
        # Get the loss for this batch
        losses[iter] = loss
        iter += 1
        if iter == eval_iters:
          break
    model.train()
    return torch.sum(losses)/iter

class Head(nn.Module):
    """
    This class represents one head of self-attention
    """

    def __init__(self, d_model, d_head, dropout):
        super().__init__()
        # Map each key, query, or value in to a d_head dimensional model.
        self.W_K = nn.Linear(d_model, d_head, bias=False)
        self.W_Q = nn.Linear(d_model, d_head, bias=False)
        self.W_V = nn.Linear(d_model, d_head, bias=False)
        self.d_head = d_head
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # (B, T, d_model)
        # B = batch_size, T = block_size in the below
        B,T,d = x.shape
        # Get the key and query representations from the embedding x
        # (B,T,d_head)
        k = self.W_K(x)
        # (B,T,d_head)
        q = self.W_Q(x)
        # (B,T,d_head)
        v = self.W_V(x)

        # Compute attention scores, and get the new representations for this head

        # (B T, d_head) @ (B, d_head, T) = (B, T, T)
        # Multiply q by k and divide by the appropriate constant
        scores = (q @ torch.transpose(k, -2, -1)) / math.sqrt(self.d_head)

        # (B, T, T)
        # Apply softmax to the final dimension of scores
        a = nn.Softmax(dim=-1)(scores)

        # Apply dropout
        a = self.dropout(a)

        # Perform the weighted aggregation of the values
        # Using a and v, get the new representations
        # (B, T, T) @ (B, T, d_head) -> (B, T, d_head)
        out = a @ v

        # For each token, return the weighted sum of the values
        return out

class MultiHeadAttention(nn.Module):
    """
    Multiple heads of self-attention in parallel
    You can have just sequential code below
    """

    def __init__(self, num_heads, d_head, dropout, d_model):
        super().__init__()
        self.heads = nn.ModuleList([Head(d_model, d_head, dropout) for _ in range(num_heads)])
        # This is to project back to the dimension of d_model. In this case, it is just a learned linear map
        self.W_O = nn.Linear(d_head * num_heads, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Concatenate the different representations per head along the last dimension
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        # Project the concatenation and apply dropout; this is the W_O in "Attention is all you need"
        out = self.dropout(self.W_O(out))

        return out

class FeedFoward(nn.Module):
    """
    A simple linear layer followed by a non-linearity; this is applied at the token level
    """

    def __init__(self, d_model, dropout):
        super().__init__()
        d_ff = 4 * d_model
        # Map each token via a linear map to d_ff, apply ReLU, map back to d_model, and then apply dropout
        # This can be done with nn.Sequential
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.ff(x)

class EncoderBlock(nn.Module):
    """
    Transformer encoder block: communication followed by computation
    These are stacked on top of each other one after another
    """

    def __init__(self, d_model, n_head, dropout):
        super().__init__()
        # Each head gets a smaller dimensional representation of the data
        # Assume each head gets a representation of dimension d_head and d_model is divisible by n_head
        d_head = d_model // n_head
        self.sa = MultiHeadAttention(n_head, d_head, dropout, d_model)
        self.ff = FeedFoward(d_model, dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

class AttentionAligner(nn.Module):
    def __init__(self, step_size, d_model, block_size, n_head, n_layer, dropout):
        super().__init__()

        # Each token directly reads off the logits for the next token from a lookup table
        # Token embeddings are from vocab_size to d_model
        self.token_embedding_table = nn.Embedding(step_size, d_model)

        # Position embeddings are from block_size (T) to d_model
        self.position_embedding_table = nn.Embedding(block_size, d_model)

        # This should be n_layer applications of a EncoderBlock
        self.blocks = nn.Sequential(*[EncoderBlock(d_model, n_head, dropout) for _ in range(n_layer)])

        # Final layer norm
        self.ln = nn.LayerNorm(d_model)

        # Linear map to vocab size
        self.ff = nn.Linear(d_model, 1)

        self.step_size = step_size

    def forward(self, idx, targets=None):
        B, T = idx.shape
        
        # idx and targets are both (B,T) tensor of integers
        # (B,T,d_model)
        tok_emb = self.token_embedding_table(torch.clip(torch.floor((idx + 1) * self.step_size-1).int(), min=0, max=self.step_size-1))
        
        # (B,T,d_model)
        pos_emb = self.position_embedding_table(torch.arange(T, device="cuda"))

        # Add positional encodings to encodings
        # (B,T,d_model)
        x = tok_emb #+ pos_emb

        # Mix up the token representations over and over via the blocks
        # (B,T,d_model)
        x = self.blocks(x)

        # Apply layer norm
        # (B,T,d_model)
        x = self.ln(x)

        # Apply the final linear map
        logits = self.ff(x).squeeze(-1)
        
        if targets is None:
            loss = None
        else:
            loss = nn.L1Loss()(logits, targets)

        return logits, loss
