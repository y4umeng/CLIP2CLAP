import torch
import torch.nn as nn

class AttentionAligner2(nn.Module):
    def __init__(self, d_model=96, block_size=512):
        super().__init__()

        self.ff1 = nn.Linear(1, d_model)
        self.ff2 = nn.Linear(d_model, 1)
        self.pos_emb = nn.Embedding(block_size, d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model, nhead=8, dim_feedforward=96)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=8)


    def forward(self, x, targets=None):
        x.unsqueeze_(-1)

        # B, T, C
        B, T, C = x.shape

        assert(T == 512)
        assert(C == 1)

        # B, 512, 1 
        x = self.ff1(x)

        # B, 512, d_model
        pos_emb = self.pos_emb(torch.arange(T, device="cuda"))

        # B, 512, d_model
        x = x + pos_emb

        x = self.transformer_encoder(x)

        logits = self.ff2(x).squeeze(-1)
        
        if targets is None:
            loss = None
        else:
            loss = torch.sum(torch.sqrt(torch.sum(torch.pow(torch.subtract(logits, targets), 2), dim=-1)))

        return logits, loss





