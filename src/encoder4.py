import torch
import torch.nn as nn

class AttentionAligner4(nn.Module):
    def __init__(self, clip_embed, d_model=1):
        super().__init__()

        self.linear = nn.Linear(512, 512)
        self.ff1 = nn.Linear(1, d_model)
        self.ff2 = nn.Linear(d_model, 1)
        self.pos_emb = nn.Embedding(clip_embed, d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model, nhead=8, dim_feedforward=96)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=8)


    def forward(self, x):
        x.unsqueeze_(-1)

        # B, T, C
        B, T, C = x.shape

        assert(T == 512)
        assert(C == 1)

        # B, 512, d_model 
        x = self.ff1(x)

        # B, 512, d_model
        pos_emb = self.pos_emb(torch.arange(T, device="cuda"))

        # B, 512, d_model
        x = x + pos_emb

        x = self.transformer_encoder(x)

        logits = self.ff2(x).squeeze(-1)

        print(logits.shape)
        
        return logits





