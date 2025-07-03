import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from einops import rearrange
from torch.nn import TransformerDecoder, TransformerDecoderLayer, TransformerEncoder, TransformerEncoderLayer

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        
    def initialize_last_layer(self):
        # Extract the last layer
        last_layer = self.layers[-1]
        nn.init.constant_(last_layer.weight, 0.0)  # Optional: Initialize the first weight to 0
        nn.init.uniform_(last_layer.bias[:2], a=-2, b=2)  # Optional: Initialize the first bias to 0
        last_layer.bias.data[:2].clamp_(-1, 1)  # Optional: Clamp the first bias to -1 and 1

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
    

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [5000, 1]: 0, 1, 2, ..., 4999
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))  # torch.arange(0, d_model, 2): [256]: 0, 2, 4, 6, 8, ..., 510  div_term: [256]
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # pe = pe.unsqueeze(0).transpose(0, 1)   # [5000, 1, 512]

        self.register_buffer('pe', pe.unsqueeze(1))
        # self.register_buffer('pe', pe)

    def forward(self, x):
        # not used in the final model
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)


class TimestepEmbedder(nn.Module):
    def __init__(self, latent_dim, dropout):
        super().__init__()
        self.latent_dim = latent_dim
        self.sequence_pos_encoder = PositionalEncoding(latent_dim, dropout)

        time_embed_dim = self.latent_dim
        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

    def forward(self, timesteps):
        return self.time_embed(self.sequence_pos_encoder.pe[timesteps]).permute(1, 0, 2)


class InputProcess(nn.Module):
    def __init__(self, input_feats, latent_dim):
        super().__init__()
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.poseEmbedding = nn.Linear(self.input_feats, self.latent_dim)

    def forward(self, x):
        x = rearrange(x, "b n d -> n b d")
        x = self.poseEmbedding(x)  # [L, bs, dim]
        return x
    



class SpatioTemporalEncoder(nn.Module):
    def __init__(self, 
                 n_landmarks=512, 
                 ldmks_dim=3,
                 window_size=64,
                 d_model=256, 
                 n_layers=6, 
                 n_heads=4, 
                 transformer_dim_feedforward=512, 
                 dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.transformer_dim_feedforward = transformer_dim_feedforward
        self.dropout = dropout

        self.mse_loss = nn.MSELoss(reduction='none')
        self.l1_loss = nn.L1Loss(reduction='none')
        
        self.spatio_embed = nn.Embedding(n_landmarks, d_model)
        self.temp_embed = nn.Embedding(window_size, d_model)
        self.time_embed = TimestepEmbedder(d_model, dropout)
        self.xt_embed = InputProcess(ldmks_dim, d_model)
        self.cond_embed = nn.Sequential(
            nn.Linear(ldmks_dim, d_model // 2), # Initial per-vertex projection
            nn.GELU(),
            nn.Linear(d_model // 2, d_model),  # Further projection
            nn.GELU(),
            # MaxPool1d will act on the M dimension, so input should be (B, D_model, M)
            nn.MaxPool1d(n_landmarks), # This reduces M to 1
            nn.Flatten(), # Flatten to (B, D_model)
            nn.Linear(d_model, d_model), # Final MLP to get to d_model
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )

        spatio_encoder_layer = TransformerEncoderLayer(d_model=d_model, 
                                                       nhead=n_heads, 
                                                       dim_feedforward=transformer_dim_feedforward, 
                                                       dropout=dropout, 
                                                       activation='gelu', 
                                                       norm_first=False, 
                                                       batch_first=False)
        self.spatio_encoder = nn.ModuleList(TransformerEncoder(spatio_encoder_layer, num_layers=1) for _ in range(n_layers))

        temporal_encoder_layer = TransformerEncoderLayer(d_model=d_model, 
                                                         nhead=n_heads, 
                                                         dim_feedforward=transformer_dim_feedforward, 
                                                         dropout=dropout, 
                                                         activation='gelu', 
                                                         norm_first=False, 
                                                         batch_first=False)
        self.temporal_encoder = nn.ModuleList(TransformerEncoder(temporal_encoder_layer, num_layers=1) for _ in range(n_layers))
        self.output_head = nn.Linear(d_model, ldmks_dim)

    def forward(self, batch, timesteps):
        bs = timesteps.shape[0]
        
        x_t = batch["x_t"]
        cond = batch["prompt"]
        
        time_emb = self.time_embed(timesteps)
        xt_emb = self.xt_embed.poseEmbedding(x_t)    # Shape B,T,L,D
        cond_emb = self.cond_embed[0:4](cond.squeeze(1))
        cond_emb = self.cond_embed[4](cond_emb.mT)
        cond_emb = self.cond_embed[5:](cond_emb)
        
        spatio_embed = self.spatio_embed.weight
        spatio_embed = spatio_embed.unsqueeze(1).repeat(1, bs, 1)
        spatio_embed = rearrange(spatio_embed, 'l b d -> b l d').unsqueeze(1)
        xt_emb = xt_emb + spatio_embed

        temporal_embed = self.temp_embed.weight[:xt_emb.shape[1]]
        temporal_embed = temporal_embed.unsqueeze(1).repeat(1, bs, 1)
        temporal_embed = rearrange(temporal_embed, 't b d -> b t d').unsqueeze(2)
        xt_emb = xt_emb + temporal_embed
        
        x_feats = xt_emb
        combined_cond = (time_emb.squeeze(0) + cond_emb).unsqueeze(1).unsqueeze(1)
        for spatio_encoder, temporal_encoder in zip(self.spatio_encoder, self.temporal_encoder):
            x_feats = x_feats + combined_cond
            x_feats = rearrange(x_feats, "b t l d -> l (b t) d")
            x_feats = spatio_encoder(x_feats)
            x_feats = rearrange(x_feats, "l (b t) d -> b t l d", b=bs)
            x_feats = x_feats + combined_cond
            x_feats = rearrange(x_feats, "b t l d -> t (b l) d")
            x_feats = temporal_encoder(x_feats)
            x_feats = rearrange(x_feats, "t (b l) d -> b t l d", b=bs)
            
        # Output Head
        predicteion = self.output_head(x_feats) 
        
        return predicteion

    def projection(self, pts3d, Ps):
        pts3d_hom = torch.cat((pts3d, torch.ones_like(pts3d[..., :1])), dim=-1)
        pts2d = torch.einsum('cij,blnj->lcni', Ps, pts3d_hom)
        pts2d = pts2d[..., :2] / pts2d[..., -1:]
        return pts2d
    
    def guide_2d_projection(self, batch, out, t, compute_grad='x_0'):
        with torch.enable_grad():
            if compute_grad == 'x_t':
                x_t = batch['x_t']
            elif compute_grad == 'x_0':
                x_t = out['pred_xstart']
            x_t = x_t.detach().requires_grad_()  # [bs, body_feat_dim, 1, T]
            
            Ks, Rs, Ts = batch['Ks'], batch['Rs'], batch['Ts']
            Ps = Ks @ torch.cat((Rs, Ts[..., None]), dim=-1)
            proj_ldmks = self.projection(x_t, Ps)
            observed_ldmks = batch['ldmks2d'][..., :2]
            
            loss_proj = self.l1_loss(proj_ldmks, observed_ldmks)
            loss_proj = loss_proj.mean()
            grad_proj = torch.autograd.grad([-loss_proj], [x_t])[0]
            x_t.detach()

        return grad_proj