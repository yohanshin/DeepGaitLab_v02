import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
from ..modules.transformer_detr import TransformerDecoder, TransformerDecoderLayer


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


class DecoderPerLandmark(nn.Module):
    def __init__(self,
                 d_model=768,
                 n_heads=8,
                 n_layers=6,
                 n_landmarks=512,
                 transformer_dim_feedforward=2048,
                 ldmks_dim=2,
                 dropout=0.1,
                 uncertainty=True,
                 visibility=True,):

        super(DecoderPerLandmark, self).__init__()

        self.output_dim = ldmks_dim+1 if uncertainty else ldmks_dim
        self.visibility = visibility

        activation = 'gelu'
        self.return_intermediate_dec = True
        normalize_before = False
        self.query_embed = nn.Embedding(n_landmarks, d_model)

        decoder_layer = TransformerDecoderLayer(d_model, n_heads, transformer_dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder_detr = TransformerDecoder(decoder_layer, n_layers, decoder_norm,
                                          return_intermediate=self.return_intermediate_dec)
        self.landmarks = MLP(d_model, d_model, self.output_dim, 3)
        if self.visibility:
            self.vis_prob = nn.Linear(d_model, 1)


    def forward(self, src, pos_embed):
        patch_pos_embed = pos_embed[:, 1:]
        bs = src.shape[0]
        patch_pos_embed = patch_pos_embed.permute(1, 0, 2)
        query_embed = self.query_embed.weight
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)

        tgt = torch.zeros_like(query_embed)
        src = rearrange(src, 'B D H W -> (H W) B D')

        hs = self.decoder_detr(tgt, src, memory_key_padding_mask=None, pos=patch_pos_embed, query_pos=query_embed)
        hs = hs.transpose(1, 2) #, memory.permute(1, 2, 0).view(bs, c, h, w)

        # Construct predictions
        pred = dict(
            joints2d=self.landmarks(hs)[-1],
            visibility=self.vis_prob(hs)[-1] if self.visibility else None,
        )

        return pred