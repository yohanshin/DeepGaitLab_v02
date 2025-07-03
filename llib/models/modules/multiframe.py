import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
from torch.nn import (TransformerDecoder, 
                      TransformerEncoder, 
                      TransformerEncoderLayer, 
                      TransformerDecoderLayer)

class MultiFrame(nn.Module):
    def __init__(self,
                 d_model=768,
                 n_heads=8,
                 n_layers=2,
                 n_global_tokens=16,
                 transformer_dim_feedforward=1024,
                 dropout=0.2,
                 width=24,
                 height=32,
                 max_num_frames=10):
        super(MultiFrame, self).__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.transformer_dim_feedforward = transformer_dim_feedforward
        self.dropout = dropout
        self.max_num_frames = max_num_frames

        self.temporal_embed = nn.Embedding(max_num_frames, d_model)
        # self.query_embed = nn.Embedding(n_global_tokens, d_model)
        self.query_embed = nn.Parameter(torch.randn(1, n_global_tokens, max_num_frames, d_model))
        self.bbox_embed = nn.Linear(3, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, height*width + 1, d_model))
        decoder_layer = TransformerDecoderLayer(d_model=d_model, 
                                                nhead=n_heads, 
                                                dim_feedforward=transformer_dim_feedforward,
                                                dropout=dropout, 
                                                batch_first=False)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer=decoder_layer, 
                                          num_layers=n_layers, 
                                          norm=decoder_norm)

        encoder_layer = TransformerEncoderLayer(d_model=d_model, 
                                                nhead=n_heads, 
                                                dim_feedforward=transformer_dim_feedforward,
                                                dropout=dropout, 
                                                batch_first=False)
        encoder_norm = nn.LayerNorm(d_model)
        self.encoder = TransformerEncoder(encoder_layer=encoder_layer, 
                                          num_layers=n_layers, 
                                          norm=encoder_norm)
        
    def forward(self, x, bbox, valid=None):
        # x: B, F, D, H, W
        # bbox: B, F, 3
        # valid: B, F
        
        B, F = bbox.shape[:2]
        # Reshape to (H*W, B*F, D) for transformer
        x = rearrange(x, "(B F) D H W -> (H W) (B F) D", B=B, F=F)
        
        # Get camera queries for each view
        temporal_queries = self.temporal_embed.weight[:F].unsqueeze(0).repeat(B, 1, 1)  # B, V, D
        bbox_embeddings = self.bbox_embed(bbox)  # B, F, D
        temporal_queries = temporal_queries + bbox_embeddings
        temporal_queries = rearrange(temporal_queries, "B F D -> (B F) D").unsqueeze(0)
        queries = self.query_embed.repeat(B, 1, 1, 1)
        queries = rearrange(queries, 'B L F D -> L (B F) D')
        # queries = self.query_embed.weight.unsqueeze(1).repeat(1, B*F, 1)

        # Add positional embedding
        pos_embed = self.pos_embed[:, 1:]  # 1, H*W, D
        x = x + pos_embed.permute(1, 0, 2)  # H*W, 1, D

        # Decode the global feature
        global_x = self.decoder(tgt=queries, memory=x)
        global_x = rearrange(global_x.squeeze(0), "L (B F) D -> (F L) B D", B=B, F=F)
        
        # Apply transformer encoder
        if valid is not None:
            valid = valid.bool().squeeze(-1)
            # Create padding mask for transformer
            # True values in mask indicate positions to mask out
            padding_mask = ~valid # B, F
            # Expand to match the L query tokens for each frame
            padding_mask = padding_mask.unsqueeze(1).repeat(1, queries.shape[0], 1)  # B, L, F
            padding_mask = rearrange(padding_mask, "B L F -> B (F L)", B=B, F=F)
        else:
            padding_mask = None
            
        # Apply transformer
        encoded_features = self.encoder(global_x, src_key_padding_mask=padding_mask)  # H*W, B*V, D
        
        # Reshape back to original format
        encoded_features = rearrange(encoded_features, "(F L) B D -> L (B F) D", F=F)
        
        return encoded_features
        